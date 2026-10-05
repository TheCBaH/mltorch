type error = Step_leaves_domain

let pp_error fmt = function
  | Step_leaves_domain ->
      Fmt.string fmt
        "a loop's last increment may leave the index domain, so it has no \
         in-domain form"

exception Refused of error

type builder = {
  mutable next_value : Ssa_id.Value.Next.t;
  mutable next_block : Ssa_id.Block.Next.t;
  mutable finished : Ssa_cfg_block.t list;
  (* the block under construction: its id, parameters and operations in
     reverse *)
  mutable id : Ssa_id.Block.t;
  mutable params : Ssa_value.t list;
  mutable body : Ssa_instr.t list;
  ranges : Ssa_range.t;
}

let fresh_value b ty =
  let id, next = Ssa_id.Value.Next.alloc b.next_value in
  b.next_value <- next;
  { Ssa_value.id; ty }

let fresh_block b =
  let id, next = Ssa_id.Block.Next.alloc b.next_block in
  b.next_block <- next;
  id

let emit b i = b.body <- i :: b.body

let pure b ty op =
  let v = fresh_value b ty in
  emit b
    { Ssa_instr.results = [ v ]; op; token = None; origin = Ssa_origin.Unknown };
  v

(* Ends the block under construction and starts the next. *)
let finish b terminator ~next ~params =
  b.finished <-
    {
      Ssa_cfg_block.id = b.id;
      params = b.params;
      body = List.rev b.body;
      terminator;
    }
    :: b.finished;
  b.id <- next;
  b.params <- params;
  b.body <- []

let edge target args = { Ssa_cfg_edge.target; args }
let index = Ssa_type.Scalar Ssa_type.Index

(* [iv + step] on the back edge is the one operation the lowering adds that
   carries a proof, so it is the one whose proof is checked here: the largest
   induction value plus the step stays in the domain. *)
let next_index b ~(iv : Ssa_value.t) ~step =
  (match Ssa_range.range b.ranges iv with
  | Ssa_range.Empty -> ()
  | Ssa_range.Range r ->
      if Int64.compare (Int64.add r.hi step) Ssa_const.index_max > 0 then
        raise (Refused Step_leaves_domain));
  let s = pure b index (Ssa_op.Const (Ssa_const.Index step)) in
  pure b index (Ssa_op.Index_add_in_domain (iv, s))

let lt b iv hi =
  pure b (Ssa_type.Scalar Ssa_type.Pred)
    (Ssa_op.Index_compare (Ssa_op.Compare.Lt, iv, hi))

let rec stmts b (ss : Ssa_region.t Ssa_stmt.t list) = List.iter (stmt b) ss

and stmt b : Ssa_region.t Ssa_stmt.t -> unit = function
  | Ssa_stmt.Instr i -> emit b i
  | Ssa_stmt.For { lo; hi; step; inits; results; body } ->
      let iv, carried =
        match body.Ssa_region.params with
        | iv :: carried -> (iv, carried)
        | [] -> invalid_arg "Ssa_cfg_lower: a loop without an induction value"
      in
      let header = fresh_block b
      and body_id = fresh_block b
      and exit = fresh_block b in
      finish b
        (Ssa_cfg_terminator.Jump (edge header (lo :: inits)))
        ~next:header ~params:(iv :: carried);
      let cond = lt b iv hi in
      finish b
        (Ssa_cfg_terminator.Branch
           { cond; then_ = edge body_id []; else_ = edge exit carried })
        ~next:body_id ~params:[];
      stmts b body.Ssa_region.body;
      let next = next_index b ~iv ~step in
      finish b
        (Ssa_cfg_terminator.Jump (edge header (next :: body.Ssa_region.yields)))
        ~next:exit ~params:results
  | Ssa_stmt.If { cond; results; then_; else_ } ->
      let then_id = fresh_block b
      and else_id = fresh_block b
      and join = fresh_block b in
      finish b
        (Ssa_cfg_terminator.Branch
           { cond; then_ = edge then_id []; else_ = edge else_id [] })
        ~next:then_id ~params:[];
      stmts b then_.Ssa_region.body;
      finish b
        (Ssa_cfg_terminator.Jump (edge join then_.Ssa_region.yields))
        ~next:else_id ~params:[];
      stmts b else_.Ssa_region.body;
      finish b
        (Ssa_cfg_terminator.Jump (edge join else_.Ssa_region.yields))
        ~next:join ~params:results
  | Ssa_stmt.Ordered_sum { lo; hi; seed; token; results; body } ->
      let iv, body_effect =
        match body.Ssa_region.params with
        | [ iv; e ] -> (iv, e)
        | _ -> invalid_arg "Ssa_cfg_lower: an ordered sum's parameters"
      in
      let acc = fresh_value b seed.Ssa_value.ty
      and enter = fresh_value b Ssa_type.Effect in
      let header = fresh_block b
      and body_id = fresh_block b
      and exit = fresh_block b in
      finish b
        (Ssa_cfg_terminator.Jump (edge header [ lo; seed; token ]))
        ~next:header ~params:[ iv; acc; enter ];
      let cond = lt b iv hi in
      finish b
        (Ssa_cfg_terminator.Branch
           {
             cond;
             then_ = edge body_id [ enter ];
             else_ = edge exit [ acc; enter ];
           })
        ~next:body_id ~params:[ body_effect ];
      stmts b body.Ssa_region.body;
      let term, effect_out =
        match body.Ssa_region.yields with
        | [ term; e ] -> (term, e)
        | _ -> invalid_arg "Ssa_cfg_lower: an ordered sum's yields"
      in
      (* the same left fold the structured form defines: a binary32 add rounds
         once, a vector adds lane by lane *)
      let add = Ssa_op.Float_binary (Expr.Value.Add, acc, term) in
      let sum =
        match seed.Ssa_value.ty with
        | Ssa_type.Vec _ | Ssa_type.Mask _ ->
            pure b seed.Ssa_value.ty (Ssa_op.Lanewise add)
        | Ssa_type.Effect | Ssa_type.Local | Ssa_type.Scalar _ ->
            pure b seed.Ssa_value.ty add
      in
      let next = next_index b ~iv ~step:1L in
      finish b
        (Ssa_cfg_terminator.Jump (edge header [ next; sum; effect_out ]))
        ~next:exit ~params:results

let program (p : Ssa_program.t) =
  let entry = p.Ssa_program.entry in
  let first = Ssa_id.Block.Next.first in
  let id, next_block = Ssa_id.Block.Next.alloc first in
  let b =
    {
      next_value = p.Ssa_program.next_value;
      next_block;
      finished = [];
      id;
      params = entry.Ssa_region.params;
      body = [];
      ranges = Ssa_range.analyze p;
    }
  in
  match
    stmts b entry.Ssa_region.body;
    match entry.Ssa_region.yields with
    | [ e ] -> finish b (Ssa_cfg_terminator.Return e) ~next:id ~params:[]
    | _ -> invalid_arg "Ssa_cfg_lower: the entry yields the effect alone"
  with
  | exception Refused e -> Error e
  | () ->
      Ok
        {
          Ssa_cfg.buffers = p.Ssa_program.buffers;
          entry = id;
          blocks = List.rev b.finished;
          scan_limits = p.Ssa_program.scan_limits;
          next_value = b.next_value;
        }
