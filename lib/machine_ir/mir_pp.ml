(* The deterministic printer. Names are assigned by traversal — blocks in
   reverse postorder from the entry, values in definition order within them —
   so two builder histories that produce the same program print the same text.
   The revision id never appears. A stage prints its own opcodes and
   terminators through the same naming. *)

module Names = struct
  type t = { blocks : int Mir_id.Block.Map.t; values : int Mir_id.Value.Map.t }

  let block t fmt b =
    match Mir_id.Block.Map.find_opt b t.blocks with
    | Some n -> Fmt.pf fmt "bb%d" n
    | None -> Fmt.pf fmt "?%a" Mir_id.Block.pp b

  let value t fmt (v : Mir_value.t) =
    let sigil =
      if Mir_type.equal v.Mir_value.ty Mir_type.Order then "!" else "%"
    in
    match Mir_id.Value.Map.find_opt v.Mir_value.id t.values with
    | Some n -> Fmt.pf fmt "%s%d" sigil n
    | None -> Fmt.pf fmt "%s?%a" sigil Mir_id.Value.pp v.Mir_value.id

  let values t fmt vs = Fmt.(list ~sep:(any ", ") (value t)) fmt vs
end

(* Blocks in printing order: reachable ones in reverse postorder, then the rest
   in list order. *)
let block_order ~edges (f : ('op, 'term) Mir_func.t) =
  let g =
    Mir_graph.make ~entry:f.Mir_func.entry
      (List.map
         (fun (b : ('op, 'term) Mir_block.t) ->
           ( b.Mir_block.id,
             List.map
               (fun (e : Mir_edge.t) -> e.Mir_edge.target)
               (edges b.Mir_block.terminator) ))
         f.Mir_func.blocks)
  in
  let reachable =
    List.filter_map (fun id -> Mir_func.find_block f id) g.Mir_graph.rpo
  in
  reachable
  @ List.filter
      (fun (b : ('op, 'term) Mir_block.t) ->
        not (Mir_graph.reachable g b.Mir_block.id))
      f.Mir_func.blocks

let names ~edges (f : ('op, 'term) Mir_func.t) =
  let blocks = block_order ~edges f in
  let bmap, _ =
    List.fold_left
      (fun (m, n) (b : ('op, 'term) Mir_block.t) ->
        (Mir_id.Block.Map.add b.Mir_block.id n m, n + 1))
      (Mir_id.Block.Map.empty, 0)
      blocks
  in
  let vmap = ref Mir_id.Value.Map.empty and next = ref 0 in
  let name (v : Mir_value.t) =
    if not (Mir_id.Value.Map.mem v.Mir_value.id !vmap) then (
      vmap := Mir_id.Value.Map.add v.Mir_value.id !next !vmap;
      incr next)
  in
  List.iter
    (fun (b : ('op, 'term) Mir_block.t) ->
      List.iter name b.Mir_block.params;
      name b.Mir_block.order;
      List.iter
        (fun (i : 'op Mir_instr.t) ->
          List.iter name i.Mir_instr.results;
          Option.iter (fun o -> name o.Mir_order.output) i.Mir_instr.order)
        b.Mir_block.body)
    blocks;
  (blocks, { Names.blocks = bmap; values = !vmap })

let pp_typed n fmt (v : Mir_value.t) =
  Fmt.pf fmt "%a: %a" (Names.value n) v Mir_type.pp v.Mir_value.ty

let pp_edge n fmt (e : Mir_edge.t) =
  Fmt.pf fmt "%a(%a; %a)" (Names.block n) e.Mir_edge.target (Names.values n)
    e.Mir_edge.args (Names.value n) e.Mir_edge.order

let pp_op n fmt (op : Mir_op.t) =
  let v = Names.value n and vs = Names.values n in
  let name = Mir_op.name op in
  match op with
  | Mir_op.Addr view -> Fmt.pf fmt "addr %a" Mir_id.View.pp view
  | Mir_op.Undef view -> Fmt.pf fmt "undef %a" Mir_id.View.pp view
  | Mir_op.Bitcast (ty, a) -> Fmt.pf fmt "bitcast.%a %a" Mir_type.pp ty v a
  | Mir_op.Call (c, args) -> Fmt.pf fmt "call %a(%a)" Mir_op.Callee.pp c vs args
  | Mir_op.Const c -> Fmt.pf fmt "const %a" Mir_const.pp c
  | Mir_op.Event (e, k) -> Fmt.pf fmt "event %s x%Ld" (Mir_event.name e) k
  | Mir_op.Load { Mir_op.Access.addr; align; _ } ->
      Fmt.pf fmt "%s [%a] align %Ld" name v addr align
  | Mir_op.Store ({ Mir_op.Access.addr; align; _ }, x) ->
      Fmt.pf fmt "%s [%a], %a align %Ld" name v addr v x align
  | Mir_op.Copy _ | Mir_op.Fbinary _ | Mir_op.Fcmp _ | Mir_op.Fconvert _
  | Mir_op.Ffma _ | Mir_op.Fto_sint _ | Mir_op.Funary _ | Mir_op.Iarith _
  | Mir_op.Icmp _ | Mir_op.Idiv _ | Mir_op.Iext _ | Mir_op.Itrunc _
  | Mir_op.Narrow _ | Mir_op.Pbinary _ | Mir_op.Pnot _ | Mir_op.Ptr_add _
  | Mir_op.Select _ ->
      Fmt.pf fmt "%s %a" name vs (Mir_op.operands op)

let pp_term n fmt = function
  | Mir_terminator.Branch { Mir_branch.cond; then_; else_ } ->
      Fmt.pf fmt "branch %a, %a, %a" (Names.value n) cond (pp_edge n) then_
        (pp_edge n) else_
  | Mir_terminator.Fail { Mir_fail.failure; payload; order } ->
      Fmt.pf fmt "fail %a(%a; %a)" Mir_failure.pp failure (Names.values n)
        payload (Names.value n) order
  | Mir_terminator.Jump e -> Fmt.pf fmt "jump %a" (pp_edge n) e
  | Mir_terminator.Return { Mir_return.values; order } ->
      Fmt.pf fmt "return %a; %a" (Names.values n) values (Names.value n) order

let pp_instr ~op ~origins n fmt (i : 'op Mir_instr.t) =
  let results = i.Mir_instr.results in
  let outs =
    List.map (Fmt.str "%a" (Names.value n)) results
    @
    match i.Mir_instr.order with
    | Some o -> [ Fmt.str "%a" (Names.value n) o.Mir_order.output ]
    | None -> []
  in
  if outs <> [] then Fmt.pf fmt "%s = " (String.concat ", " outs);
  op n fmt i.Mir_instr.op;
  (match i.Mir_instr.order with
  | Some o -> Fmt.pf fmt ", %a" (Names.value n) o.Mir_order.input
  | None -> ());
  if origins then Fmt.pf fmt "  ; %a" Mir_origin.pp i.Mir_instr.origin

let pp_func ?(origins = false) ~op ~term ~edges fmt
    (f : ('op, 'term) Mir_func.t) =
  let blocks, n = names ~edges f in
  Fmt.pf fmt "func %s -> (%a) {" f.Mir_func.name
    Fmt.(list ~sep:(any ", ") Mir_type.pp)
    f.Mir_func.results;
  List.iter
    (fun (b : ('op, 'term) Mir_block.t) ->
      Fmt.pf fmt "@,%a(%a; %a):" (Names.block n) b.Mir_block.id
        Fmt.(list ~sep:(any ", ") (pp_typed n))
        b.Mir_block.params (Names.value n) b.Mir_block.order;
      List.iter
        (fun i -> Fmt.pf fmt "@,  %a" (pp_instr ~op ~origins n) i)
        b.Mir_block.body;
      Fmt.pf fmt "@,  %a" (term n) b.Mir_block.terminator)
    blocks;
  Fmt.pf fmt "@,}"

let pp_region fmt (r : Mir_region.t) =
  Fmt.pf fmt "%a: %Ld bytes align %Ld %s" Mir_id.Region.pp r.Mir_region.id
    r.Mir_region.size r.Mir_region.align
    (Mir_region.init_name r.Mir_region.init)

let pp_view fmt (v : Mir_view.t) =
  Fmt.pf fmt "%a: %a[%Ld, +%Ld) %s %s%a" Mir_id.View.pp v.Mir_view.id
    Mir_id.Region.pp v.Mir_view.region v.Mir_view.offset v.Mir_view.size
    (Mir_view.perm_name v.Mir_view.perm)
    (Mir_view.role_name v.Mir_view.role)
    Fmt.(option (any " " ++ Expr.Source.pp))
    v.Mir_view.source

let pp_helper fmt (h : Mir_helper.t) =
  Fmt.pf fmt "%a: %s v%d (%a) -> (%a) %s" Mir_id.Helper.pp h.Mir_helper.id
    h.Mir_helper.name h.Mir_helper.version
    Fmt.(list ~sep:(any ", ") Mir_type.pp)
    h.Mir_helper.params
    Fmt.(list ~sep:(any ", ") Mir_type.pp)
    h.Mir_helper.results
    (Mir_helper.Effect.name h.Mir_helper.effects)

let pp_program_with ?origins ~op ~term ~edges fmt
    (p : ('op, 'term) Mir_program.t) =
  Fmt.pf fmt "@[<v>data_model %s"
    (Mir_layout.Data_model.name p.Mir_program.data_model);
  (match p.Mir_program.planning with
  | Some s ->
      Fmt.pf fmt "@,planning %s"
        (String.concat " "
           (String.split_on_char '\n' (Mir_planning.to_string s)))
  | None -> Fmt.pf fmt "@,planning none");
  List.iter (fun r -> Fmt.pf fmt "@,%a" pp_region r) p.Mir_program.regions;
  List.iter (fun v -> Fmt.pf fmt "@,%a" pp_view v) p.Mir_program.views;
  List.iter (fun h -> Fmt.pf fmt "@,%a" pp_helper h) p.Mir_program.helpers;
  List.iter
    (fun (f : ('op, 'term) Mir_func.t) ->
      let main =
        if Mir_id.Func.equal f.Mir_func.id p.Mir_program.main then " main"
        else ""
      in
      Fmt.pf fmt "@,%a%s %a" Mir_id.Func.pp f.Mir_func.id main
        (pp_func ?origins ~op ~term ~edges)
        f)
    p.Mir_program.funcs;
  Fmt.pf fmt "@]"

let generic ?origins fmt (p : Mir_program.generic) =
  pp_program_with ?origins ~op:pp_op ~term:pp_term ~edges:Mir_terminator.edges
    fmt p
