(* Copying statements with fresh definitions. A clone has no definition in common
   with its original: every value a statement defines, every region parameter
   and every region id is new, and every use inside the copy follows the
   renaming, while a use of a value defined outside the copied statements stays
   what it was unless [subst] says otherwise. Fresh ids are drawn from the
   rewrite context, so a clone and its original can both live in one program. *)

type t = { context : Ssa_rewrite.t; renaming : (int, Ssa_value.t) Hashtbl.t }

let create context ~subst =
  let renaming = Hashtbl.create 32 in
  List.iter
    (fun ((from : Ssa_value.t), to_) ->
      Hashtbl.replace renaming (from.Ssa_value.id :> int) to_)
    subst;
  { context; renaming }

let value t (v : Ssa_value.t) =
  match Hashtbl.find_opt t.renaming (v.Ssa_value.id :> int) with
  | Some w -> w
  | None -> v

(* The new definition standing for [v]. *)
let define t (v : Ssa_value.t) =
  let w = Ssa_rewrite.fresh t.context v.Ssa_value.ty in
  Hashtbl.replace t.renaming (v.Ssa_value.id :> int) w;
  w

let fresh_region t =
  let id, next = Ssa_id.Region.Next.alloc t.context.Ssa_rewrite.next_region in
  t.context.Ssa_rewrite.next_region <- next;
  id

let rec stmt t : Ssa_region.t Ssa_stmt.t -> Ssa_region.t Ssa_stmt.t = function
  | Ssa_stmt.Instr i ->
      (* operands are read before the results are defined *)
      let op = Ssa_op.map_operands (value t) i.Ssa_instr.op in
      let token = Option.map (value t) i.Ssa_instr.token in
      let results = List.map (define t) i.Ssa_instr.results in
      Ssa_stmt.Instr { i with Ssa_instr.op; token; results }
  | Ssa_stmt.For f ->
      let lo = value t f.lo and hi = value t f.hi in
      let inits = List.map (value t) f.inits in
      let body = region t f.body in
      let results = List.map (define t) f.results in
      Ssa_stmt.For { f with lo; hi; inits; results; body }
  | Ssa_stmt.If f ->
      let cond = value t f.cond in
      let then_ = region t f.then_ in
      let else_ = region t f.else_ in
      let results = List.map (define t) f.results in
      Ssa_stmt.If { cond; results; then_; else_ }
  | Ssa_stmt.Ordered_sum f ->
      let lo = value t f.lo and hi = value t f.hi in
      let seed = value t f.seed and token = value t f.token in
      let body = region t f.body in
      let results = List.map (define t) f.results in
      Ssa_stmt.Ordered_sum { lo; hi; seed; token; results; body }

and region t (r : Ssa_region.t) =
  let id = fresh_region t in
  let params = List.map (define t) r.Ssa_region.params in
  let body = List.map (stmt t) r.Ssa_region.body in
  let yields = List.map (value t) r.Ssa_region.yields in
  { Ssa_region.id; params; body; yields }

let stmts t l = List.map (stmt t) l
