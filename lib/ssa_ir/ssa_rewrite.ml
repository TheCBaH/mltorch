(* Rebuilding a program with a rule applied to every statement. A pass states
   what happens to one statement; this walks the regions, keeps the
   substitution of replaced values up to date, and returns the next revision.

   The walk is post-order: a statement's nested regions are rewritten before the
   statement itself is offered to the rule, so an outer decision (removing a loop
   whose body has just collapsed to one trip) sees the simplified body. The
   statements a rule returns are final for this walk; the pass driver repeats the
   walk when it wants a fixpoint. *)

type t = {
  subst : (int, Ssa_value.t) Hashtbl.t;
  mutable next_value : Ssa_id.Value.Next.t;
  mutable next_region : Ssa_id.Region.Next.t;
  mutable changed : bool;
}

(* Redirect every use of [from] to [to_] from here on. *)
let alias t ~(from : Ssa_value.t) ~(to_ : Ssa_value.t) =
  Hashtbl.replace t.subst (from.Ssa_value.id :> int) to_;
  t.changed <- true

let fresh t ty =
  let id, next = Ssa_id.Value.Next.alloc t.next_value in
  t.next_value <- next;
  { Ssa_value.id; ty }

let mark_changed t = t.changed <- true

let rec resolve t (v : Ssa_value.t) =
  match Hashtbl.find_opt t.subst (v.Ssa_value.id :> int) with
  | Some w -> resolve t w
  | None -> v

let resolve_instr t (i : Ssa_instr.t) =
  {
    i with
    Ssa_instr.op = Ssa_op.map_operands (resolve t) i.Ssa_instr.op;
    token = Option.map (resolve t) i.Ssa_instr.token;
  }

(* The operands a statement reads before its regions run. *)
let resolve_head t : Ssa_region.t Ssa_stmt.t -> Ssa_region.t Ssa_stmt.t =
  function
  | Ssa_stmt.Instr i -> Ssa_stmt.Instr (resolve_instr t i)
  | Ssa_stmt.For f ->
      Ssa_stmt.For
        {
          f with
          lo = resolve t f.lo;
          hi = resolve t f.hi;
          inits = List.map (resolve t) f.inits;
        }
  | Ssa_stmt.If f -> Ssa_stmt.If { f with cond = resolve t f.cond }
  | Ssa_stmt.Ordered_sum f ->
      Ssa_stmt.Ordered_sum
        {
          f with
          lo = resolve t f.lo;
          hi = resolve t f.hi;
          seed = resolve t f.seed;
          token = resolve t f.token;
        }

(* Substitution into a statement and everything nested in it: for statements a
   pass moves into a new scope after the values they read have been aliased
   (a unit loop's body, once its parameters are known). *)
let rec resolve_deep t s =
  match resolve_head t s with
  | Ssa_stmt.Instr _ as s -> s
  | Ssa_stmt.For f -> Ssa_stmt.For { f with body = resolve_region t f.body }
  | Ssa_stmt.If f ->
      Ssa_stmt.If
        {
          f with
          then_ = resolve_region t f.then_;
          else_ = resolve_region t f.else_;
        }
  | Ssa_stmt.Ordered_sum f ->
      Ssa_stmt.Ordered_sum { f with body = resolve_region t f.body }

and resolve_region t (r : Ssa_region.t) =
  {
    r with
    Ssa_region.body = List.map (resolve_deep t) r.Ssa_region.body;
    yields = List.map (resolve t) r.Ssa_region.yields;
  }

type rule = t -> Ssa_region.t Ssa_stmt.t -> Ssa_region.t Ssa_stmt.t list

let program ?(enter = ignore) ?(leave = ignore) ?(enter_stmt = ignore)
    (rule : rule) (p : Ssa_program.t) =
  let t =
    {
      subst = Hashtbl.create 64;
      next_value = p.Ssa_program.next_value;
      next_region = p.Ssa_program.next_region;
      changed = false;
    }
  in
  let rec region (r : Ssa_region.t) =
    enter ();
    let body = List.concat_map stmt r.Ssa_region.body in
    leave ();
    {
      r with
      Ssa_region.body;
      yields = List.map (resolve t) r.Ssa_region.yields;
    }
  and stmt s =
    let s = resolve_head t s in
    enter_stmt s;
    let s =
      match s with
      | Ssa_stmt.Instr _ -> s
      | Ssa_stmt.For f -> Ssa_stmt.For { f with body = region f.body }
      | Ssa_stmt.If f ->
          let then_ = region f.then_ in
          let else_ = region f.else_ in
          Ssa_stmt.If { f with then_; else_ }
      | Ssa_stmt.Ordered_sum f ->
          Ssa_stmt.Ordered_sum { f with body = region f.body }
    in
    rule t s
  in
  let entry = region p.Ssa_program.entry in
  ( {
      p with
      Ssa_program.entry;
      revision = Ssa_id.Revision.succ p.Ssa_program.revision;
      next_value = t.next_value;
      next_region = t.next_region;
    },
    t.changed )
