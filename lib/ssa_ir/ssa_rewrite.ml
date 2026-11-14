(* Rebuilding a program with a rule applied to every statement. A pass states
   what happens to one statement; this walks the regions, keeps the
   substitution of replaced values up to date, and returns the next revision.

   The walk is post-order: a statement's nested regions are rewritten before the
   statement itself is offered to the rule, so an outer decision (removing a loop
   whose body has just collapsed to one trip) sees the simplified body. The
   statements a rule returns are final for this walk; the pass driver repeats the
   walk when it wants a fixpoint. *)

type t = {
  subst : Ssa_value.t Ssa_id.Value.Tbl.t;
  mutable next_value : Ssa_id.Value.Next.t;
  mutable next_region : Ssa_id.Region.Next.t;
  mutable changed : bool;
}

(* Redirect every use of [from] to [to_] from here on. *)
let alias t ~(from : Ssa_value.t) ~(to_ : Ssa_value.t) =
  Ssa_id.Value.Tbl.replace t.subst from.Ssa_value.id to_;
  t.changed <- true

let fresh t ty =
  let id, next = Ssa_id.Value.Next.alloc t.next_value in
  t.next_value <- next;
  { Ssa_value.id; ty }

let mark_changed t = t.changed <- true

let rec resolve t (v : Ssa_value.t) =
  match Ssa_id.Value.Tbl.find_opt t.subst v.Ssa_value.id with
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

(* The origin a statement stands for: an instruction's own, or the first
   known one inside a structured statement. *)
let rec origin_of : Ssa_region.t Ssa_stmt.t -> Ssa_origin.t = function
  | Ssa_stmt.Instr i -> i.Ssa_instr.origin
  | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
      origin_in body
  | Ssa_stmt.If { then_; else_; _ } -> (
      match origin_in then_ with
      | Ssa_origin.Unknown -> origin_in else_
      | o -> o)

and origin_in (r : Ssa_region.t) =
  List.fold_left
    (fun acc s -> match acc with Ssa_origin.Unknown -> origin_of s | o -> o)
    Ssa_origin.Unknown r.Ssa_region.body

(* A rule that makes an instruction without a source gives it the origin of the
   statement it replaces: what a pass moves or rewrites keeps its origin, and a
   clone gets it too. An instruction with an origin of its own keeps it. *)
let inherit_origin origin (s : Ssa_region.t Ssa_stmt.t) =
  match (origin, s) with
  | Ssa_origin.Unknown, _ -> s
  | _, Ssa_stmt.Instr ({ Ssa_instr.origin = Ssa_origin.Unknown; _ } as i) ->
      Ssa_stmt.Instr { i with Ssa_instr.origin }
  | _ -> s

let program ?(enter = ignore) ?(leave = ignore) ?(enter_stmt = ignore)
    (rule : rule) (p : Ssa_program.t) =
  let t =
    {
      subst = Ssa_id.Value.Tbl.create 64;
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
    let origin = origin_of s in
    List.map (inherit_origin origin) (rule t s)
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
