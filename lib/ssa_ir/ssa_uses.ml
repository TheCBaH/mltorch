(* Definitions and uses of one program revision, with the region tree: the
   facts every pass starts from. Bound to the revision it was built for: a
   pass that returns a new revision builds its own, and [fresh] is how a
   consumer proves it is not reading a stale one. Nothing here is a claim about
   values, only about structure. *)

type position = {
  region : Ssa_id.Region.t;
  index : int;
      (** the defining statement within its region, -1 for a parameter *)
}

type t = {
  revision : Ssa_id.Revision.t;
  def : position Ssa_id.Value.Map.t;
  uses : int Ssa_id.Value.Map.t;
  parent : Ssa_id.Region.t option Ssa_id.Region.Map.t;
      (** the region a region is nested in, [None] for the entry *)
}

let bump m (v : Ssa_value.t) =
  Ssa_id.Value.Map.update v.Ssa_value.id
    (function None -> Some 1 | Some n -> Some (n + 1))
    m

let build (p : Ssa_program.t) =
  let def = ref Ssa_id.Value.Map.empty in
  let uses = ref Ssa_id.Value.Map.empty in
  let parent = ref Ssa_id.Region.Map.empty in
  let define region index (v : Ssa_value.t) =
    def := Ssa_id.Value.Map.add v.Ssa_value.id { region; index } !def
  in
  let use v = uses := bump !uses v in
  let rec region ~within (r : Ssa_region.t) =
    parent := Ssa_id.Region.Map.add r.Ssa_region.id within !parent;
    let id = r.Ssa_region.id in
    List.iter (define id (-1)) r.Ssa_region.params;
    List.iteri
      (fun index s ->
        (match s with
        | Ssa_stmt.Instr i ->
            List.iter use (Ssa_instr.operands i);
            List.iter (define id index) i.Ssa_instr.results
        | Ssa_stmt.For { lo; hi; inits; results; body; _ } ->
            use lo;
            use hi;
            List.iter use inits;
            List.iter (define id index) results;
            region ~within:(Some id) body
        | Ssa_stmt.If { cond; results; then_; else_ } ->
            use cond;
            List.iter (define id index) results;
            region ~within:(Some id) then_;
            region ~within:(Some id) else_
        | Ssa_stmt.Ordered_sum { lo; hi; seed; token; results; body } ->
            use lo;
            use hi;
            use seed;
            use token;
            List.iter (define id index) results;
            region ~within:(Some id) body);
        ())
      r.Ssa_region.body;
    List.iter use r.Ssa_region.yields
  in
  region ~within:None p.Ssa_program.entry;
  {
    revision = p.Ssa_program.revision;
    def = !def;
    uses = !uses;
    parent = !parent;
  }

let fresh t (p : Ssa_program.t) =
  Ssa_id.Revision.equal t.revision p.Ssa_program.revision

let use_count t (v : Ssa_value.t) =
  Option.value (Ssa_id.Value.Map.find_opt v.Ssa_value.id t.uses) ~default:0

let defined_at t (v : Ssa_value.t) =
  Ssa_id.Value.Map.find_opt v.Ssa_value.id t.def

(* Whether [region] is [ancestor] or nested in it. *)
let rec within t ~region ~ancestor =
  Ssa_id.Region.equal region ancestor
  ||
  match Ssa_id.Region.Map.find_opt region t.parent with
  | Some (Some up) -> within t ~region:up ~ancestor
  | Some None | None -> false

(* Lexical dominance: a definition is visible at a statement of [region] when it
   is a parameter of, or defined by an earlier statement of, [region] or an
   enclosing region. [index] is the position of the using statement. *)
let visible t (v : Ssa_value.t) ~region ~index =
  match defined_at t v with
  | None -> false
  | Some d ->
      if Ssa_id.Region.equal d.region region then d.index < index
      else within t ~region ~ancestor:d.region
