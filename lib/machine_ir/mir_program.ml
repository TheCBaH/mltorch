(* A whole Machine IR program: functions, the memory objects they address,
   helper requirements, the data model and the planning summary its numerics
   were resolved under. [revision] identifies this immutable snapshot and is
   never part of a semantic hash or the normalized text. *)
type ('op, 'term) t = {
  data_model : Mir_layout.Data_model.t;
  regions : Mir_region.t list;
  views : Mir_view.t list;
  helpers : Mir_helper.t list;
  funcs : ('op, 'term) Mir_func.t list;
  main : Mir_id.Func.t;
  planning : Mir_planning.t option;
  revision : Mir_id.Revision.t;
}

let find_func t id =
  List.find_opt
    (fun (f : (_, _) Mir_func.t) -> Mir_id.Func.equal f.Mir_func.id id)
    t.funcs

let find_view t id =
  List.find_opt
    (fun (v : Mir_view.t) -> Mir_id.View.equal v.Mir_view.id id)
    t.views

let find_region t id =
  List.find_opt
    (fun (r : Mir_region.t) -> Mir_id.Region.equal r.Mir_region.id id)
    t.regions

let find_helper t id =
  List.find_opt
    (fun (h : Mir_helper.t) -> Mir_id.Helper.equal h.Mir_helper.id id)
    t.helpers

(* The generic stage's instantiation. *)
type generic = (Mir_op.t, Mir_terminator.t) t
