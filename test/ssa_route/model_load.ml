open Loop_ir
module G = Graph_ir

(* A .pt2 as the route gate runs it: the canonical graph, the bundle built from
   it and the archive's own constants (captures materialized). *)

let err pp x = Err.or_raise ~pp_error:pp x

let load ?config path =
  let archive = err Pt2_archive.pp_error (Pt2_archive.open_pt2 path) in
  let lowered =
    err Native_interp.pp_error (Native_interp.lower_archive archive)
  in
  let constants =
    err Native_interp.pp_error (Native_interp.preload archive lowered)
  in
  let (Native_interp.Transformed t) =
    err Native_interp.pp_error
      (Native_interp.transform_lowered ~constants lowered
         ~passes:[ Pipeline.canonical ~fold:true ])
  in
  let store, _ =
    err Const_ssa_materialize.pp_error
      (Const_ssa_materialize.materialize
         (Native_interp.capture_resolver archive)
         t.constant_store)
  in
  let constants =
    G.Tensor_id.Map.union
      (fun _ _ v -> Some v)
      t.constants
      (Constant_store.materialized store)
  in
  (err Loop_bundle.pp_error (Loop_bundle.build ?config t.graph), constants)
