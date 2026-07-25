(* Pure lowering of the static tensor subset of an ExportedProgram. *)

type error =
  [ `Unsupported_input of string
  | `Unsupported_operator of string
  | `Malformed_graph of string
  | `Tensor_bridge of string
  | `Eval of Eval_direct.error
  | `Build of Graph_builder.error
  | `Provenance of Pt2_native_graph.error ]

val pp_error : Format.formatter -> [< error ] -> unit

(* Lowers a root exported graph into one native graph.  PT2 SSA names remain
   solely in the provenance wrapper; native execution addresses every edge by
   [Tensor_id].  This first static slice covers the ResNet-18 export set. *)
val lower :
  Pytorch_types.ExportedProgram.t -> (Pt2_native_graph.t, error) Core.result

val lower_archive : Pt2_archive.t -> (Pt2_native_graph.t, error) Core.result

(* Execute a one-user-input static graph.  Captured tensor payloads are loaded
   through the sidecar's [Tensor_id -> target] map, never through native IR. *)
val run :
  Pt2_archive.t -> input:Pt2_tensor.t -> (Tensor.packed list, error) Core.result
