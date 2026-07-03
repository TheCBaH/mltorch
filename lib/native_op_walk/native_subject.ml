(* The subject of a native-op walk: a one-node graph plus the concrete input
   tensors it was synthesized with. [Native_verify] runs both the Direct and the
   Symbolic backend on this and checks they agree. [target] labels the op for the
   walk output (the [Walk.Op] passes only the subject to [verify]). *)

type t = {
  target : string;
  graph : Graph_ir.graph;
  inputs : (Tensor_id.t * Tensor.packed) list;
}
