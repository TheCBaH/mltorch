(** The producer's content digest of a named tensor map: for each tensor in
    order, the line [json.dumps([name, "torch.<dtype>", [dims]])] and a newline,
    then the tensor's raw little-endian row-major bytes, all through SHA-256. A
    file hash cannot stand in for it: it is how the logical tensors of a case
    are tied to the descriptor that names them, independent of how the [.pt]
    file happened to be serialized. *)

module Dtype = Pt2_checkpoint_map.Dtype

val torch_dtype_name : Dtype.t -> string
(** ["torch.float32"], ["torch.int64"], ["torch.bool"], ... *)

val preamble :
  string ->
  Dtype.t ->
  int64 list ->
  (string, [> `Digest_name of string ]) result
(** The line above, for a name that is plain ASCII; any other name is not
    reproduced (the producer escapes it, this reader does not need to). *)

val digest :
  (string * Logical.t) list ->
  (Pt2_sha256.Digest.t, [> `Digest_name of string ]) result
