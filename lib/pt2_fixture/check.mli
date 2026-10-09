(** The comparisons every layer repeats, each naming the layer it concerns. *)

module Pin = Pt2_checkpoint_map.Document.Pin

val field :
  Fault.layer ->
  Fault.field ->
  actual:string ->
  expected:string ->
  (unit, [> Fault.error ]) Err.t

val pin : Fault.layer -> Pin.t -> Pin.t -> (unit, [> Fault.error ]) Err.t
(** Equal in every field; reports the digest, then the size, then the rest. *)

val digest :
  Fault.layer ->
  Pt2_sha256.Digest.t ->
  Pt2_sha256.Digest.t ->
  (unit, [> Fault.error ]) Err.t

val size : Fault.layer -> int64 -> int64 -> (unit, [> Fault.error ]) Err.t
