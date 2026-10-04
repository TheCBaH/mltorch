(** Reading and writing the payload files (weights, inputs, outputs) of the
    whole-model backends. The layouts are {!C_payload_layout}'s, shared by the C
    and Wasm hosts; only the identity in a header differs. *)

open Graph_ir
open Loop_ir

val io : (unit -> 'a) -> ('a, [> `Io of string ]) result
val unix_io : (unit -> 'a) -> ('a, [> `Io of string ]) result
val mkdir_p : string -> unit

val write_payload :
  C_payload_layout.t ->
  identity:string ->
  path:string ->
  Tensor.packed list ->
  (unit, [> `Io of string ]) result
(** The header, zero padding, and each tensor at its entry's offset. *)

val bound_tensors :
  C_payload_layout.t ->
  lookup:(Tensor_id.t -> Tensor.packed option) ->
  missing:
    (Tensor_id.t ->
    ([> `Binding_mismatch of Kernel_eval.Binding_mismatch.t ] as 'e)) ->
  (Tensor.packed list, 'e) result
(** One tensor per entry, checked against the entry's signature. *)

val read_payload :
  C_payload_layout.t ->
  identity:string ->
  path:string ->
  (Tensor.packed list, [> `Bad_output of string | `Io of string ]) result
(** Validates the file's size and header, then decodes every entry. *)

val write_region :
  C_payload_layout.t ->
  identity:string ->
  Unix.file_descr ->
  base:int ->
  Tensor.packed list ->
  (unit, [> `Io of string ]) result
(** {!write_payload} into the region of an open file that starts at [base]. *)

val read_region :
  C_payload_layout.t ->
  identity:string ->
  Unix.file_descr ->
  base:int ->
  (Tensor.packed list, [> `Bad_output of string | `Io of string ]) result
(** {!read_payload} from such a region. *)
