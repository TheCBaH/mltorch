(** Where every tensor of a whole-model C run lives.

    The workspace is one aligned byte region: the arena pools the storage plan
    placed (one per arena and element kind), then a scratch region shared by
    every invocation in turn. Weights and inputs are the mapped payload files,
    used in place. All sizes and offsets are [int64] and checked; nothing here
    allocates or maps anything. *)

open Graph_ir

type location =
  | Inputs of int64  (** byte offset from the start of the inputs file *)
  | Weights of int64
  | Workspace of int64  (** byte offset from the start of the workspace *)

type error =
  [ `Config_unsupported of Storage_script.Config.t
    (** only [Separate] layout with borrowed constants and inputs is admitted *)
  | `Edge_unplaced of Tensor_id.t
  | `Slot_out_of_pool of Tensor_id.t
  | `Storage_units of Core.Storage_units.error
  | `Synthetic_not_f32 of Tensor_id.t
  | `Workspace_overflow ]

val pp_error : Format.formatter -> [< error ] -> unit

(** What one invocation needs beyond the arena: its [Loop_stmt.Alloc] arrays
    ([local_doubles] of them, at the start), then a region for every buffer that
    is neither a graph edge nor an arena slot. *)
module Scratch : sig
  type fill = Zero | Value of float

  type carve = { position : int; offset : int64; bytes : int64; fill : fill }
  (** [position] indexes the program's buffers; [offset] is from the start of
      the scratch region. *)

  type t = { local_bytes : int64; carves : carve list; bytes : int64 }
end

val scratch :
  Loop_bundle.invocation -> local_doubles:int64 -> (Scratch.t, [> error ]) Err.t

type t

val create :
  Loop_bundle.t ->
  weights:C_payload_layout.t ->
  inputs:C_payload_layout.t ->
  scratch_bytes:int64 ->
  (t, [> error ]) Err.t
(** [scratch_bytes] is the largest {!Scratch.t.bytes} over the invocations:
    invocations run one after another and each re-initialises what it uses. *)

val locate : t -> Tensor_id.t -> (location, [> error ]) Err.t

val scratch_offset : t -> int64
(** Where the scratch region starts in the workspace. *)

val bytes : t -> int64
(** The whole workspace, scratch included. *)

val alignment : t -> int64
(** The base alignment the workspace needs: every offset above is a multiple of
    it or of the alignment its own tensor needs, both satisfied by a base
    aligned to this. *)

val pools :
  t -> (Storage_script.Arena_id.t * Alloc_script.Kind.t * int64 * int64) list
(** Arena, kind, workspace offset and bytes of every pool, for reports. *)

val graph_arena_bytes : t -> int64
(** The pools alone: the figure the storage plan reports, without scratch. *)
