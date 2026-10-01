(* The lifetime script of every tensor a run touches, by storage role: what
   [Alloc_script] records for a node's outputs, plus the graph's constants and
   inputs, the boundaries of a run (model preparation, input population,
   result publication and release), and the logical arena each block is placed
   in (see .ai/ on the tensor arena). [Eval_direct.storage_script] produces it
   from the same fold a run follows.

   Events, in order (the order is behavior: it is the lifetime model):
   [Boundary Model_init], one [Alloc] per used constant in [g.inputs] order;
   [Boundary Input_population], one [Alloc] per graph input in [g.inputs] order,
   then the inputs no node reads freed; per non-sink node, its [Node] marker,
   its output allocations and its releases; [Boundary Result_publication];
   [Boundary Result_release], then a [Free] for every block the run still owns,
   in id order. Constants and borrowed payloads are never freed: they are not
   the run's to reclaim. A graph output or retained edge lives to
   [Result_release], whatever its role, so an input or constant forwarded as an
   output keeps the longest lifetime and is one block, never copied or freed
   because of its role. *)

open Graph_common
open Core.Storage_units

(** A logical arena. *)
module Arena_id : sig
  type t = Constants | Execution | Inputs | Intermediates | Outputs

  val all : t list
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit
end

module Boundary : sig
  type t = Input_population | Model_init | Result_publication | Result_release

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

(** Which logical arenas a run uses. [Separate]: constants, copied inputs,
    intermediates and outputs each in their own. [Shared_execution]: constants
    alone, and one arena for copied inputs, intermediates and outputs. *)
module Layout : sig
  type t = Separate | Shared_execution

  val pp : Format.formatter -> t -> unit
end

(** How a constant or graph input reaches a run. [Borrowed]: the caller's own
    payload, used in place and never reclaimed. [Copied]: copied into an arena
    slot, which only the copy occupies. *)
module Ownership : sig
  type t = Borrowed | Copied

  val pp : Format.formatter -> t -> unit
end

module Config : sig
  type t = { layout : Layout.t; constants : Ownership.t; inputs : Ownership.t }

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

module Role : sig
  type t = Constant | Input | Intermediate | Output

  val pp : Format.formatter -> t -> unit
end

module Block : sig
  type t = {
    alloc : Alloc_script.Alloc.t;
        (** Size, alignment and signature. Its [eligible] is [Alloc_script]'s
            and means nothing here: [arena] says where the block lives. *)
    role : Role.t;
    arena : Arena_id.t option;
        (** [None]: outside every arena: a borrowed payload, a quantized edge,
            or a fresh allocation. *)
  }

  val equal : t -> t -> bool
end

module Event : sig
  type t =
    | Alloc of Block.t
    | Boundary of Boundary.t
    | Free of Tensor_id.t
    | Node of Node_id.t

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

type t

val make : Config.t -> Alignment_policy.t -> Event.t list -> t
(** For [Eval_direct.storage_script]: the events are its fold's, unchecked. *)

val config : t -> Config.t
val policy : t -> Alignment_policy.t
val events : t -> Event.t list

val arena_of :
  Config.t -> Role.t -> quantized:bool -> kept:bool -> Arena_id.t option
(** The arena a block of [role] lives in under [config]; [kept]: it lives to
    [Result_release] (a graph output or retained edge, forwarded inputs
    included). A quantized block, and a borrowed constant or input, lives
    outside every arena: a pool holds one element kind and no quantization. *)

val arena_script : t -> Arena_id.t -> Alloc_script.t
(** The [Alloc_script] one arena's planner solves: every [Alloc], [Free] and
    [Node] event in order, boundaries dropped, a block [eligible] iff it lives
    in that arena. *)

val first_difference : t -> t -> Alloc_script.Position.t option
(** Where two scripts first differ, config and policy included (position 0 when
    those differ); [None] iff they are equal. *)

val peak_bytes :
  t ->
  where:(Block.t -> bool) ->
  (Byte_size.t, [> `Peak_bytes_overflow of Tensor_id.t ]) Err.t
(** The most bytes of the blocks [where] selects live at once. *)

val pp : Format.formatter -> t -> unit
