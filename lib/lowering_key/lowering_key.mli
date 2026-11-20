(** What the lowering inspector can be asked to show.

    A key is a node plus a {!variant}; the variant carries its family, so a
    family with another family's axes is unrepresentable. Every closed type has
    [name]/[of_name] and the whole variant has one textual spelling,
    [family[:segment[/segment...]]], shared by the CLI flag, the wire and the
    UI. A missing segment is the default, "what runs by default": [name] always
    writes the segments that differ from the default and nothing else, so
    [of_name (name v) = Ok v] and equal variants spell identically. *)

module Dialect : sig
  (* @alphabetical *)
  type t = Native | Native4d

  val all : t list
  val name : t -> string
  val of_name : string -> t option
end

module Metric : sig
  (** The numbers an overlay can colour a node by. The names are the wire's. *)

  (* @alphabetical *)
  type t =
    | Coral_cycles  (** the Coral estimate of cycles *)
    | Scalar_loops  (** loops the planner tried and kept scalar *)
    | Vector_loops  (** loops the planner vectorized *)

  val all : t list
  val name : t -> string
  val of_name : string -> t option
end

module Frontend : sig
  (* @alphabetical *)
  type t = Direct | Structured

  val all : t list
  val name : t -> string
  val of_name : string -> t option
end

module Pipeline : sig
  (* @alphabetical *)
  type t = Exact | Planned | Representation

  val all : t list
  val name : t -> string
  val of_name : string -> t option
end

module Target : sig
  (* @alphabetical *)
  type t = Aarch64 | Riscv32 | Riscv32_grouped | X86_64

  val all : t list
  val name : t -> string
  val of_name : string -> t option
end

module Mir_stage : sig
  (* @alphabetical *)
  type t = Allocated | Generic | Realized | Scanned | Selected

  val all : t list
  val name : t -> string
  val of_name : string -> t option
end

module Coral_view : sig
  (* @alphabetical *)
  type t = Estimate | Rvv_kernel | Transfer_plan | Zvt_kernel

  val all : t list
  val name : t -> string
  val of_name : string -> t option
end

module Loop_spec : sig
  type t = {
    passes : Loop_ir.Loop_opt.Pass.t list;
        (** a sub-sequence of the fixed pipeline, in pipeline order, no
            duplicates ({!make} normalises) *)
    numerics : Loop_ir.Loop_numerics.t option;
        (** [None]: the family's own default (what its driver runs) *)
  }

  val make :
    ?numerics:Loop_ir.Loop_numerics.t -> Loop_ir.Loop_opt.Pass.t list -> t

  val default : t
  (** every pass, the family's default numerics *)
end

module Ssa_spec : sig
  type t = {
    frontend : Frontend.t;
    pipeline : Pipeline.t;
    numerics : Loop_ir.Loop_numerics.t option;
        (** meaningful for [Planned]; [None] is the family's default *)
  }

  val default : t
  (** [Structured], [Planned], default numerics *)
end

module Route : sig
  (* @alphabetical *)
  type t = Loop of Loop_spec.t | Ssa of Ssa_spec.t

  val default : t
  (** [Loop Loop_spec.default]: the drivers' emitter; SSA is their opt-in *)
end

module Mir_spec : sig
  type t = { target : Target.t; stage : Mir_stage.t }

  val default : t
  (** [X86_64], [Realized] *)
end

(* @alphabetical *)
type family = Asm | C | Coral | Js | Loop | Machine_ir | Ssa | Wat

(* @alphabetical *)
type variant =
  | Asm of Target.t
  | C of Route.t
  | Coral of Coral_view.t
  | Js of Route.t
  | Loop of Loop_spec.t
  | Machine_ir of Mir_spec.t
  | Ssa of Ssa_spec.t
  | Wat of Route.t

val families : family list

val reaches : Dialect.t -> family -> bool
(** Whether the family is offered for the dialect's graphs: the families the
    Native4D dialect has committed gates for (C, JavaScript, Loop, SSA and
    WebAssembly) and every family for Native. The rest are refused rather than
    shown unchecked. *)

val family_name : family -> string
val family_of_name : string -> family option
val family : variant -> family

val default : family -> variant
(** the variant a bare [family] name means *)

(** Why a spelling was refused. *)
module Parse_error : sig
  (* @alphabetical *)
  type t =
    | Duplicate_segment of { family : family; segment : string }
    | Empty_segment of family
    | Unknown_family of string
    | Unknown_segment of { family : family; segment : string }

  val pp : Format.formatter -> t -> unit
end

val name : variant -> string
(** [family] alone for the default, else [family:segments]. *)

val of_name : string -> (variant, Parse_error.t) result
val pp : Format.formatter -> variant -> unit

module Node_id = Graph_ir.Node_id

type t = { dialect : Dialect.t; node : Node_id.t; variant : variant }

val make : ?dialect:Dialect.t -> Node_id.t -> variant -> t
val equal : t -> t -> bool
val pp_key : Format.formatter -> t -> unit
