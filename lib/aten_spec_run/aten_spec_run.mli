(* Run a decoded verification spec through the ATen-vs-native harness. The
   spec's tensor inputs are synthesized from its seed (one reproducible PCG
   stream), lowered to a single-node graph, and executed through the
   interpreter's ATen dispatch and, where the report compares, the native
   bridge. The input synthesis and the output-arity rules stay private. *)

val run : ?ppf:Format.formatter -> Aten_spec.Op_spec.t -> bool
(** Execute on both paths and report. [true] when the native and ATen outputs
    agreed, or the op has no native implementation (reported, and a pass);
    [false] on a mismatch or an error. *)

val eval_print : ?ppf:Format.formatter -> Aten_spec.Op_spec.t -> unit
(** Evaluate on both paths and print each output tensor, ATen above native, to
    be compared by eye. *)

val eval_report : ?ppf:Format.formatter -> Aten_spec.Op_spec.t -> unit
(** ATen only, no native comparison: the op's call, each output's shape and a
    status line. The promoted cram output is the golden, there being no PyTorch
    ground truth to compare against. *)

val walk_eval :
  ?ppf:Format.formatter -> steps:int -> Aten_spec.Op_spec.t -> unit
(** Redraw one tensor axis per step, in axis order, from its own distribution
    while every other axis keeps its previous value; the configuration never
    changes. Prints the modified axis, the configuration and a compact summary
    of the output. ATen only. *)

val compare_report : ?ppf:Format.formatter -> Aten_spec.Op_spec.t -> bool
(** [run] with a richer per-step report: the built configuration and the ATen
    output's shape and statistics before the status line. Suitable as a
    [Walk_core.Walk.run] [~verify]. *)
