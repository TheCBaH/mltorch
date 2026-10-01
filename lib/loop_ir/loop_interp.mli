(** The Loop IR reference interpreter. It exists before either text backend is
    trusted: the emitted JavaScript is checked against it, and it is checked
    against [Kernel_eval].

    Its errors are [Expr.Eval.error]'s rows, so a failure the Loop IR raises and
    the one [Kernel_eval] reports name the same kind and carry the same payload
    without a translation table between two vocabularies. *)

type error =
  [ Expr.Eval.error
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Unbound_input of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

type counters = {
  mutable emitters : int;
  mutable keys : int;
  mutable loads : int;
  mutable locals : int;
  mutable reductions : int;
  mutable scan_updates : int;
  mutable scans : int;
}
(** [Region_execution.counters]'s once-per-key evidence. [loads] counts every
    evaluated [Load]; the rest count [Mark] statements. *)

val counters : unit -> counters

val run :
  ?counters:counters ->
  ?outputs:(Tensor_id.t -> Tensor.packed option) ->
  Loop_program.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
(** Executes the program. Input buffers are validated against their signature
    ([Kernel_eval.check_binding]); an Output buffer is allocated fresh, unless
    [outputs] binds it, in which case it is validated the same way and
    zero-filled (a caller's slot may hold poison or a stale value, and Loop's
    accumulation semantics assume a fresh zero). Scratch is always fresh. The
    result holds exactly the Output buffers -- [outputs]' own tensor, when it
    bound one. Loops are iterative; expression recursion is bounded by the
    program's [max_depth]. An unchecked access out of range is a defect in the
    program and raises [Invalid_argument], never a typed failure: only an
    explicit [Fail_if] is one. *)

val allocate : Loop_buffer.t -> Tensor.packed
(** A zeroed tensor for an Output or Scratch buffer, of its declared shape and
    format. Shared with the in-process JavaScript executor so both allocate a
    result the same way. [Invalid_argument] for a format no lowered program
    stores. *)
