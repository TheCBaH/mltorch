(** A seeded generator of placed graphs: a kernel of several values, each a
    pointwise function of earlier ones, so a value can feed several consumers.
    The sharing and structure gates run every such graph under its stored form
    and under the fused form the planner admits. A case is [(seed, length)]: the
    same pair regenerates the same kernel on every run. *)

type case = {
  seed : int64;
  kernel : Kernel.t;
  outputs : Tensor_id.t list;
      (** the kernel's outputs: the only values both forms store *)
  input : Tensor.packed;
}

val generate : seed:int64 -> length:int -> case
(** [length] values over one binary32 input of eight cells; the last is always
    an output and each earlier one is an output with probability one half. *)

val bind : case -> Tensor_id.t -> Tensor.packed option
(** The binding of the case's one input. *)
