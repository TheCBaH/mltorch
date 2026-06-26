(* Concrete evaluation: values are floats, indices are plain ints, an input is a
   real tensor. See .ai/native_compute_design.md §3. *)

include
  Semantics.SEMANTICS
    with type t = float
     and type 'role index = int
     and type input = Tensor.packed
