(* Concrete evaluation: values are floats, indices are ['role Dim.t] (see
   direct.ml), an input is a real tensor. See .ai/native_compute_design.md §3. *)

include
  Semantics.SEMANTICS
    with type t = float
     and type 'role index = 'role Dim.t
     and type input = Tensor.packed
