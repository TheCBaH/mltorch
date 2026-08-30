(* Pointwise ops: each output pixel reads the input(s) at the same coord. The
   functor is over SEMANTICS, so the same definition serves Direct and Symbolic.
   [out] is the output coord itself, one index expression per axis.
   See .ai/native_compute_design.md §2.

   This file is now a thin facade: unary ops live in pointwise_unary.ml,
   activation ops in pointwise_activation.ml, and the broadcasting helpers
   plus binary ops in pointwise_binary.ml. Every alias below is exactly what
   was a top-level binding in this file before the split, so external
   references such as [Pointwise.Add] or [Pointwise.Bin] are unaffected.. *)

let broadcast_output_shape = Pointwise_binary.broadcast_output_shape
let broadcast_coord = Pointwise_binary.broadcast_coord

module Clamp = Pointwise_unary.Clamp
module Clone = Pointwise_unary.Clone
module Expand = Pointwise_unary.Expand
module Sqrt = Pointwise_unary.Sqrt
module Hardsigmoid = Pointwise_activation.Hardsigmoid
module Hardswish = Pointwise_activation.Hardswish
module Hardtanh = Pointwise_activation.Hardtanh
module Relu = Pointwise_activation.Relu
module Sigmoid = Pointwise_activation.Sigmoid
module Silu = Pointwise_activation.Silu
module Gelu = Pointwise_activation.Gelu
module Binary = Pointwise_binary.Binary
module Bin = Pointwise_binary.Bin
module Scalar_bin = Pointwise_binary.Scalar_bin
module Scalar_binary = Pointwise_binary.Scalar_binary
module Add = Pointwise_binary.Add
module Add_scalar = Pointwise_binary.Add_scalar
module Div = Pointwise_binary.Div
module Div_scalar = Pointwise_binary.Div_scalar
module Mul = Pointwise_binary.Mul
module Mul_scalar = Pointwise_binary.Mul_scalar
module Pow = Pointwise_binary.Pow
module Sub = Pointwise_binary.Sub
