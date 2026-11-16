(** A convolution whose windows do not overlap, as one RVV (Zve32f) function
    under the ILP32F calling convention: [a0] holds x's address (NHWC, one
    image), [a1] the packed weights, [a2] the bias (one value per output
    channel), [a3] y's address (NHWC). The function returns [0] in [a0] and
    writes y alone. It covers the stride-2, one-tap case and every other window
    no wider than its stride.

    The packing is the kernel's contract, made once ahead of time: the weights
    are channel-major, then kernel row, kernel column, then the output channel,
    so a tap's weights for every output channel are one vector load.

    Every output keeps the convolution's nested sums, in binary32: per input
    channel, an inner sum over a kernel row's columns, folded into a sum over
    the rows, folded into the output's running sum over channels, each sum
    seeded with positive zero and each product rounded before it is added; the
    bias is added last. A flat sum over the same taps rounds differently (the
    counterexample the window fixture pins), so this ordering is the kernel's
    meaning. There is no padding: a padded tap is skipped by the reference, not
    added as a zero, and this kernel has none to skip.

    The output channels are one register group (at most 16, an e32 m4 group on
    the 128-bit vector unit); the function uses no scalar register that the
    convention keeps, and leaves the vector configuration undefined. *)

(** The geometry of one window convolution. *)
module Window : sig
  type t = {
    channels : Core.Dim.extent Core.Dim.t;  (** input channels *)
    in_h : Core.Dim.extent Core.Dim.t;
    in_w : Core.Dim.extent Core.Dim.t;
    kernel_h : Core.Dim.extent Core.Dim.t;
    kernel_w : Core.Dim.extent Core.Dim.t;
    out_channels : Core.Dim.extent Core.Dim.t;
    stride_h : Core.Dim.extent Core.Dim.t;
    stride_w : Core.Dim.extent Core.Dim.t;
  }
end

val output_extents :
  Window.t ->
  ( Core.Dim.extent Core.Dim.t * Core.Dim.extent Core.Dim.t,
    Kernel_refusal.t )
  result
(** The output's height and width, or why the window is not one the kernel
    takes: a kernel past its stride (overlap) or past the input. *)

val symbol : Window.t -> string
(** The function's name, which spells the geometry. *)

val module_ :
  Window.t ->
  ( Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_,
    Kernel_refusal.t )
  result
