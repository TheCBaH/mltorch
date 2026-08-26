(* Shared windowed-axis arithmetic for conv, pool: 0 <= out*stride + k - pad < in_extent.
   [output_extent] asks "how many output positions"; [Compute.window] asks "for this
   output position, which k values and where do they read." Two closed forms over the
   same four numbers, co-located here instead of duplicated per op.
   See .ai/native_compute_design.md §2b. *)

(* Every factor is bounded against the per-axis ceiling BEFORE any arithmetic,
   and only then combined in [int64].

   Widening alone is not enough, which is the trap this went through once.
   [Op_config.Pos]/[Nonneg] and [Dim.extent] enforce SIGN, not magnitude, so a
   model-supplied dilation can be [max_int]; on a 63-bit-[int] build
   [Int64.mul] of that by a kernel of 5 wraps to a small negative number, and
   the bound placed after the multiplication then reads a value inside its
   range. With the factors bounded first, every product and sum below is at
   most 2^62 and cannot overflow [int64] at all.

   [dilation * (kernel - 1)] is the aggregate CLAUDE.md names: two individually
   in-range factors whose product need not be, on a backend where [int] is 32
   bits and the ceiling is 0x8000_0000. Reached by conv2d, conv2d.padding,
   convolution AND both pooling ops, so one bound here covers every windowed
   axis in the engine. See .ai/js_backends_design.md. *)
let limit = Kernel.Limits.Hard.extent

let factor ~(what : Shape_error.Window_over_limit.quantity) (n : int) :
    (int64, Shape_error.t) Err.t =
  let n = Int64.of_int n in
  if n >= limit then
    Err.fail
      (`Window_over_limit
         Shape_error.Window_over_limit.{ what; value = n; limit })
  else Err.return n

(* Floor division, which is what the window formula means. [Int64.div]
   truncates TOWARD ZERO, so a negative numerator -- a window wider than the
   padded input -- rounded the wrong way: in_extent 1 with kernel 2 and stride 2
   gives numerator -1, and (-1 / 2) + 1 = 1, one accepted output position where
   the floor is 0. ATen rejects that configuration ("Calculated output size:
   (1x0x0). Output size is too small"); native returned a value computed from a
   partial window. *)
let floor_div a b =
  if a >= 0L then Int64.div a b
  else Int64.neg (Int64.div (Int64.add (Int64.neg a) (Int64.sub b 1L)) b)

(* [ceil_mode] follows ATen's `pooling_output_shape_pad_lr`: round the
   quotient up instead of down (by biasing the numerator with [stride - 1]
   before the floor division this function already does), then pull the
   result back by one if that rounding would start the last window entirely
   past the real input plus its left/before padding -- "ensure the last
   pooling starts inside the image", in ATen's own words. Required (not
   optional-with-default), like every other window factor here: conv always
   passes [false], the prior floor-only behavior, explicitly. *)
let output_extent ~(ceil_mode : bool) ~(kernel : Dim.extent Dim.t)
    ~(stride : Op_config.Pos.t) ~(pad_before : Op_config.Nonneg.t)
    ~(pad_after : Op_config.Nonneg.t) ~(dilation : Op_config.Pos.t)
    ~(in_extent : Dim.extent Dim.t) : (Dim.extent Dim.t, Shape_error.t) Err.t =
  let open Err.Syntax in
  let* k = factor ~what:`Kernel (kernel :> int) in
  let* d = factor ~what:`Dilation (dilation :> int) in
  let* s = factor ~what:`Stride (stride :> int) in
  let* i = factor ~what:`Input_extent (in_extent :> int) in
  let* pb = factor ~what:`Padding (pad_before :> int) in
  let* pa = factor ~what:`Padding (pad_after :> int) in
  let effective_kernel = Int64.add (Int64.mul (Int64.sub k 1L) d) 1L in
  if effective_kernel >= limit then
    Err.fail
      (`Window_over_limit
         Shape_error.Window_over_limit.
           { what = `Effective_kernel; value = effective_kernel; limit })
  else
    let numerator =
      Int64.sub (Int64.add (Int64.add i pb) pa) effective_kernel
    in
    let numerator =
      if ceil_mode then Int64.add numerator (Int64.sub s 1L) else numerator
    in
    let out = Int64.add (floor_div numerator s) 1L in
    let out =
      if ceil_mode && Int64.mul (Int64.sub out 1L) s >= Int64.add i pb then
        Int64.sub out 1L
      else out
    in
    if out < 1L then
      Err.fail
        (`Window
           Shape_error.Window.
             { out; in_extent; kernel; stride; pad_before; pad_after; dilation })
    else if out >= limit then
      Err.fail
        (`Window_over_limit
           Shape_error.Window_over_limit.
             { what = `Output_extent; value = out; limit })
    else Err.return (Dim.extent (Int64.to_int out))

module Compute (S : sig
  type 'role index

  val index_extent : Dim.extent Dim.t -> Semantics.delta index
  val index_const : int -> Semantics.delta index
  val of_index : Semantics.position index -> Semantics.delta index

  val index_add :
    Semantics.delta index -> Semantics.delta index -> Semantics.delta index

  val index_scale : int -> Semantics.delta index -> Semantics.delta index

  val index_floor_div_pos :
    Semantics.delta index -> Op_config.Pos.t -> Semantics.delta index

  val index_ceil_div_pos :
    Semantics.delta index -> Op_config.Pos.t -> Semantics.delta index

  val index_min :
    Semantics.delta index -> Semantics.delta index -> Semantics.delta index

  val clamp_low : Semantics.delta index -> Semantics.position index
  val assume_index : Semantics.delta index -> Semantics.position index
end) =
struct
  (* Valid kernel-offset range [lo, hi) for one output position, plus [src k]:
     the source position for offset k. Callers iterate k in [lo,hi) and read
     at [src k] — always a valid index by construction (the clip above), so
     max-pooling's [max_reduce] never silently reads 0 from the pad region.
     [lo]/[src k] are [position]; [hi] is [delta] (an upper bound, never read). *)
  type window = {
    lo : Semantics.position S.index;
    hi : Semantics.delta S.index;
    src : Semantics.position S.index -> Semantics.position S.index;
  }

  let window ~(kernel : Dim.extent Dim.t) ~(stride : Op_config.Pos.t)
      ~(pad_before : Op_config.Nonneg.t) ~(dilation : Op_config.Pos.t)
      ~(in_extent : Dim.extent Dim.t) (out_axis : Semantics.position S.index) :
      window =
    let stride = (stride :> int) and pad_before = (pad_before :> int) in
    let out = S.of_index out_axis in
    let start =
      S.index_add (S.index_scale stride out) (S.index_const (-pad_before))
    in
    let neg_start = S.index_scale (-1) start in
    let lo = S.clamp_low (S.index_ceil_div_pos neg_start dilation) in
    let hi =
      S.index_min (S.index_extent kernel)
        (S.index_add
           (S.index_floor_div_pos
              (S.index_add
                 (S.index_add (S.index_extent in_extent) (S.index_const (-1)))
                 neg_start)
              dilation)
           (S.index_const 1))
    in
    (* [assume_index]: for k in [lo,hi),
       stride*out + k*dilation - pad_before is in [0,in_extent). *)
    let src k =
      S.assume_index
        (S.index_add start (S.index_scale (dilation :> int) (S.of_index k)))
    in
    { lo; hi; src }
end
