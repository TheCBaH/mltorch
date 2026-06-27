(* 2D pooling (NHWC): [MaxPool2d] and [AvgPool2d]. Output[n,t,d,h,w,c] reduces
   over the kernel window at the SAME channel c — no channel mixing, unlike
   conv (no [in_channels], no weight). Dilation and ceil_mode aren't modeled
   (dilation=1, ceil_mode=false implicitly) — resnet18's pool is kernel=3,
   stride=2, pad=1, dilation=1, ceil_mode=false, and nothing else needs more
   yet.

   Both share [Window_axis] with conv: the kh/kw reduction bounds are clipped
   to exactly the kernel offsets whose source position lands inside the
   input, so the position [load] reads is always valid by construction. For
   [MaxPool2d] this isn't just convenient — it's required for correctness: a
   guarded-read-returns-0 fallback for the padding region (as a naive [load]
   would do) would be wrong for [max_reduce] over real, possibly-negative data,
   where 0 is not a safe stand-in for "no value here." See
   .ai/native_compute_design.md §4. *)

(* Pooling output shape, shared by both pools (their [params] are field-
   compatible): N/T/D/C pass through from [x_shape] unchanged (pooling never
   touches channels); only H/W shrink by [Window_axis.output_extent]. All
   extent-space, no [:> int] round-trips. See .ai/native_compute_design.md §2b. *)
let window_output_shape ~(x_shape : Vec6.shape)
    ~(kernel : Dim.extent Dim.t Op_config.Hw.t)
    ~(stride : Op_config.Pos.t Op_config.Hw.t)
    ~(pad : Op_config.Nonneg.t Op_config.Hw.t) =
  let out_extent axis ~kernel ~stride ~pad =
    Window_axis.output_extent ~in_extent:(Vec6.get x_shape axis) ~kernel ~stride
      ~pad
  in
  Vec6.set
    (Vec6.set x_shape Axis.H
       (out_extent Axis.H ~kernel:kernel.h ~stride:stride.h ~pad:pad.h))
    Axis.W
    (out_extent Axis.W ~kernel:kernel.w ~stride:stride.w ~pad:pad.w)

module MaxPool2d = struct
  (* Matches ATen's `max_pool2d` (the value output only; `max_pool2d_with_indices`
     — what resnet18's graph actually calls — also returns the argmax indices
     for backprop, not modeled here: this engine is inference-only, the
     indices output is unused in eval mode). *)

  (* Same field shape as [Conv2d.params] minus [in_channels] (pooling never
     reduces over channels). See .ai/native_op_config.md, which already
     names this as the expected carry-over. *)
  type params = {
    kernel : Dim.extent Dim.t Op_config.Hw.t;
    stride : Op_config.Pos.t Op_config.Hw.t;
    pad : Op_config.Nonneg.t Op_config.Hw.t;
  }

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    window_output_shape ~x_shape ~kernel:p.kernel ~stride:p.stride ~pad:p.pad

  module Compute (S : Semantics.SEMANTICS) = struct
    module Wa = Window_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Axis.t -> Semantics.position S.index) =
      let wh =
        Wa.window ~kernel:p.kernel.h ~stride:p.stride.h ~pad:p.pad.h
          ~in_extent:(Vec6.get x_shape Axis.H) (out Axis.H)
      in
      let ww =
        Wa.window ~kernel:p.kernel.w ~stride:p.stride.w ~pad:p.pad.w
          ~in_extent:(Vec6.get x_shape Axis.W) (out Axis.W)
      in
      S.max_reduce ~lo:wh.lo ~hi:wh.hi (fun kh ->
          S.max_reduce ~lo:ww.lo ~hi:ww.hi (fun kw ->
              S.load x (fun a ->
                  match a with
                  | Axis.H -> wh.src kh
                  | Axis.W -> ww.src kw
                  | _ -> out a)))
  end
end

module AvgPool2d = struct
  (* Same [params] shape as [MaxPool2d] (kernel/stride/pad, no [in_channels]).
     Matches ATen's `avg_pool2d` defaults: `count_include_pad=true`,
     `divisor_override=None`, `ceil_mode=false` — i.e. every window divides by
     the FULL kernel area, even where part of it falls in the padding region.
     That divisor is therefore a position-independent constant, which is why
     this doesn't need the [Window_axis] window for the divisor — only for the
     sum, exactly as in [MaxPool2d]: clipping the reduction range means the
     padding region simply isn't summed (contributing the same 0 it would
     under a guarded read, but never via an out-of-bounds index), while the
     divisor still counts it. *)
  type params = {
    kernel : Dim.extent Dim.t Op_config.Hw.t;
    stride : Op_config.Pos.t Op_config.Hw.t;
    pad : Op_config.Nonneg.t Op_config.Hw.t;
  }

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    window_output_shape ~x_shape ~kernel:p.kernel ~stride:p.stride ~pad:p.pad

  module Compute (S : Semantics.SEMANTICS) = struct
    module Wa = Window_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Axis.t -> Semantics.position S.index) =
      let wh =
        Wa.window ~kernel:p.kernel.h ~stride:p.stride.h ~pad:p.pad.h
          ~in_extent:(Vec6.get x_shape Axis.H) (out Axis.H)
      in
      let ww =
        Wa.window ~kernel:p.kernel.w ~stride:p.stride.w ~pad:p.pad.w
          ~in_extent:(Vec6.get x_shape Axis.W) (out Axis.W)
      in
      let total =
        S.sum ~lo:wh.lo ~hi:wh.hi (fun kh ->
            S.sum ~lo:ww.lo ~hi:ww.hi (fun kw ->
                S.load x (fun a ->
                    match a with
                    | Axis.H -> wh.src kh
                    | Axis.W -> ww.src kw
                    | _ -> out a)))
      in
      let area = (p.kernel.h :> int) * (p.kernel.w :> int) in
      S.div total (S.const (float_of_int area))
  end
end
