(* 2D convolution (NHWC). Output[n,t,d,h,w,oc] reduces over input channels and the
   kernel window; the input position is the affine index [out*stride + k - pad],
   built through the SEMANTICS index ops so it specialises to either concrete ints
   (Direct) or index expressions (Symbolic). The kh/kw reduction bounds are
   clipped (via [Index_max]/[Index_min]) to exactly the kernel offsets whose source
   position lands inside the input — so that position is always a valid index,
   never a generate-then-guard read of the padding region (see
   .ai/native_compute_design.md §4 "Open issue"). Weight is laid out
   [Cout,1,1,Kh,Kw,Cin]; bias is [1,1,1,1,1,Cout] (broadcast — extent-1 axes
   still go through [load]'s ordinary broadcast). See
   .ai/native_compute_design.md §2. *)

module Conv2d = struct
  (* [params]/[output_shape] are outside [Compute] so Direct/Symbolic share one
     [params] type. Field types per .ai/native_op_config.md. *)
  type params = {
    kernel : Dim.extent Dim.t Op_config.Hw.t;
    in_channels : Dim.extent Dim.t;
    stride : Op_config.Pos.t Op_config.Hw.t;
    pad : Op_config.Nonneg.t Op_config.Hw.t;
  }

  (* N/T/D pass through; H/W shrink via [Window_axis.output_extent]; C = Cout from
     weight_shape (weight is [Cout,1,1,Kh,Kw,Cin] — Cout isn't in params). *)
  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) : Vec6.shape =
    let out_extent axis ~kernel ~stride ~pad =
      Window_axis.output_extent ~in_extent:(Vec6.get x_shape axis) ~kernel
        ~stride ~pad
    in
    (* Start from the input shape (N/T/D pass through unchanged) and replace only
       the axes the op resizes — all in extent-space, no [:> int] round-trips. *)
    Vec6.set
      (Vec6.set
         (Vec6.set x_shape Axis.H
            (out_extent Axis.H ~kernel:p.kernel.h ~stride:p.stride.h
               ~pad:p.pad.h))
         Axis.W
         (out_extent Axis.W ~kernel:p.kernel.w ~stride:p.stride.w ~pad:p.pad.w))
      Axis.C
      (Vec6.get weight_shape Axis.N)

  module Compute (S : Semantics.SEMANTICS) = struct
    module Wa = Window_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x ~weight ~bias
        (out : Axis.t -> Semantics.position S.index) : S.t =
      let oc = out Axis.C in
      let wh =
        Wa.window ~kernel:p.kernel.h ~stride:p.stride.h ~pad:p.pad.h
          ~in_extent:(Vec6.get x_shape Axis.H) (out Axis.H)
      in
      let ww =
        Wa.window ~kernel:p.kernel.w ~stride:p.stride.w ~pad:p.pad.w
          ~in_extent:(Vec6.get x_shape Axis.W) (out Axis.W)
      in
      let acc =
        S.sum ~lo:S.index_zero ~hi:(S.index_extent p.in_channels) (fun ic ->
            S.sum ~lo:wh.lo ~hi:wh.hi (fun kh ->
                S.sum ~lo:ww.lo ~hi:ww.hi (fun kw ->
                    let x_idx a =
                      match a with
                      | Axis.H -> wh.src kh
                      | Axis.W -> ww.src kw
                      | Axis.C -> ic
                      | _ -> out a
                    in
                    let w_idx a =
                      match a with
                      | Axis.N -> oc
                      | Axis.H -> kh
                      | Axis.W -> kw
                      | Axis.C -> ic
                      | _ -> S.index_zero
                    in
                    S.mul (S.load x x_idx) (S.load weight w_idx))))
      in
      S.add acc
        (S.load bias (fun a -> match a with Axis.C -> oc | _ -> S.index_zero))
  end
end
