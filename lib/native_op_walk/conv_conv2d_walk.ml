(* The random-walk config spaces of the ops in [Conv_conv2d], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Conv2d = struct
  open Conv_conv2d.Conv2d

  (* This op's random-walk config space + the native-layout (NHWC) operand shape
     derivations (weight: N=out_channels, C=in_channels/groups, H/W=kernel; bias:
     C=out_channels — see [validate_channels]). [cascade] enforces the native
     conv's constraints (channels divisible by groups; input large enough for a
     >=1 output). Not shared with the ATen conv walk. *)
  module Walk (L : Walk_core.Limits.S) = struct
    (* Compound config: the input tensor is one [Shape.t] entry (its C axis is
       in_channels); kernel/stride/pad/dilation are H/W pairs; out_channels
       ("filters") and groups are scalars. A functor over the global Limits. *)
    type cfg = {
      shape : Walk_core.Shape.t;
      out_channels : int;
      kernel : Walk_core.Walk.hw;
      stride : Walk_core.Walk.hw;
      pad : Walk_core.Walk.hw;
      dilation : Walk_core.Walk.hw;
      groups : int;
    }

    let l = L.limits

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 8; w = 8; c = 4 };
        out_channels = 8;
        kernel = { Walk_core.Walk.h = 3; w = 3 };
        stride = { Walk_core.Walk.h = 1; w = 1 };
        pad = { Walk_core.Walk.h = 1; w = 1 };
        dilation = { Walk_core.Walk.h = 1; w = 1 };
        groups = 1;
      }

    let cascade c =
      let in_channels =
        Walk_core.Window_math.round_up_multiple ~n:c.shape.Walk_core.Shape.c
          ~m:c.groups
      in
      let out_channels =
        Walk_core.Window_math.round_up_multiple ~n:c.out_channels ~m:c.groups
      in
      let { Walk_core.Walk.h = kh; w = kw } = c.kernel in
      let { Walk_core.Walk.h = ph; w = pw } = c.pad in
      let { Walk_core.Walk.h = dh; w = dw } = c.dilation in
      let h =
        Walk_core.Window_math.grow_input ~in_size:c.shape.Walk_core.Shape.h
          ~pad:ph ~kernel:kh ~dilation:dh
      in
      let w =
        Walk_core.Window_math.grow_input ~in_size:c.shape.Walk_core.Shape.w
          ~pad:pw ~kernel:kw ~dilation:dw
      in
      {
        c with
        shape = { c.shape with Walk_core.Shape.c = in_channels; h; w };
        out_channels;
      }

    let mk_window ~kernel ~stride ~pad ~dilation : axis_window =
      {
        kernel = Dim.extent kernel;
        stride = Op_config.Pos.of_int stride;
        pad_before = Op_config.Nonneg.of_int pad;
        pad_after = Op_config.Nonneg.of_int pad;
        dilation = Op_config.Pos.of_int dilation;
      }

    let params (c : cfg) : params =
      let { Walk_core.Walk.h = kh; w = kw } = c.kernel in
      let { Walk_core.Walk.h = sh; w = sw } = c.stride in
      let { Walk_core.Walk.h = ph; w = pw } = c.pad in
      let { Walk_core.Walk.h = dh; w = dw } = c.dilation in
      {
        h = mk_window ~kernel:kh ~stride:sh ~pad:ph ~dilation:dh;
        w = mk_window ~kernel:kw ~stride:sw ~pad:pw ~dilation:dw;
        in_channels = Dim.extent c.shape.Walk_core.Shape.c;
        groups = Op_config.Pos.of_int c.groups;
      }

    let x_shape (c : cfg) = Walk_shape.vec6 c.shape

    let weight_shape (c : cfg) =
      let { Walk_core.Walk.h = kh; w = kw } = c.kernel in
      Vec6.shape ~n:c.out_channels ~t:1 ~d:1 ~h:kh ~w:kw
        ~c:(c.shape.Walk_core.Shape.c / c.groups)

    let bias_shape (c : cfg) =
      Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:c.out_channels

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" l
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          int_axis "out_channels" ~lo:1 ~hi:l.max_channels (fun c v ->
              { c with out_channels = v });
          hw_axis "kernel" ~lo:1 ~hi:l.max_kernel (fun c v ->
              { c with kernel = v });
          hw_axis "stride" ~lo:1 ~hi:l.max_stride (fun c v ->
              { c with stride = v });
          hw_axis "pad" ~lo:0 ~hi:l.max_pad (fun c v -> { c with pad = v });
          hw_axis "dilation" ~lo:1 ~hi:l.max_dilation (fun c v ->
              { c with dilation = v });
          int_axis "groups" ~lo:1 ~hi:4 (fun c v -> { c with groups = v });
        ]

    let pp fmt (c : cfg) =
      let { Walk_core.Walk.h = kh; w = kw } = c.kernel in
      let { Walk_core.Walk.h = sh; w = sw } = c.stride in
      let { Walk_core.Walk.h = ph; w = pw } = c.pad in
      let { Walk_core.Walk.h = dh; w = dw } = c.dilation in
      Format.fprintf fmt
        "{shape=%a kernel=%dx%d stride=%dx%d pad=%dx%d dilation=%dx%d \
         groups=%d out_c=%d}"
        Walk_core.Shape.pp c.shape kh kw sh sw ph pw dh dw c.groups
        c.out_channels
  end
end
