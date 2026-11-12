(* The random-walk config spaces of the ops in [Conv_padding], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Conv2d_padding = struct
  open Conv_padding.Conv2d_padding

  (* Its own config space: a string padding mode ("valid"/"same") instead of
     explicit pads, with the kernel carried by the weight shape. [cascade]
     enforces the native constraint that "same" requires unit stride (choosing
     "same" resets the strides to 1) plus channel divisibility and a big-enough
     input. Weight layout matches Conv2d (N=out_c, C=in_c/groups, H/W=kernel). *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = {
      shape : Walk_core.Shape.t;
      out_channels : int;
      kernel : Walk_core.Walk.hw;
      stride : Walk_core.Walk.hw;
      dilation : Walk_core.Walk.hw;
      groups : int;
      padding : padding;
    }

    let l = L.limits

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 8; w = 8; c = 4 };
        out_channels = 8;
        kernel = { Walk_core.Walk.h = 3; w = 3 };
        stride = { Walk_core.Walk.h = 1; w = 1 };
        dilation = { Walk_core.Walk.h = 1; w = 1 };
        groups = 1;
        padding = Same;
      }

    let cascade c =
      (* "same" requires unit stride (its native constraint). *)
      let stride =
        match c.padding with
        | Same -> { Walk_core.Walk.h = 1; w = 1 }
        | Valid -> c.stride
      in
      let in_channels =
        Walk_core.Window_math.round_up_multiple ~n:c.shape.Walk_core.Shape.c
          ~m:c.groups
      in
      let out_channels =
        Walk_core.Window_math.round_up_multiple ~n:c.out_channels ~m:c.groups
      in
      let { Walk_core.Walk.h = kh; w = kw } = c.kernel in
      let { Walk_core.Walk.h = dh; w = dw } = c.dilation in
      let h =
        Walk_core.Window_math.grow_input ~in_size:c.shape.Walk_core.Shape.h
          ~pad:0 ~kernel:kh ~dilation:dh
      in
      let w =
        Walk_core.Window_math.grow_input ~in_size:c.shape.Walk_core.Shape.w
          ~pad:0 ~kernel:kw ~dilation:dw
      in
      {
        c with
        stride;
        out_channels;
        shape = { c.shape with Walk_core.Shape.c = in_channels; h; w };
      }

    let params (c : cfg) : params =
      let { Walk_core.Walk.h = sh; w = sw } = c.stride in
      let { Walk_core.Walk.h = dh; w = dw } = c.dilation in
      {
        stride =
          {
            Op_config.Hw.h = Op_config.Pos.of_int sh;
            w = Op_config.Pos.of_int sw;
          };
        padding = c.padding;
        dilation =
          {
            Op_config.Hw.h = Op_config.Pos.of_int dh;
            w = Op_config.Pos.of_int dw;
          };
        groups = Op_config.Pos.of_int c.groups;
      }

    let x_shape (c : cfg) = Walk_shape.vec6 c.shape

    let weight_shape (c : cfg) =
      let { Walk_core.Walk.h = kh; w = kw } = c.kernel in
      Vec6.shape ~n:c.out_channels ~t:1 ~d:1 ~h:kh ~w:kw
        ~c:(c.shape.Walk_core.Shape.c / c.groups)

    let bias_shape (c : cfg) =
      Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:c.out_channels

    (* Odd kernels keep "same" well-behaved; the global Limits cap channels and
       extents (so a list is used only for the genuinely-discrete kernel + the
       padding mode). *)
    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" l
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          int_axis "out_channels" ~lo:1 ~hi:l.max_channels (fun c v ->
              { c with out_channels = v });
          field_axis "kernel"
            [ { h = 1; w = 1 }; { h = 3; w = 3 }; { h = 5; w = 5 } ]
            (fun c v -> { c with kernel = v });
          hw_axis "stride" ~lo:1 ~hi:l.max_stride (fun c v ->
              { c with stride = v });
          hw_axis "dilation" ~lo:1 ~hi:l.max_dilation (fun c v ->
              { c with dilation = v });
          int_axis "groups" ~lo:1 ~hi:4 (fun c v -> { c with groups = v });
          field_axis "padding" [ Valid; Same ] (fun c v ->
              { c with padding = v });
        ]

    let pp fmt (c : cfg) =
      let { Walk_core.Walk.h = kh; w = kw } = c.kernel in
      let { Walk_core.Walk.h = sh; w = sw } = c.stride in
      let { Walk_core.Walk.h = dh; w = dw } = c.dilation in
      Format.fprintf fmt
        "{shape=%a kernel=%dx%d stride=%dx%d dilation=%dx%d groups=%d out_c=%d \
         padding=%s}"
        Walk_core.Shape.pp c.shape kh kw sh sw dh dw c.groups c.out_channels
        (string_of_padding c.padding)
  end
end
