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
      out_channels : Dim.extent Dim.t;
      kernel : Dim.extent Dim.t Walk_core.Walk.hw;
      stride : Op_config.Pos.t Walk_core.Walk.hw;
      pad : Op_config.Nonneg.t Walk_core.Walk.hw;
      dilation : Op_config.Pos.t Walk_core.Walk.hw;
      groups : Dim.extent Dim.t;
    }

    let l = L.limits

    let initial =
      {
        shape =
          {
            Walk_core.Shape.n = Dim.one;
            t = Dim.one;
            d = Dim.one;
            h = Dim.extent 8;
            w = Dim.extent 8;
            c = Dim.extent 4;
          };
        out_channels = Dim.extent 8;
        kernel = { Walk_core.Walk.h = Dim.extent 3; w = Dim.extent 3 };
        stride =
          {
            Walk_core.Walk.h = Op_config.Pos.of_int 1;
            w = Op_config.Pos.of_int 1;
          };
        pad =
          {
            Walk_core.Walk.h = Op_config.Nonneg.of_int 1;
            w = Op_config.Nonneg.of_int 1;
          };
        dilation =
          {
            Walk_core.Walk.h = Op_config.Pos.of_int 1;
            w = Op_config.Pos.of_int 1;
          };
        groups = Dim.one;
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
      { kernel; stride; pad_before = pad; pad_after = pad; dilation }

    let params (c : cfg) : params =
      let { Walk_core.Walk.h = kh; w = kw } = c.kernel in
      let { Walk_core.Walk.h = sh; w = sw } = c.stride in
      let { Walk_core.Walk.h = ph; w = pw } = c.pad in
      let { Walk_core.Walk.h = dh; w = dw } = c.dilation in
      {
        h = mk_window ~kernel:kh ~stride:sh ~pad:ph ~dilation:dh;
        w = mk_window ~kernel:kw ~stride:sw ~pad:pw ~dilation:dw;
        in_channels = c.shape.Walk_core.Shape.c;
        groups = Op_config.Pos.of_int (c.groups :> int);
      }

    let x_shape (c : cfg) = Walk_shape.vec6 c.shape

    let weight_shape (c : cfg) =
      let { Walk_core.Walk.h = kh; w = kw } = c.kernel in
      Vec6.make ~n:c.out_channels ~t:Dim.one ~d:Dim.one ~h:kh ~w:kw
        ~c:(Option.get (Dim.div_exact c.shape.Walk_core.Shape.c ~by:c.groups))

    let bias_shape (c : cfg) =
      Vec6.make ~n:Dim.one ~t:Dim.one ~d:Dim.one ~h:Dim.one ~w:Dim.one
        ~c:c.out_channels

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" l
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          range_axis
            (module Extent)
            "out_channels" ~lo:Extent.one
            ~hi:(Extent.of_int l.max_channels)
            (fun c v -> { c with out_channels = v });
          hw_axis
            (module Extent)
            "kernel" ~lo:Extent.one
            ~hi:(Extent.of_int l.max_kernel)
            (fun c v -> { c with kernel = v });
          hw_axis
            (module Pos)
            "stride" ~lo:(Pos.of_int 1) ~hi:(Pos.of_int l.max_stride)
            (fun c v -> { c with stride = v });
          hw_axis
            (module Nonneg)
            "pad" ~lo:(Nonneg.of_int 0) ~hi:(Nonneg.of_int l.max_pad)
            (fun c v -> { c with pad = v });
          hw_axis
            (module Pos)
            "dilation" ~lo:(Pos.of_int 1)
            ~hi:(Pos.of_int l.max_dilation)
            (fun c v -> { c with dilation = v });
          range_axis
            (module Extent)
            "groups" ~lo:Extent.one ~hi:(Extent.of_int 4)
            (fun c v -> { c with groups = v });
        ]

    let pp fmt (c : cfg) =
      let { Walk_core.Walk.h = kh; w = kw } = c.kernel in
      let { Walk_core.Walk.h = sh; w = sw } = c.stride in
      let { Walk_core.Walk.h = ph; w = pw } = c.pad in
      let { Walk_core.Walk.h = dh; w = dw } = c.dilation in
      Format.fprintf fmt
        "{shape=%a kernel=%ax%a stride=%ax%a pad=%ax%a dilation=%ax%a \
         groups=%a out_c=%a}"
        Walk_core.Shape.pp c.shape Dim.pp kh Dim.pp kw Op_config.Pos.pp sh
        Op_config.Pos.pp sw Op_config.Nonneg.pp ph Op_config.Nonneg.pp pw
        Op_config.Pos.pp dh Op_config.Pos.pp dw Dim.pp c.groups Dim.pp
        c.out_channels
  end
end
