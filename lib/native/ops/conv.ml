(* 2D convolution (NHWC). Output[n,t,d,h,w,oc] reduces over input channels and the
   kernel window; the input position is the affine index [out*stride + k - pad],
   built through the SEMANTICS index ops so it specialises to either concrete ints
   (Direct) or index expressions (Symbolic). The kh/kw reduction bounds are
   clipped (via [Index_max]/[Index_min]) to exactly the kernel offsets whose source
   position lands inside the input — so that position is always a valid index,
   never a generate-then-guard read of the padding region (see
   .ai/native_compute_design.md §4 "Open issue"). Weight is laid out
   [Cout,1,1,Kh,Kw,Cin_per_group]; bias is [1,1,1,1,1,Cout] (broadcast —
   extent-1 axes still go through [load]'s ordinary broadcast). See
   .ai/native_compute_design.md §2. *)

let or_invalid_arg = function
  | Ok x -> x
  | Error e ->
      invalid_arg (Format.asprintf "%a" Shape_error.pp e.Core.Error.kind)

module Conv2d = struct
  (* [params]/[output_shape] are outside [Compute] so Direct/Symbolic share one
     [params] type. Field types per .ai/native_op_config.md. *)
  type axis_window = {
    kernel : Dim.extent Dim.t;
    stride : Op_config.Pos.t;
    pad_before : Op_config.Nonneg.t;
    pad_after : Op_config.Nonneg.t;
    dilation : Op_config.Pos.t;
  }

  type params = {
    h : axis_window;
    w : axis_window;
    in_channels : Dim.extent Dim.t;
    groups : Op_config.Pos.t;
  }

  let axis_window_jsont : axis_window Jsont.t =
    Jsont.Object.map ~kind:"conv2d_axis_window"
      (fun kernel stride pad_before pad_after dilation ->
        { kernel; stride; pad_before; pad_after; dilation })
    |> Jsont.Object.mem "kernel" Dim.extent_jsont ~enc:(fun w -> w.kernel)
    |> Jsont.Object.mem "stride" Op_config.Pos.jsont ~enc:(fun w -> w.stride)
    |> Jsont.Object.mem "pad_before" Op_config.Nonneg.jsont ~enc:(fun w ->
        w.pad_before)
    |> Jsont.Object.mem "pad_after" Op_config.Nonneg.jsont ~enc:(fun w ->
        w.pad_after)
    |> Jsont.Object.mem "dilation" Op_config.Pos.jsont ~enc:(fun w ->
        w.dilation)
    |> Jsont.Object.finish

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"conv2d_params" (fun h w in_channels groups ->
        { h; w; in_channels; groups })
    |> Jsont.Object.mem "h" axis_window_jsont ~enc:(fun p -> p.h)
    |> Jsont.Object.mem "w" axis_window_jsont ~enc:(fun p -> p.w)
    |> Jsont.Object.mem "in_channels" Dim.extent_jsont ~enc:(fun p ->
        p.in_channels)
    |> Jsont.Object.mem "groups" Op_config.Pos.jsont ~enc:(fun p -> p.groups)
    |> Jsont.Object.finish

  let pp_axis_window fmt (w : axis_window) =
    Fmt.pf fmt
      "@[<hv>{kernel=%a;@ stride=%a;@ pad_before=%a;@ pad_after=%a;@ \
       dilation=%a}@]"
      Dim.pp w.kernel Op_config.Pos.pp w.stride Op_config.Nonneg.pp w.pad_before
      Op_config.Nonneg.pp w.pad_after Op_config.Pos.pp w.dilation

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{h=%a;@ w=%a;@ in_channels=%a;@ groups=%a}@]"
      pp_axis_window p.h pp_axis_window p.w Dim.pp p.in_channels
      Op_config.Pos.pp p.groups

  (* This op's random-walk config space + the native-layout (NHWC) operand shape
     derivations (weight: N=out_channels, C=in_channels/groups, H/W=kernel; bias:
     C=out_channels — see [validate_channels]). [cascade] enforces the native
     conv's constraints (channels divisible by groups; input large enough for a
     >=1 output). Lives with the op; not shared with the ATen conv walk. *)
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

    let x_shape (c : cfg) = Walk_bridge.vec6 c.shape

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

  (* The op payload: its params plus its operand edges. Carrying the operands here
     (rather than in [Graph_ir]'s variant) lets the op own its own serialisation,
     dataflow accessors and pretty-printer. *)
  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t;
    bias : Tensor_ref.t option;
  }

  let name = "Conv2d"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          x = get "x" Tensor_ref.jsont;
          weight = get "weight" Tensor_ref.jsont;
          bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt_bias =
          match t.bias with None -> [] | Some r -> [ ("bias", ref_ r) ]
        in
        Json_util.jobj
          (opt_bias
          @ [
              ("params", Json_util.enc params_jsont t.params);
              ("weight", ref_ t.weight);
              ("x", ref_ t.x);
            ]))
      Jsont.json

  let operands (t : t) = [ t.x; t.weight ] @ Option.to_list t.bias

  let map_operands f (t : t) =
    { t with x = f t.x; weight = f t.weight; bias = Option.map f t.bias }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>conv2d@ x=%a@ weight=%a@ bias=%a@ params=%a@]" pp_ref
      t.x pp_ref t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  let validate_channels ~(weight_shape : Vec6.shape) (p : params) =
    let open Core.Syntax in
    let in_channels = (p.in_channels :> int) in
    let groups = (p.groups :> int) in
    let out_channels = (Vec6.get weight_shape Axis.N :> int) in
    let weight_in_per_group = (Vec6.get weight_shape Axis.C :> int) in
    let* () =
      if in_channels mod groups <> 0 then
        Core.fail
          (`Convolution
             (Shape_error.Convolution.In_channels_not_divisible_by_groups
                Shape_error.Convolution.{ channels = in_channels; groups }))
      else Core.return ()
    in
    let* () =
      if out_channels mod groups <> 0 then
        Core.fail
          (`Convolution
             (Shape_error.Convolution.Out_channels_not_divisible_by_groups
                Shape_error.Convolution.{ channels = out_channels; groups }))
      else Core.return ()
    in
    let in_per_group = in_channels / groups in
    let* () =
      if weight_in_per_group <> in_per_group then
        Core.fail
          (`Convolution
             (Shape_error.Convolution.Weight_channels_mismatch
                Shape_error.Convolution.
                  { weight_in_per_group; expected_in_per_group = in_per_group }))
      else Core.return ()
    in
    let* () =
      if not (Dim.equal (Vec6.get weight_shape Axis.H) p.h.kernel) then
        Core.fail
          (`Convolution
             (Shape_error.Convolution.Weight_kernel_mismatch
                Shape_error.Convolution.
                  {
                    axis = Axis.H;
                    weight_extent = Vec6.get weight_shape Axis.H;
                    kernel_extent = p.h.kernel;
                  }))
      else Core.return ()
    in
    let* () =
      if not (Dim.equal (Vec6.get weight_shape Axis.W) p.w.kernel) then
        Core.fail
          (`Convolution
             (Shape_error.Convolution.Weight_kernel_mismatch
                Shape_error.Convolution.
                  {
                    axis = Axis.W;
                    weight_extent = Vec6.get weight_shape Axis.W;
                    kernel_extent = p.w.kernel;
                  }))
      else Core.return ()
    in
    Core.return (in_per_group, out_channels / groups)

  (* N/T/D pass through; H/W shrink via [Window_axis.output_extent]; C = Cout from
     weight_shape. *)
  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    let open Core.Syntax in
    let* _in_per_group, _out_per_group = validate_channels ~weight_shape p in
    let* () =
      if not (Dim.equal (Vec6.get x_shape Axis.C) p.in_channels) then
        Core.fail
          (`Convolution
             (Shape_error.Convolution.Input_channels_mismatch
                Shape_error.Convolution.
                  {
                    input_channels = Vec6.get x_shape Axis.C;
                    expected_in_channels = p.in_channels;
                  }))
      else Core.return ()
    in
    let out_extent axis (w : axis_window) =
      Window_axis.output_extent ~in_extent:(Vec6.get x_shape axis)
        ~kernel:w.kernel ~stride:w.stride ~pad_before:w.pad_before
        ~pad_after:w.pad_after ~dilation:w.dilation
    in
    (* Start from the input shape (N/T/D pass through unchanged) and replace only
       the axes the op resizes — all in extent-space, no [:> int] round-trips. *)
    let* h = out_extent Axis.H p.h in
    let+ w = out_extent Axis.W p.w in
    Vec6.set
      (Vec6.set (Vec6.set x_shape Axis.H h) Axis.W w)
      Axis.C
      (Vec6.get weight_shape Axis.N)

  module Compute (S : Semantics.SEMANTICS) = struct
    module Wa = Window_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
        ~x ~weight ~bias (out : Semantics.position S.index Vec6.t) =
      let in_per_group, out_per_group =
        or_invalid_arg (validate_channels ~weight_shape p)
      in
      let oc = Vec6.get out Axis.C in
      let group =
        if (p.groups :> int) = 1 then S.index_const 0
        else
          S.index_floor_div_pos (S.of_index oc)
            (Op_config.Pos.of_int out_per_group)
      in
      let wh =
        Wa.window ~kernel:p.h.kernel ~stride:p.h.stride
          ~pad_before:p.h.pad_before ~dilation:p.h.dilation
          ~in_extent:(Vec6.get x_shape Axis.H) (Vec6.get out Axis.H)
      in
      let ww =
        Wa.window ~kernel:p.w.kernel ~stride:p.w.stride
          ~pad_before:p.w.pad_before ~dilation:p.w.dilation
          ~in_extent:(Vec6.get x_shape Axis.W) (Vec6.get out Axis.W)
      in
      (* [load6] (6 explicit indices), not [load]+a closure: this is the
         highest-call-volume site in the engine (one call per input-channel x
         kernel-h x kernel-w x output-pixel), and a closure built fresh per
         call — as [x_idx]/[w_idx] used to be here — allocates every time.
         See .ai/pt2_inference_perf.md. *)
      let on = Vec6.get out Axis.N
      and ot = Vec6.get out Axis.T
      and od = Vec6.get out Axis.D in
      let acc =
        S.sum ~lo:S.index_zero
          ~hi:(S.index_extent (Dim.extent in_per_group))
          (fun local_ic ->
            S.sum ~lo:wh.lo ~hi:wh.hi (fun kh ->
                S.sum ~lo:ww.lo ~hi:ww.hi (fun kw ->
                    let ic =
                      S.assume_index
                        (S.index_add
                           (S.index_scale in_per_group group)
                           (S.of_index local_ic))
                    in
                    S.mul
                      (S.load6 x ~n:on ~t:ot ~d:od ~h:(wh.src kh) ~w:(ww.src kw)
                         ~c:ic)
                      (S.load6 weight ~n:oc ~t:S.index_zero ~d:S.index_zero
                         ~h:kh ~w:kw ~c:local_ic))))
      in
      S.add acc
        (S.load bias
           (Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
              ~h:S.index_zero ~w:S.index_zero ~c:oc))
  end
end

module Conv2d_padding = struct
  type padding = Valid | Same

  let padding_of_string = function
    | "valid" -> Valid
    | "same" -> Same
    | s -> invalid_arg ("Conv2d_padding: invalid padding " ^ s)

  let string_of_padding = function Valid -> "valid" | Same -> "same"

  let padding_jsont : padding Jsont.t =
    Jsont.map ~kind:"conv2d_padding_mode"
      ~dec:(fun s ->
        try padding_of_string s
        with Invalid_argument msg -> Jsont.Error.msgf Jsont.Meta.none "%s" msg)
      ~enc:string_of_padding Jsont.string

  let pp_padding fmt p = Fmt.string fmt (string_of_padding p)

  type params = {
    stride : Op_config.Pos.t Op_config.Hw.t;
    padding : padding;
    dilation : Op_config.Pos.t Op_config.Hw.t;
    groups : Op_config.Pos.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"conv2d_padding_params"
      (fun stride padding dilation groups ->
        { stride; padding; dilation; groups })
    |> Jsont.Object.mem "stride" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.stride)
    |> Jsont.Object.mem "padding" padding_jsont ~enc:(fun p -> p.padding)
    |> Jsont.Object.mem "dilation" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.dilation)
    |> Jsont.Object.mem "groups" Op_config.Pos.jsont ~enc:(fun p -> p.groups)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{stride=%a;@ padding=%a;@ dilation=%a;@ groups=%a}@]"
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.stride pp_padding p.padding
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.dilation Op_config.Pos.pp p.groups

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

    let x_shape (c : cfg) = Walk_bridge.vec6 c.shape

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

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t;
    bias : Tensor_ref.t option;
  }

  let name = "Conv2d_padding"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          x = get "x" Tensor_ref.jsont;
          weight = get "weight" Tensor_ref.jsont;
          bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt_bias =
          match t.bias with None -> [] | Some r -> [ ("bias", ref_ r) ]
        in
        Json_util.jobj
          (opt_bias
          @ [
              ("params", Json_util.enc params_jsont t.params);
              ("weight", ref_ t.weight);
              ("x", ref_ t.x);
            ]))
      Jsont.json

  let operands (t : t) = [ t.x; t.weight ] @ Option.to_list t.bias

  let map_operands f (t : t) =
    { t with x = f t.x; weight = f t.weight; bias = Option.map f t.bias }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>conv2d_padding@ x=%a@ weight=%a@ bias=%a@ params=%a@]"
      pp_ref t.x pp_ref t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  let same_padding ~(kernel : Dim.extent Dim.t) ~(stride : Op_config.Pos.t)
      ~(dilation : Op_config.Pos.t) =
    if (stride :> int) <> 1 then
      Core.fail
        (`Convolution
           (Shape_error.Convolution.Same_padding_requires_stride_one { stride }))
    else
      let total = (dilation :> int) * ((kernel :> int) - 1) in
      Core.return
        ( Op_config.Nonneg.of_int (total / 2),
          Op_config.Nonneg.of_int (total - (total / 2)) )

  let axis_window ~(padding : padding) ~(kernel : Dim.extent Dim.t)
      ~(stride : Op_config.Pos.t) ~(dilation : Op_config.Pos.t) :
      (Conv2d.axis_window, Shape_error.t) Core.result =
    let open Core.Syntax in
    let* pad_before, pad_after =
      match padding with
      | Valid ->
          Core.return (Op_config.Nonneg.of_int 0, Op_config.Nonneg.of_int 0)
      | Same -> same_padding ~kernel ~stride ~dilation
    in
    Core.return { Conv2d.kernel; stride; pad_before; pad_after; dilation }

  let to_conv2d_params ~(weight_shape : Vec6.shape) (p : params) :
      (Conv2d.params, Shape_error.t) Core.result =
    let open Core.Syntax in
    let groups = (p.groups :> int) in
    let in_channels = (Vec6.get weight_shape Axis.C :> int) * groups in
    let* h =
      axis_window ~padding:p.padding
        ~kernel:(Vec6.get weight_shape Axis.H)
        ~stride:p.stride.h ~dilation:p.dilation.h
    in
    let+ w =
      axis_window ~padding:p.padding
        ~kernel:(Vec6.get weight_shape Axis.W)
        ~stride:p.stride.w ~dilation:p.dilation.w
    in
    { Conv2d.h; w; in_channels = Dim.extent in_channels; groups = p.groups }

  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    let open Core.Syntax in
    let* p = to_conv2d_params ~weight_shape p in
    Conv2d.output_shape ~x_shape ~weight_shape p

  module Compute (S : Semantics.SEMANTICS) = struct
    module C = Conv2d.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
        ~x ~weight ~bias out =
      C.pixel
        (or_invalid_arg (to_conv2d_params ~weight_shape p))
        ~x_shape ~weight_shape ~x ~weight ~bias out
  end
end

module Convolution = struct
  type params = {
    stride : Op_config.Pos.t Op_config.Hw.t;
    padding : Op_config.Nonneg.t Op_config.Hw.t;
    dilation : Op_config.Pos.t Op_config.Hw.t;
    transposed : bool;
    output_padding : Op_config.Nonneg.t Op_config.Hw.t;
    groups : Op_config.Pos.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"convolution_params"
      (fun stride padding dilation transposed output_padding groups ->
        { stride; padding; dilation; transposed; output_padding; groups })
    |> Jsont.Object.mem "stride" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.stride)
    |> Jsont.Object.mem "padding" (Op_config.Hw.jsont Op_config.Nonneg.jsont)
         ~enc:(fun p -> p.padding)
    |> Jsont.Object.mem "dilation" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.dilation)
    |> Jsont.Object.mem "transposed" Jsont.bool ~enc:(fun p -> p.transposed)
    |> Jsont.Object.mem "output_padding"
         (Op_config.Hw.jsont Op_config.Nonneg.jsont) ~enc:(fun p ->
           p.output_padding)
    |> Jsont.Object.mem "groups" Op_config.Pos.jsont ~enc:(fun p -> p.groups)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt
      "@[<hv>{stride=%a;@ padding=%a;@ dilation=%a;@ transposed=%b;@ \
       output_padding=%a;@ groups=%a}@]"
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.stride
      (Op_config.Hw.pp Op_config.Nonneg.pp)
      p.padding
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.dilation p.transposed
      (Op_config.Hw.pp Op_config.Nonneg.pp)
      p.output_padding Op_config.Pos.pp p.groups

  (* Reuses Conv2d's config space (a non-transposed convolution is shape-
     identical); only the [params] form differs (Hw fields, transposed=false,
     output_padding=0). *)
  module Walk (L : Walk_core.Limits.S) = struct
    module W = Conv2d.Walk (L)
    include W

    let params (c : W.cfg) : params =
      let { Walk_core.Walk.h = sh; w = sw } = c.stride in
      let { Walk_core.Walk.h = ph; w = pw } = c.pad in
      let { Walk_core.Walk.h = dh; w = dw } = c.dilation in
      {
        stride =
          {
            Op_config.Hw.h = Op_config.Pos.of_int sh;
            w = Op_config.Pos.of_int sw;
          };
        padding =
          {
            Op_config.Hw.h = Op_config.Nonneg.of_int ph;
            w = Op_config.Nonneg.of_int pw;
          };
        dilation =
          {
            Op_config.Hw.h = Op_config.Pos.of_int dh;
            w = Op_config.Pos.of_int dw;
          };
        transposed = false;
        output_padding =
          {
            Op_config.Hw.h = Op_config.Nonneg.of_int 0;
            w = Op_config.Nonneg.of_int 0;
          };
        groups = Op_config.Pos.of_int c.groups;
      }
  end

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t;
    bias : Tensor_ref.t option;
  }

  let name = "Convolution"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          x = get "x" Tensor_ref.jsont;
          weight = get "weight" Tensor_ref.jsont;
          bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt_bias =
          match t.bias with None -> [] | Some r -> [ ("bias", ref_ r) ]
        in
        Json_util.jobj
          (opt_bias
          @ [
              ("params", Json_util.enc params_jsont t.params);
              ("weight", ref_ t.weight);
              ("x", ref_ t.x);
            ]))
      Jsont.json

  let operands (t : t) = [ t.x; t.weight ] @ Option.to_list t.bias

  let map_operands f (t : t) =
    { t with x = f t.x; weight = f t.weight; bias = Option.map f t.bias }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>convolution@ x=%a@ weight=%a@ bias=%a@ params=%a@]"
      pp_ref t.x pp_ref t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  let require_forward_2d (p : params) =
    let open Core.Syntax in
    let* () =
      if p.transposed then
        Core.fail
          (`Convolution Shape_error.Convolution.Transposed_not_supported)
      else Core.return ()
    in
    if (p.output_padding.h :> int) <> 0 || (p.output_padding.w :> int) <> 0 then
      Core.fail
        (`Convolution
           (Shape_error.Convolution.Output_padding_nonzero
              Shape_error.Convolution.
                { h = p.output_padding.h; w = p.output_padding.w }))
    else Core.return ()

  let axis_window ~kernel ~stride ~pad ~dilation : Conv2d.axis_window =
    { Conv2d.kernel; stride; pad_before = pad; pad_after = pad; dilation }

  let to_conv2d_params ~(weight_shape : Vec6.shape) (p : params) :
      (Conv2d.params, Shape_error.t) Core.result =
    let open Core.Syntax in
    let* () = require_forward_2d p in
    let groups = (p.groups :> int) in
    let in_channels = (Vec6.get weight_shape Axis.C :> int) * groups in
    Core.return
      {
        Conv2d.h =
          axis_window
            ~kernel:(Vec6.get weight_shape Axis.H)
            ~stride:p.stride.h ~pad:p.padding.h ~dilation:p.dilation.h;
        w =
          axis_window
            ~kernel:(Vec6.get weight_shape Axis.W)
            ~stride:p.stride.w ~pad:p.padding.w ~dilation:p.dilation.w;
        in_channels = Dim.extent in_channels;
        groups = p.groups;
      }

  let transposed_output_axis ~(in_extent : Dim.extent Dim.t)
      ~(kernel : Dim.extent Dim.t) ~(stride : Op_config.Pos.t)
      ~(pad : Op_config.Nonneg.t) ~(dilation : Op_config.Pos.t)
      ~(output_padding : Op_config.Nonneg.t) =
    let out =
      (((in_extent :> int) - 1) * (stride :> int))
      - (2 * (pad :> int))
      + ((dilation :> int) * ((kernel :> int) - 1))
      + (output_padding :> int)
      + 1
    in
    if out < 1 then
      Core.fail
        (`Convolution
           (Shape_error.Convolution.Transposed_output_non_positive
              Shape_error.Convolution.
                {
                  out;
                  in_extent;
                  kernel;
                  stride;
                  pad;
                  dilation;
                  output_padding;
                }))
    else Core.return (Dim.extent out)

  let validate_transposed_channels ~(x_shape : Vec6.shape)
      ~(weight_shape : Vec6.shape) (p : params) =
    let open Core.Syntax in
    let groups = (p.groups :> int) in
    let in_channels = (Vec6.get x_shape Axis.C :> int) in
    let weight_in_channels = (Vec6.get weight_shape Axis.N :> int) in
    let out_per_group = (Vec6.get weight_shape Axis.C :> int) in
    let* () =
      if weight_in_channels <> in_channels then
        Core.fail
          (`Convolution
             (Shape_error.Convolution.Transposed_weight_input_mismatch
                Shape_error.Convolution.
                  {
                    weight_input_channels = Vec6.get weight_shape Axis.N;
                    input_channels = Vec6.get x_shape Axis.C;
                  }))
      else Core.return ()
    in
    let* () =
      if in_channels mod groups <> 0 then
        Core.fail
          (`Convolution
             (Shape_error.Convolution
              .Transposed_input_channels_not_divisible_by_groups
                Shape_error.Convolution.{ channels = in_channels; groups }))
      else Core.return ()
    in
    Core.return (in_channels / groups, out_per_group)

  let transposed_output_shape ~(x_shape : Vec6.shape)
      ~(weight_shape : Vec6.shape) (p : params) =
    let open Core.Syntax in
    let* _in_per_group, out_per_group =
      validate_transposed_channels ~x_shape ~weight_shape p
    in
    let* h =
      transposed_output_axis ~in_extent:(Vec6.get x_shape Axis.H)
        ~kernel:(Vec6.get weight_shape Axis.H)
        ~stride:p.stride.h ~pad:p.padding.h ~dilation:p.dilation.h
        ~output_padding:p.output_padding.h
    in
    let+ w =
      transposed_output_axis ~in_extent:(Vec6.get x_shape Axis.W)
        ~kernel:(Vec6.get weight_shape Axis.W)
        ~stride:p.stride.w ~pad:p.padding.w ~dilation:p.dilation.w
        ~output_padding:p.output_padding.w
    in
    Vec6.set
      (Vec6.set (Vec6.set x_shape Axis.H h) Axis.W w)
      Axis.C
      (Dim.extent (out_per_group * (p.groups :> int)))

  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    if p.transposed then transposed_output_shape ~x_shape ~weight_shape p
    else
      let open Core.Syntax in
      let* p = to_conv2d_params ~weight_shape p in
      Conv2d.output_shape ~x_shape ~weight_shape p

  let bias_shape ~(weight_shape : Vec6.shape) (p : params) =
    let channels =
      if p.transposed then
        Dim.extent ((Vec6.get weight_shape Axis.C :> int) * (p.groups :> int))
      else Vec6.get weight_shape Axis.N
    in
    Vec6.set (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) Axis.C channels

  module Compute (S : Semantics.SEMANTICS) = struct
    module C = Conv2d.Compute (S)

    let projected_pos ~(stride : Op_config.Pos.t) ~(pad : Op_config.Nonneg.t)
        ~(dilation : Op_config.Pos.t) input_pos kernel_pos =
      S.index_add
        (S.index_add
           (S.index_scale (stride :> int) (S.of_index input_pos))
           (S.index_scale (dilation :> int) (S.of_index kernel_pos)))
        (S.index_const (-(pad :> int)))

    let transposed_pixel (p : params) ~(x_shape : Vec6.shape)
        ~(weight_shape : Vec6.shape) ~x ~weight ~bias out =
      let in_per_group, out_per_group =
        or_invalid_arg (validate_transposed_channels ~x_shape ~weight_shape p)
      in
      let oc = Vec6.get out Axis.C in
      let group =
        if (p.groups :> int) = 1 then S.index_const 0
        else
          S.index_floor_div_pos (S.of_index oc)
            (Op_config.Pos.of_int out_per_group)
      in
      let local_oc =
        S.assume_index
          (S.index_add (S.of_index oc) (S.index_scale (-out_per_group) group))
      in
      let acc =
        S.sum ~lo:S.index_zero
          ~hi:(S.index_extent (Dim.extent in_per_group))
          (fun local_ic ->
            S.sum ~lo:S.index_zero
              ~hi:(S.index_extent (Vec6.get x_shape Axis.H))
              (fun ih ->
                S.sum ~lo:S.index_zero
                  ~hi:(S.index_extent (Vec6.get x_shape Axis.W))
                  (fun iw ->
                    S.sum ~lo:S.index_zero
                      ~hi:(S.index_extent (Vec6.get weight_shape Axis.H))
                      (fun kh ->
                        S.sum ~lo:S.index_zero
                          ~hi:(S.index_extent (Vec6.get weight_shape Axis.W))
                          (fun kw ->
                            let ic =
                              S.assume_index
                                (S.index_add
                                   (S.index_scale in_per_group group)
                                   (S.of_index local_ic))
                            in
                            let h_matches =
                              S.index_eq
                                (S.of_index (Vec6.get out Axis.H))
                                (projected_pos ~stride:p.stride.h
                                   ~pad:p.padding.h ~dilation:p.dilation.h ih kh)
                            in
                            let w_matches =
                              S.index_eq
                                (S.of_index (Vec6.get out Axis.W))
                                (projected_pos ~stride:p.stride.w
                                   ~pad:p.padding.w ~dilation:p.dilation.w iw kw)
                            in
                            let product =
                              S.mul
                                (S.load x
                                   (out |> Vec6.set_h ih |> Vec6.set_w iw
                                  |> Vec6.set_c ic))
                                (S.load weight
                                   (Vec6.make ~n:ic ~t:S.index_zero
                                      ~d:S.index_zero ~h:kh ~w:kw ~c:local_oc))
                            in
                            S.select h_matches
                              (S.select w_matches product (S.const 0.))
                              (S.const 0.))))))
      in
      S.add acc
        (S.load bias
           (Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
              ~h:S.index_zero ~w:S.index_zero ~c:oc))

    let pixel (p : params) ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
        ~x ~weight ~bias out =
      if p.transposed then
        transposed_pixel p ~x_shape ~weight_shape ~x ~weight ~bias out
      else
        C.pixel
          (or_invalid_arg (to_conv2d_params ~weight_shape p))
          ~x_shape ~weight_shape ~x ~weight ~bias out
  end
end
