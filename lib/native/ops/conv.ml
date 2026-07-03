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
    let in_channels = (p.in_channels :> int) in
    let groups = (p.groups :> int) in
    let out_channels = (Vec6.get weight_shape Axis.N :> int) in
    let weight_in_per_group = (Vec6.get weight_shape Axis.C :> int) in
    if in_channels mod groups <> 0 then
      invalid_arg "Conv2d: in_channels must be divisible by groups";
    if out_channels mod groups <> 0 then
      invalid_arg "Conv2d: out_channels must be divisible by groups";
    let in_per_group = in_channels / groups in
    if weight_in_per_group <> in_per_group then
      invalid_arg
        "Conv2d: weight C extent must equal in_channels divided by groups";
    if not (Dim.equal (Vec6.get weight_shape Axis.H) p.h.kernel) then
      invalid_arg "Conv2d: weight H extent must equal h.kernel";
    if not (Dim.equal (Vec6.get weight_shape Axis.W) p.w.kernel) then
      invalid_arg "Conv2d: weight W extent must equal w.kernel";
    (in_per_group, out_channels / groups)

  (* N/T/D pass through; H/W shrink via [Window_axis.output_extent]; C = Cout from
     weight_shape. *)
  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    let _in_per_group, _out_per_group = validate_channels ~weight_shape p in
    if not (Dim.equal (Vec6.get x_shape Axis.C) p.in_channels) then
      invalid_arg "Conv2d: input C extent must equal in_channels";
    let out_extent axis (w : axis_window) =
      Window_axis.output_extent ~in_extent:(Vec6.get x_shape axis)
        ~kernel:w.kernel ~stride:w.stride ~pad_before:w.pad_before
        ~pad_after:w.pad_after ~dilation:w.dilation
    in
    (* Start from the input shape (N/T/D pass through unchanged) and replace only
       the axes the op resizes — all in extent-space, no [:> int] round-trips. *)
    Vec6.set
      (Vec6.set
         (Vec6.set x_shape Axis.H (out_extent Axis.H p.h))
         Axis.W (out_extent Axis.W p.w))
      Axis.C
      (Vec6.get weight_shape Axis.N)

  module Compute (S : Semantics.SEMANTICS) = struct
    module Wa = Window_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
        ~x ~weight ~bias (out : Axis.t -> Semantics.position S.index) =
      let in_per_group, out_per_group = validate_channels ~weight_shape p in
      let oc = out Axis.C in
      let group =
        if (p.groups :> int) = 1 then S.index_const 0
        else
          S.index_floor_div_pos (S.of_index oc)
            (Op_config.Pos.of_int out_per_group)
      in
      let wh =
        Wa.window ~kernel:p.h.kernel ~stride:p.h.stride
          ~pad_before:p.h.pad_before ~dilation:p.h.dilation
          ~in_extent:(Vec6.get x_shape Axis.H) (out Axis.H)
      in
      let ww =
        Wa.window ~kernel:p.w.kernel ~stride:p.w.stride
          ~pad_before:p.w.pad_before ~dilation:p.w.dilation
          ~in_extent:(Vec6.get x_shape Axis.W) (out Axis.W)
      in
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
                      | Axis.C -> local_ic
                      | _ -> S.index_zero
                    in
                    S.mul (S.load x x_idx) (S.load weight w_idx))))
      in
      S.add acc
        (S.load bias (fun a -> match a with Axis.C -> oc | _ -> S.index_zero))
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
      invalid_arg
        "Conv2d_padding: padding=\"same\" is not supported for strided \
         convolutions";
    let total = (dilation :> int) * ((kernel :> int) - 1) in
    ( Op_config.Nonneg.of_int (total / 2),
      Op_config.Nonneg.of_int (total - (total / 2)) )

  let axis_window ~(padding : padding) ~(kernel : Dim.extent Dim.t)
      ~(stride : Op_config.Pos.t) ~(dilation : Op_config.Pos.t) :
      Conv2d.axis_window =
    let pad_before, pad_after =
      match padding with
      | Valid -> (Op_config.Nonneg.of_int 0, Op_config.Nonneg.of_int 0)
      | Same -> same_padding ~kernel ~stride ~dilation
    in
    { Conv2d.kernel; stride; pad_before; pad_after; dilation }

  let to_conv2d_params ~(weight_shape : Vec6.shape) (p : params) : Conv2d.params
      =
    let groups = (p.groups :> int) in
    let in_channels = (Vec6.get weight_shape Axis.C :> int) * groups in
    {
      Conv2d.h =
        axis_window ~padding:p.padding
          ~kernel:(Vec6.get weight_shape Axis.H)
          ~stride:p.stride.h ~dilation:p.dilation.h;
      w =
        axis_window ~padding:p.padding
          ~kernel:(Vec6.get weight_shape Axis.W)
          ~stride:p.stride.w ~dilation:p.dilation.w;
      in_channels = Dim.extent in_channels;
      groups = p.groups;
    }

  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    Conv2d.output_shape ~x_shape ~weight_shape
      (to_conv2d_params ~weight_shape p)

  module Compute (S : Semantics.SEMANTICS) = struct
    module C = Conv2d.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
        ~x ~weight ~bias out =
      C.pixel
        (to_conv2d_params ~weight_shape p)
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
    if p.transposed then
      invalid_arg "Convolution: transposed convolutions are not supported";
    if (p.output_padding.h :> int) <> 0 || (p.output_padding.w :> int) <> 0 then
      invalid_arg
        "Convolution: output_padding must be zero for non-transposed \
         convolution"

  let axis_window ~kernel ~stride ~pad ~dilation : Conv2d.axis_window =
    { Conv2d.kernel; stride; pad_before = pad; pad_after = pad; dilation }

  let to_conv2d_params ~(weight_shape : Vec6.shape) (p : params) : Conv2d.params
      =
    require_forward_2d p;
    let groups = (p.groups :> int) in
    let in_channels = (Vec6.get weight_shape Axis.C :> int) * groups in
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
      invalid_arg "Convolution: transposed output extent must be positive";
    Dim.extent out

  let validate_transposed_channels ~(x_shape : Vec6.shape)
      ~(weight_shape : Vec6.shape) (p : params) =
    let groups = (p.groups :> int) in
    let in_channels = (Vec6.get x_shape Axis.C :> int) in
    let weight_in_channels = (Vec6.get weight_shape Axis.N :> int) in
    let out_per_group = (Vec6.get weight_shape Axis.C :> int) in
    if weight_in_channels <> in_channels then
      invalid_arg
        "Convolution: transposed weight N extent must equal input C extent";
    if in_channels mod groups <> 0 then
      invalid_arg "Convolution: input C extent must be divisible by groups";
    (in_channels / groups, out_per_group)

  let transposed_output_shape ~(x_shape : Vec6.shape)
      ~(weight_shape : Vec6.shape) (p : params) =
    let _in_per_group, out_per_group =
      validate_transposed_channels ~x_shape ~weight_shape p
    in
    Vec6.set
      (Vec6.set
         (Vec6.set x_shape Axis.H
            (transposed_output_axis ~in_extent:(Vec6.get x_shape Axis.H)
               ~kernel:(Vec6.get weight_shape Axis.H)
               ~stride:p.stride.h ~pad:p.padding.h ~dilation:p.dilation.h
               ~output_padding:p.output_padding.h))
         Axis.W
         (transposed_output_axis ~in_extent:(Vec6.get x_shape Axis.W)
            ~kernel:(Vec6.get weight_shape Axis.W)
            ~stride:p.stride.w ~pad:p.padding.w ~dilation:p.dilation.w
            ~output_padding:p.output_padding.w))
      Axis.C
      (Dim.extent (out_per_group * (p.groups :> int)))

  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    if p.transposed then transposed_output_shape ~x_shape ~weight_shape p
    else
      Conv2d.output_shape ~x_shape ~weight_shape
        (to_conv2d_params ~weight_shape p)

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
        validate_transposed_channels ~x_shape ~weight_shape p
      in
      let oc = out Axis.C in
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
                                (S.of_index (out Axis.H))
                                (projected_pos ~stride:p.stride.h
                                   ~pad:p.padding.h ~dilation:p.dilation.h ih kh)
                            in
                            let w_matches =
                              S.index_eq
                                (S.of_index (out Axis.W))
                                (projected_pos ~stride:p.stride.w
                                   ~pad:p.padding.w ~dilation:p.dilation.w iw kw)
                            in
                            let product =
                              S.mul
                                (S.load x (fun a ->
                                     match a with
                                     | Axis.H -> ih
                                     | Axis.W -> iw
                                     | Axis.C -> ic
                                     | _ -> out a))
                                (S.load weight (fun a ->
                                     match a with
                                     | Axis.N -> ic
                                     | Axis.H -> kh
                                     | Axis.W -> kw
                                     | Axis.C -> local_oc
                                     | _ -> S.index_zero))
                            in
                            S.select h_matches
                              (S.select w_matches product (S.const 0.))
                              (S.const 0.))))))
      in
      S.add acc
        (S.load bias (fun a -> match a with Axis.C -> oc | _ -> S.index_zero))

    let pixel (p : params) ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
        ~x ~weight ~bias out =
      if p.transposed then
        transposed_pixel p ~x_shape ~weight_shape ~x ~weight ~bias out
      else
        C.pixel
          (to_conv2d_params ~weight_shape p)
          ~x_shape ~weight_shape ~x ~weight ~bias out
  end
end
