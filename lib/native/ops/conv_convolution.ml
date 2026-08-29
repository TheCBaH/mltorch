(* Split out of conv.ml see conv.ml (the facade). Depends on
   Conv2d, which it reuses for shape and the Walk config space (a
   non-transposed convolution is shape-equivalent to Conv2d). *)

open Conv_conv2d

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
    let open Err.Syntax in
    let* () =
      if p.transposed then
        Err.fail (`Convolution Shape_error.Convolution.Transposed_not_supported)
      else Err.return ()
    in
    if (p.output_padding.h :> int) <> 0 || (p.output_padding.w :> int) <> 0 then
      Err.fail
        (`Convolution
           (Shape_error.Convolution.Output_padding_nonzero
              Shape_error.Convolution.
                { h = p.output_padding.h; w = p.output_padding.w }))
    else Err.return ()

  let axis_window ~kernel ~stride ~pad ~dilation : Conv2d.axis_window =
    { Conv2d.kernel; stride; pad_before = pad; pad_after = pad; dilation }

  let to_conv2d_params ~(weight_shape : Vec6.shape) (p : params) :
      (Conv2d.params, Shape_error.t) Err.t =
    let open Err.Syntax in
    let* () = require_forward_2d p in
    let groups = (p.groups :> int) in
    let in_channels = (Vec6.get weight_shape Axis.C :> int) * groups in
    Err.return
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
      Err.fail
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
    else Err.return (Dim.extent out)

  let validate_transposed_channels ~(x_shape : Vec6.shape)
      ~(weight_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let groups = (p.groups :> int) in
    let in_channels = (Vec6.get x_shape Axis.C :> int) in
    let weight_in_channels = (Vec6.get weight_shape Axis.N :> int) in
    let out_per_group = (Vec6.get weight_shape Axis.C :> int) in
    let* () =
      if weight_in_channels <> in_channels then
        Err.fail
          (`Convolution
             (Shape_error.Convolution.Transposed_weight_input_mismatch
                Shape_error.Convolution.
                  {
                    weight_input_channels = Vec6.get weight_shape Axis.N;
                    input_channels = Vec6.get x_shape Axis.C;
                  }))
      else Err.return ()
    in
    let* () =
      if in_channels mod groups <> 0 then
        Err.fail
          (`Convolution
             (Shape_error.Convolution
              .Transposed_input_channels_not_divisible_by_groups
                Shape_error.Convolution.{ channels = in_channels; groups }))
      else Err.return ()
    in
    Err.return (in_channels / groups, out_per_group)

  let transposed_output_shape ~(x_shape : Vec6.shape)
      ~(weight_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
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
      let open Err.Syntax in
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
