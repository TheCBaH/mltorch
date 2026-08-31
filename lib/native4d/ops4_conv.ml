(* The convolution family of [Ops4] payloads, split out of ops4.ml under the
   tracked file-size ceiling (scripts/check-file-size.sh). [Ops4] re-exports
   every module below by the same name, so external references such as
   [Ops4.Conv_params] or [Ops4.Grouped_conv2d] are unaffected.

   Only two things force a new payload:

   - an op that NAMES AXES, which must name [Axis4.t] rather than [Axis.t] so
     that T and D are unnameable;
   - a convolution whose grouping is not one of the two forms a plain
     parameter would let a graph misrepresent as each other: [Conv2d] means
     one group and [Depthwise_conv2d] means one input channel per group, so
     neither constructor stores a caller-supplied [groups]. [Grouped_conv2d]
     is the general form, and there [groups] genuinely is a parameter — every
     count is legal, so no constructor split is protecting an illegal state. *)

(* [Conv.Conv2d.axis_window] verbatim — kernel, stride, padding and dilation are
   window arithmetic, not axis naming. The difference from
   [Conv.Conv2d.params] is the absence of [groups]: [Conv2d] means one group and
   [Depthwise_conv2d] means one input channel per group, and neither is a value
   a caller supplies. *)
module Conv_params = struct
  type t = {
    h : Conv.Conv2d.axis_window;
    w : Conv.Conv2d.axis_window;
    in_channels : Dim.extent Dim.t;
  }

  let jsont : t Jsont.t =
    Jsont.Object.map ~kind:"conv4_params" (fun h w in_channels ->
        { h; w; in_channels })
    |> Jsont.Object.mem "h" Conv.Conv2d.axis_window_jsont ~enc:(fun p -> p.h)
    |> Jsont.Object.mem "w" Conv.Conv2d.axis_window_jsont ~enc:(fun p -> p.w)
    |> Jsont.Object.mem "in_channels" Dim.extent_jsont ~enc:(fun p ->
        p.in_channels)
    |> Jsont.Object.finish

  let pp fmt (p : t) =
    Fmt.pf fmt "@[<hv>{h=%a;@ w=%a;@ in_channels=%a}@]"
      Conv.Conv2d.pp_axis_window p.h Conv.Conv2d.pp_axis_window p.w Dim.pp
      p.in_channels
end

(* Shared by every forward convolution: same operands, same layout, and the
   weight is Native's unchanged [Cout,1,1,Kh,Kw,Cin/groups], so no permutation
   is needed around the shared compute call. Parameterized over the params
   module so [Conv2d]/[Depthwise_conv2d] (no [groups] field at all) and
   [Grouped_conv2d] (a real [groups] field) share everything but that type. *)
module Make_conv_payload (Params : sig
  type t

  val jsont : t Jsont.t
  val pp : t Fmt.t
end) =
struct
  type t = {
    params : Params.t;
    x : Tensor_ref.t;
    weight : Tensor_ref.t;
    bias : Tensor_ref.t option;
  }

  let jsont ~name : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" Params.jsont;
          x = get "x" Tensor_ref.jsont;
          weight = get "weight" Tensor_ref.jsont;
          bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        Json_util.jobj
          ((match t.bias with None -> [] | Some r -> [ ("bias", ref_ r) ])
          @ [
              ("params", Json_util.enc Params.jsont t.params);
              ("weight", ref_ t.weight);
              ("x", ref_ t.x);
            ]))
      Jsont.json

  let operands (t : t) = [ t.x; t.weight ] @ Option.to_list t.bias

  let map_operands f (t : t) =
    { t with x = f t.x; weight = f t.weight; bias = Option.map f t.bias }

  let pp label (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>%s@ x=%a@ weight=%a%a@ params=%a@]" label pp_ref t.x
      pp_ref t.weight
      (Fmt.option (fun fmt b -> Fmt.pf fmt "@ bias=%a" pp_ref b))
      t.bias Params.pp t.params
end

module Conv_payload = Make_conv_payload (Conv_params)

module Conv2d = struct
  include Conv_payload

  let name = "Conv2D"
  let jsont = Conv_payload.jsont ~name
  let pp pp_ref fmt t = Conv_payload.pp "conv2d" pp_ref fmt t
end

module Depthwise_conv2d = struct
  include Conv_payload

  let name = "DepthwiseConv2D"
  let jsont = Conv_payload.jsont ~name
  let pp pp_ref fmt t = Conv_payload.pp "depthwise_conv2d" pp_ref fmt t
end

(* The general form: every count is a legal [groups], so unlike [Conv2d] and
   [Depthwise_conv2d] there is no illegal state a constructor split would be
   protecting, and [groups] is an ordinary field. *)
module Grouped_conv_params = struct
  type t = {
    h : Conv.Conv2d.axis_window;
    w : Conv.Conv2d.axis_window;
    in_channels : Dim.extent Dim.t;
    groups : Op_config.Pos.t;
  }

  let jsont : t Jsont.t =
    Jsont.Object.map ~kind:"grouped_conv4_params" (fun h w in_channels groups ->
        { h; w; in_channels; groups })
    |> Jsont.Object.mem "h" Conv.Conv2d.axis_window_jsont ~enc:(fun p -> p.h)
    |> Jsont.Object.mem "w" Conv.Conv2d.axis_window_jsont ~enc:(fun p -> p.w)
    |> Jsont.Object.mem "in_channels" Dim.extent_jsont ~enc:(fun p ->
        p.in_channels)
    |> Jsont.Object.mem "groups" Op_config.Pos.jsont ~enc:(fun p -> p.groups)
    |> Jsont.Object.finish

  let pp fmt (p : t) =
    Fmt.pf fmt "@[<hv>{h=%a;@ w=%a;@ in_channels=%a;@ groups=%a}@]"
      Conv.Conv2d.pp_axis_window p.h Conv.Conv2d.pp_axis_window p.w Dim.pp
      p.in_channels Op_config.Pos.pp p.groups
end

module Grouped_conv_payload = Make_conv_payload (Grouped_conv_params)

module Grouped_conv2d = struct
  include Grouped_conv_payload

  let name = "GroupedConv2D"
  let jsont = Grouped_conv_payload.jsont ~name
  let pp pp_ref fmt t = Grouped_conv_payload.pp "grouped_conv2d" pp_ref fmt t
end

(* Transposed convolution keeps [output_padding], which only exists for the
   transposed direction, and drops [transposed] (the constructor says it) and
   [groups] (the dialect has none). Its weight is [Cin,1,1,Kh,Kw,Cout], again
   Native's layout unchanged. *)
module Transposed_conv2d = struct
  type params = {
    stride : Op_config.Pos.t Op_config.Hw.t;
    padding : Op_config.Nonneg.t Op_config.Hw.t;
    dilation : Op_config.Pos.t Op_config.Hw.t;
    output_padding : Op_config.Nonneg.t Op_config.Hw.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"transposed_conv4_params"
      (fun stride padding dilation output_padding ->
        { stride; padding; dilation; output_padding })
    |> Jsont.Object.mem "stride" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.stride)
    |> Jsont.Object.mem "padding" (Op_config.Hw.jsont Op_config.Nonneg.jsont)
         ~enc:(fun p -> p.padding)
    |> Jsont.Object.mem "dilation" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.dilation)
    |> Jsont.Object.mem "output_padding"
         (Op_config.Hw.jsont Op_config.Nonneg.jsont) ~enc:(fun p ->
           p.output_padding)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt
      "@[<hv>{stride=%a;@ padding=%a;@ dilation=%a;@ output_padding=%a}@]"
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.stride
      (Op_config.Hw.pp Op_config.Nonneg.pp)
      p.padding
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.dilation
      (Op_config.Hw.pp Op_config.Nonneg.pp)
      p.output_padding

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t;
    bias : Tensor_ref.t option;
  }

  let name = "TransposedConv2D"

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
        Json_util.jobj
          ((match t.bias with None -> [] | Some r -> [ ("bias", ref_ r) ])
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
    Fmt.pf fmt "@[<hv 2>transposed_conv2d@ x=%a@ weight=%a%a@ params=%a@]"
      pp_ref t.x pp_ref t.weight
      (Fmt.option (fun fmt b -> Fmt.pf fmt "@ bias=%a" pp_ref b))
      t.bias pp_params t.params
end
