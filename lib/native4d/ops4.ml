(* Native4D payloads for the ops whose Native counterpart cannot be reused
   as-is. Everything else IS reused: [Pointwise.Add], [Pool.MaxPool2d] and
   friends already expose exactly the [name]/[jsont]/[operands]/[map_operands]/
   [pp] an op-registry entry wants, and their parameters name no axis and carry
   no shape, so there is nothing four-axis about them to restate. See
   .ai/native4d_add_op.md.

   Only two things force a new payload:

   - an op that NAMES AXES, which must name [Axis4.t] rather than [Axis.t] so
     that T and D are unnameable (Mean_keepdims, Permute4, Rms_norm);
   - an op that CARRIES A SHAPE, which must carry [Shape4.t] (Reshape4);
   - a convolution, whose grouping moves from a parameter into the choice of
     constructor: the dialect has no general grouped convolution, so [groups]
     is not a number the IR should be able to hold. *)

(* ---- convolution ---------------------------------------------------------- *)

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

(* Shared by the two forward convolutions: same operands, same layout, and the
   weight is Native's unchanged [Cout,1,1,Kh,Kw,Cin/groups], so no permutation
   is needed around the shared compute call. *)
module Conv_payload = struct
  type t = {
    params : Conv_params.t;
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
          params = get "params" Conv_params.jsont;
          x = get "x" Tensor_ref.jsont;
          weight = get "weight" Tensor_ref.jsont;
          bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        Json_util.jobj
          ((match t.bias with None -> [] | Some r -> [ ("bias", ref_ r) ])
          @ [
              ("params", Json_util.enc Conv_params.jsont t.params);
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
      t.bias Conv_params.pp t.params
end

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

(* ---- reduction ------------------------------------------------------------ *)

(* Keep-dimensions only: .ai/native4d_design.md §1 says reduction is represented
   ONLY in keep-dimensions form, so there is no [keepdim] field to set wrong. A
   Native [Mean keepdim=false] legalizes to this plus a [Reshape4], and only
   when the packed shape stays four-axis (correction C1). *)
module Mean_keepdims = struct
  type params = { dims : Axis4.t list }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"mean_keepdims_params" (fun dims -> { dims })
    |> Jsont.Object.mem "dims" (Jsont.list Axis4.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis4.pp))
      p.dims

  type t = { params : params; x : Tensor_ref.t }

  let name = "MeanKeepDims"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        { params = get "params" params_jsont; x = get "x" Tensor_ref.jsont })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("x", Json_util.enc Tensor_ref.jsont t.x);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>mean_keepdims@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params
end

(* ---- normalisation -------------------------------------------------------- *)

(* Retained fused rather than decomposed, so the legalization claims [Identical]
   and the verifier proves it structurally. The decomposition in §7.7
   materialises the squares before the reduction and so moves an f32 rounding
   boundary, which would make it [Equivalent] — weaker evidence for no gain,
   since the compute already exists. *)
module Rms_norm = struct
  type params = { dims : Axis4.t list; eps : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"rms_norm4_params" (fun dims eps -> { dims; eps })
    |> Jsont.Object.mem "dims" (Jsont.list Axis4.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.mem "eps" Json_util.f32_jsont ~enc:(fun p -> p.eps)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a;@ eps=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis4.pp))
      p.dims Fmt.float p.eps

  type t = { params : params; x : Tensor_ref.t; weight : Tensor_ref.t option }

  let name = "RmsNorm"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          x = get "x" Tensor_ref.jsont;
          weight = Json_util.opt_field ms "weight" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        Json_util.jobj
          ((match t.weight with None -> [] | Some w -> [ ("weight", ref_ w) ])
          @ [ ("params", Json_util.enc params_jsont t.params); ("x", ref_ t.x) ]
          ))
      Jsont.json

  let operands (t : t) = t.x :: Option.to_list t.weight

  let map_operands f (t : t) =
    { t with x = f t.x; weight = Option.map f t.weight }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>rms_norm@ x=%a%a@ params=%a@]" pp_ref t.x
      (Fmt.option (fun fmt w -> Fmt.pf fmt "@ weight=%a" pp_ref w))
      t.weight pp_params t.params
end

(* ---- data movement -------------------------------------------------------- *)

(* A four-axis bijection, as [(out_axis, in_axis)] pairs like Native's — but
   over [Axis4.t], so T and D cannot appear and the "does not use T or D as
   semantic axes" check has nothing left to check. Completing it to the six-axis
   perm shared compute wants is [Eval_op4]'s job: T and D map to themselves. *)
module Permute4 = struct
  type perm = (Axis4.t * Axis4.t) list

  let perm_jsont : perm Jsont.t =
    Jsont.list
      (Jsont.Object.map ~kind:"axis4_pair" (fun in_ax out_ax -> (out_ax, in_ax))
      |> Jsont.Object.mem "in" Axis4.jsont ~enc:snd
      |> Jsont.Object.mem "out" Axis4.jsont ~enc:fst
      |> Jsont.Object.finish)

  let lookup (perm : perm) out_axis =
    Option.value (List.assoc_opt out_axis perm) ~default:out_axis

  let of_fn f : perm = List.map (fun axis -> (axis, f axis)) Axis4.all
  let identity : perm = of_fn Fun.id

  let pp_perm fmt (perm : perm) =
    Fmt.brackets
      (Fmt.list ~sep:Fmt.comma (fun fmt (out_axis, in_axis) ->
           Fmt.pf fmt "@[<h>%a<-%a@]" Axis4.pp out_axis Axis4.pp in_axis))
      fmt
      (List.filter (fun (o, i) -> not (Axis4.equal o i)) perm)

  type t = { perm : perm; x : Tensor_ref.t }

  let name = "Permute4"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        { perm = get "perm" perm_jsont; x = get "x" Tensor_ref.jsont })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("perm", Json_util.enc perm_jsont t.perm);
            ("x", Json_util.enc Tensor_ref.jsont t.x);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>permute4@ x=%a@ perm=%a@]" pp_ref t.x pp_perm t.perm
end

(* Row-major reinterpretation, as Native's — but the target is a [Shape4.t], so
   a reshape cannot take a tensor out of the dialect. *)
module Reshape4 = struct
  type params = { shape : Shape4.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"reshape4_params" (fun shape -> { shape })
    |> Jsont.Object.mem "shape" Shape4.jsont ~enc:(fun p -> p.shape)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{shape=%a}@]" Shape4.pp p.shape

  type t = { params : params; x : Tensor_ref.t }

  let name = "Reshape4"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        { params = get "params" params_jsont; x = get "x" Tensor_ref.jsont })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("x", Json_util.enc Tensor_ref.jsont t.x);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>reshape4@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params
end
