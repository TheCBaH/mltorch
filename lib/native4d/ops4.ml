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
   - a convolution whose grouping is not one of the two forms a plain
     parameter would let a graph misrepresent as each other: [Conv2d] means
     one group and [Depthwise_conv2d] means one input channel per group, so
     neither constructor stores a caller-supplied [groups]. [Grouped_conv2d]
     is the general form, and there [groups] genuinely is a parameter — every
     count is legal, so no constructor split is protecting an illegal state. *)

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

(* Elementwise-maximum twin of [Mean_keepdims], same keep-dimensions-only
   representation and the same reason: .ai/native4d_design.md §1's reduction
   rule is not specific to the mean. A Native [Amax keepdim=false] legalizes to
   this plus a [Reshape4], exactly as [Mean keepdim=false] does (correction
   C1). *)
module Max_keepdims = struct
  type params = { dims : Axis4.t list }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"max_keepdims_params" (fun dims -> { dims })
    |> Jsont.Object.mem "dims" (Jsont.list Axis4.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis4.pp))
      p.dims

  type t = { params : params; x : Tensor_ref.t }

  let name = "MaxKeepDims"

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
    Fmt.pf fmt "@[<hv 2>max_keepdims@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params
end

(* Plain-summation twin of [Mean_keepdims], same keep-dimensions-only
   representation and the same reason: .ai/native4d_design.md §1's reduction
   rule is not specific to the mean. A Native [Sum keepdim=false] legalizes to
   this plus a [Reshape4], exactly as [Mean keepdim=false] does (correction
   C1). *)
module Sum_keepdims = struct
  type params = { dims : Axis4.t list }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"sum_keepdims_params" (fun dims -> { dims })
    |> Jsont.Object.mem "dims" (Jsont.list Axis4.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis4.pp))
      p.dims

  type t = { params : params; x : Tensor_ref.t }

  let name = "SumKeepDims"

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
    Fmt.pf fmt "@[<hv 2>sum_keepdims@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params
end

(* L2-vector-norm twin of [Mean_keepdims]/[Max_keepdims]: same
   keep-dimensions-only representation, for the same reason. A Native
   [Vector_norm keepdim=false] legalizes to this plus a [Reshape4], exactly
   as [Mean]/[Amax] do (correction C1). *)
module Vector_norm_keepdims = struct
  type params = { dims : Axis4.t list }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"vector_norm_keepdims_params" (fun dims -> { dims })
    |> Jsont.Object.mem "dims" (Jsont.list Axis4.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis4.pp))
      p.dims

  type t = { params : params; x : Tensor_ref.t }

  let name = "VectorNormKeepDims"

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
    Fmt.pf fmt "@[<hv 2>vector_norm_keepdims@ x=%a@ params=%a@]" pp_ref t.x
      pp_params t.params
end

(* ---- normalisation -------------------------------------------------------- *)

(* Training batch norm cannot use inference [Batch_norm]'s depthwise-conv
   legalization: its mean and inverse standard deviation are reductions of the
   live input and are exposed as two additional outputs.  Keeping the channel
   as [Axis4.t] makes T/D unrepresentable in this dialect. *)
module Batch_norm_no_stats = struct
  type params = { channel : Axis4.t; eps : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"batch_norm_no_stats4_params" (fun channel eps ->
        { channel; eps })
    |> Jsont.Object.mem "channel" Axis4.jsont ~enc:(fun p -> p.channel)
    |> Jsont.Object.mem "eps" Json_util.f32_jsont ~enc:(fun p -> p.eps)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{channel=%a;@ eps=%a}@]" Axis4.pp p.channel Fmt.float
      p.eps

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t option;
    bias : Tensor_ref.t option;
  }

  let name = "BatchNormNoStats"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          x = get "x" Tensor_ref.jsont;
          weight = Json_util.opt_field ms "weight" Tensor_ref.jsont;
          bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt k = function None -> [] | Some r -> [ (k, ref_ r) ] in
        Json_util.jobj
          (opt "bias" t.bias @ opt "weight" t.weight
          @ [ ("params", Json_util.enc params_jsont t.params); ("x", ref_ t.x) ]
          ))
      Jsont.json

  let operands (t : t) =
    (t.x :: Option.to_list t.weight) @ Option.to_list t.bias

  let map_operands f (t : t) =
    {
      t with
      x = f t.x;
      weight = Option.map f t.weight;
      bias = Option.map f t.bias;
    }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>batch_norm_no_stats@ x=%a%a%a@ params=%a@]" pp_ref t.x
      (Fmt.option (fun fmt w -> Fmt.pf fmt "@ weight=%a" pp_ref w))
      t.weight
      (Fmt.option (fun fmt b -> Fmt.pf fmt "@ bias=%a" pp_ref b))
      t.bias pp_params t.params
end

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

(* Retained fused for the same reason [Rms_norm] is, and with one more: the
   decomposition needs the MEAN twice -- once to centre and once inside the
   variance -- so writing it out would either recompute the reduction or
   materialise it, and both move an f32 rounding boundary the fused form does
   not have. The claim stays [Identical].

   Both affine operands stay [Tensor_ref.t option] rather than being filled
   here: [Eval_op4] fills them exactly as [Eval_op] does, so the two dialects
   agree on what an absent operand means. *)
module Layer_norm = struct
  type params = { dims : Axis4.t list; eps : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"layer_norm4_params" (fun dims eps -> { dims; eps })
    |> Jsont.Object.mem "dims" (Jsont.list Axis4.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.mem "eps" Json_util.f32_jsont ~enc:(fun p -> p.eps)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a;@ eps=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis4.pp))
      p.dims Fmt.float p.eps

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t option;
    bias : Tensor_ref.t option;
  }

  let name = "LayerNorm"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          x = get "x" Tensor_ref.jsont;
          weight = Json_util.opt_field ms "weight" Tensor_ref.jsont;
          bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt k = function None -> [] | Some r -> [ (k, ref_ r) ] in
        Json_util.jobj
          (opt "bias" t.bias @ opt "weight" t.weight
          @ [ ("params", Json_util.enc params_jsont t.params); ("x", ref_ t.x) ]
          ))
      Jsont.json

  let operands (t : t) =
    (t.x :: Option.to_list t.weight) @ Option.to_list t.bias

  let map_operands f (t : t) =
    {
      t with
      x = f t.x;
      weight = Option.map f t.weight;
      bias = Option.map f t.bias;
    }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>layer_norm@ x=%a%a%a@ params=%a@]" pp_ref t.x
      (Fmt.option (fun fmt w -> Fmt.pf fmt "@ weight=%a" pp_ref w))
      t.weight
      (Fmt.option (fun fmt b -> Fmt.pf fmt "@ bias=%a" pp_ref b))
      t.bias pp_params t.params
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

(* ---- boundary synthesis --------------------------------------------------- *)

(* Native's [Pad] with its axes narrowed to the dialect's four. Its own payload
   for the FIRST reason in .ai/native4d_add_op.md -- it names axes, so the field
   is [Axis4.t] and T/D are unsayable in a four-axis graph.

   The MODE stays a parameter rather than becoming a choice of constructor,
   which is the third trigger in that file and does not apply here: the dialect
   supports both modes Native does, so there is no value the constructor would
   be making unrepresentable. Splitting [Pad4] into [Constant_pad4]/[Reflect_pad4]
   would be the right move only if Native4D supported fewer modes than Native.

   The amounts are Native's own [Pad.Pad.entry] -- signed, so cropping crosses
   unchanged -- and the mode is Native's [Pad.Pad.mode]. Neither names an axis
   nor carries a shape, so restating them here would be a second definition free
   to drift from the [Compute] that reads them. *)
module Pad4 = struct
  type params = { pads : (Axis4.t * Pad.Pad.entry) list; mode : Pad.Pad.mode }

  let entry_jsont : (Axis4.t * Pad.Pad.entry) Jsont.t =
    Jsont.Object.map ~kind:"pad4_entry" (fun axis before after ->
        (axis, { Pad.Pad.before; after }))
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:fst
    |> Jsont.Object.mem "before" Jsont.int ~enc:(fun (_, e) -> e.Pad.Pad.before)
    |> Jsont.Object.mem "after" Jsont.int ~enc:(fun (_, e) -> e.Pad.Pad.after)
    |> Jsont.Object.finish

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"pad4_params" (fun pads mode -> { pads; mode })
    |> Jsont.Object.mem "pads" (Jsont.list entry_jsont) ~enc:(fun p -> p.pads)
    |> Jsont.Object.mem "mode" Pad.Pad.mode_jsont ~enc:(fun p -> p.mode)
    |> Jsont.Object.finish

  let pp_entry fmt (axis, (e : Pad.Pad.entry)) =
    Fmt.pf fmt "@[<h>%a:%d,%d@]" Axis4.pp axis e.Pad.Pad.before e.Pad.Pad.after

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{pads=%a mode=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_entry))
      p.pads Pad.Pad.pp_mode p.mode

  type t = { params : params; x : Tensor_ref.t }

  let name = "Pad4"

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
    Fmt.pf fmt "@[<hv 2>pad4@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params
end

(* ---- selection ------------------------------------------------------------ *)

(* Native's [Slice] with its axis narrowed to the dialect's four. Its own payload
   for the FIRST reason in .ai/native4d_add_op.md -- it names an axis, so the
   field is [Axis4.t] and T/D are unsayable in a four-axis graph.

   The BOUNDS are Native's own and cross unchanged: they are canonical by the
   time any payload exists, and nothing about narrowing the axis set changes
   what [0 <= start <= stop <= extent] means. Restating the shape rule or the
   pixel map here would be a second definition free to drift from the [Compute]
   that reads them, so both are delegated to [Split.Slice]. *)
module Slice4 = struct
  type params = {
    axis : Axis4.t;
    start : int;
    stop : int;
    step : Op_config.Pos.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"slice4_params" (fun axis start stop step ->
        { axis; start; stop; step = Op_config.Pos.of_int step })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "start" Jsont.int ~enc:(fun p -> p.start)
    |> Jsont.Object.mem "stop" Jsont.int ~enc:(fun p -> p.stop)
    |> Jsont.Object.mem "step" Jsont.int ~enc:(fun p -> (p.step :> int))
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a start=%d stop=%d step=%d}@]" Axis4.pp p.axis
      p.start p.stop
      (p.step :> int)

  type t = { params : params; x : Tensor_ref.t }

  let name = "Slice4"

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
    Fmt.pf fmt "@[<hv 2>slice4@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params
end

(* ---- joining ---------------------------------------------------------------

   Native's [Concat] with its axis narrowed to the dialect's four. Its own
   payload for the FIRST reason in .ai/native4d_add_op.md -- it names an axis,
   so the field is [Axis4.t] and T/D are unsayable in a four-axis graph.

   [xs] is a variadic OPERAND list, unlike every other op above except
   [Layer_norm]/[Depthwise_conv2d]'s fixed few -- the same "operand count is
   not just output count" case Native's own [Concat] documents. Nothing here
   restates the shape rule or the pixel map: both delegate to [Concat.Concat],
   which is what keeps them from drifting apart. *)
module Concat4 = struct
  type params = { axis : Axis4.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"concat4_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis4.pp p.axis

  type t = { params : params; xs : Tensor_ref.t list }

  let name = "Concat4"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          xs = get "xs" (Jsont.list Tensor_ref.jsont);
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ( "xs",
              Json_util.jarr (List.map (Json_util.enc Tensor_ref.jsont) t.xs) );
          ])
      Jsont.json

  let operands (t : t) = t.xs
  let map_operands f (t : t) = { t with xs = List.map f t.xs }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>concat4@ xs=%a@ params=%a@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_ref))
      t.xs pp_params t.params
end

(* ---- splitting ------------------------------------------------------------ *)

(* The dialect's first MULTI-OUTPUT op: one output per coordinate of [axis], so
   its arity comes from the operand's signature rather than from the op.

   Needs its own payload for the first of the three reasons in
   .ai/native4d_add_op.md — it NAMES AN AXIS, so the field is [Axis4.t] and T/D
   are unsayable. Everything else is Native's [Split.Unbind]: the shape rule and
   the per-pixel algorithm are delegated, never restated.

   The representable set is narrower than it looks, and [Shape4.of_vec6] is what
   enforces it rather than any rule written here. Dropping an axis shifts every
   axis OUTSIDE it one place outward, so a four-axis [N,H,W,C] input unbound
   along H/W/C puts N's extent on T. That is inside the dialect only when N=1 —
   a rank-3 source is the common way to get there, but any batch-1 input
   qualifies. Unbinding N always converts, since nothing is outside it. *)
module Unbind = struct
  type params = { axis : Axis4.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"unbind_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis4.pp p.axis

  type t = { params : params; x : Tensor_ref.t }

  let name = "Unbind"

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
    Fmt.pf fmt "@[<hv 2>unbind@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params
end

(* The four-axis counterpart of [Factory.Zeros].  Its result shape is typed as
   [Shape4.t], so a factory cannot smuggle a T/D extent into this dialect. *)
module Zeros4 = struct
  type params = { shape : Shape4.t; fmt : Payload.packed_fmt }
  type t = { params : params }

  let name = "Zeros4"

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"zeros4_params" (fun shape fmt -> { shape; fmt })
    |> Jsont.Object.mem "shape" Shape4.jsont ~enc:(fun p -> p.shape)
    |> Jsont.Object.mem "fmt" Payload.packed_fmt_jsont ~enc:(fun p -> p.fmt)
    |> Jsont.Object.finish

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        { params = Json_util.req_field ms "params" params_jsont name })
      ~enc:(fun t ->
        Json_util.jobj [ ("params", Json_util.enc params_jsont t.params) ])
      Jsont.json

  let operands _ = []
  let map_operands _ t = t

  let pp _ fmt (t : t) =
    let (Payload.Fmt elt) = t.params.fmt in
    Fmt.pf fmt "@[<hv 2>zeros4@ shape=%a@ fmt=%s@]" Shape4.pp t.params.shape
      (Payload.fmt_name elt)
end

(* Four-axis form of [Factory.Arange].  A rank-one ATen range arrives as C,
 * leaving N/H/W unit, so it remains a valid Native4D factory without an
 * artificial reshape. *)
module Arange4 = struct
  type params = {
    start : float;
    stop : float;
    step : float;
    fmt : Payload.packed_fmt;
  }

  type t = { params : params }

  let name = "Arange4"

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"arange4_params" (fun start stop step fmt ->
        { start; stop; step; fmt })
    |> Jsont.Object.mem "start" Json_util.f32_jsont ~enc:(fun p -> p.start)
    |> Jsont.Object.mem "stop" Json_util.f32_jsont ~enc:(fun p -> p.stop)
    |> Jsont.Object.mem "step" Json_util.f32_jsont ~enc:(fun p -> p.step)
    |> Jsont.Object.mem "fmt" Payload.packed_fmt_jsont ~enc:(fun p -> p.fmt)
    |> Jsont.Object.finish

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        { params = Json_util.req_field ms "params" params_jsont name })
      ~enc:(fun t ->
        Json_util.jobj [ ("params", Json_util.enc params_jsont t.params) ])
      Jsont.json

  let operands _ = []
  let map_operands _ t = t

  let pp _ fmt (t : t) =
    let (Payload.Fmt elt) = t.params.fmt in
    Fmt.pf fmt "@[<hv 2>arange4@ start=%g@ stop=%g@ step=%g@ fmt=%s@]"
      t.params.start t.params.stop t.params.step (Payload.fmt_name elt)
end
