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

(* ---- convolution ----------------------------------------------------------

   The whole family lives in ops4_conv.ml, split out under the tracked
   file-size ceiling; every alias below is exactly what was a top-level
   module here before the split, so external references such as
   [Ops4.Conv_params] or [Ops4.Grouped_conv2d] are unaffected. *)

module Conv_params = Ops4_conv.Conv_params
module Make_conv_payload = Ops4_conv.Make_conv_payload
module Conv_payload = Ops4_conv.Conv_payload
module Conv2d = Ops4_conv.Conv2d
module Depthwise_conv2d = Ops4_conv.Depthwise_conv2d
module Grouped_conv_params = Ops4_conv.Grouped_conv_params
module Grouped_conv_payload = Ops4_conv.Grouped_conv_payload
module Grouped_conv2d = Ops4_conv.Grouped_conv2d
module Transposed_conv2d = Ops4_conv.Transposed_conv2d

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

(* [channel] is [Axis4.t] for the same reason [Batch_norm_no_stats]'s is --
   naming an axis -- though the domain check restricts it to C, matching
   every importer today (`Norm.GroupNorm`'s own doc comment). [groups] is an
   ordinary field, not a constructor choice: every divisor of the channel
   extent is equally representable, so there is no illegal state a
   constructor split would be protecting, the same argument [Grouped_conv2d]
   makes for its own [groups]. *)
module Group_norm4 = struct
  type params = { channel : Axis4.t; groups : Op_config.Pos.t; eps : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"group_norm4_params" (fun channel groups eps ->
        { channel; groups; eps })
    |> Jsont.Object.mem "channel" Axis4.jsont ~enc:(fun p -> p.channel)
    |> Jsont.Object.mem "groups" Op_config.Pos.jsont ~enc:(fun p -> p.groups)
    |> Jsont.Object.mem "eps" Json_util.f32_jsont ~enc:(fun p -> p.eps)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{channel=%a;@ groups=%a;@ eps=%a}@]" Axis4.pp p.channel
      Op_config.Pos.pp p.groups Fmt.float p.eps

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t option;
    bias : Tensor_ref.t option;
  }

  let name = "GroupNorm4"

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
    Fmt.pf fmt "@[<hv 2>group_norm4@ x=%a%a%a@ params=%a@]" pp_ref t.x
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

(* [Expand]'s own payload for the same reason [Reshape4]'s is -- it CARRIES A
   SHAPE, so the target is [Shape4.t] rather than [Vec6.shape]: a broadcast
   that fanned an axis onto T or D would leave the dialect, and typing the
   target is what makes that unconstructible rather than merely unchecked.
   The domain therefore needs no per-axis gate of its own (unlike [Pad]'s
   axis-keyed entries): [Graph_shape4]'s [four] wrap and the lowerer's own
   [Shape4.of_vec6] already catch it, the same as [Reshape4]. *)
module Expand4 = struct
  type params = { size : Shape4.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"expand4_params" (fun size -> { size })
    |> Jsont.Object.mem "size" Shape4.jsont ~enc:(fun p -> p.size)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{size=%a}@]" Shape4.pp p.size

  type t = { params : params; x : Tensor_ref.t }

  let name = "Expand4"

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
    Fmt.pf fmt "@[<hv 2>expand4@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params
end

(* [Repeat]'s own payload for the same reason [Expand4]'s is -- it CARRIES A
   PER-AXIS MULTIPLIER over every axis (not one named axis, unlike
   [RepeatInterleave4] below), so the target is [Shape4.t] rather than
   [Vec6.shape]: a multiplier that tiled T or D away from 1 would leave the
   dialect, and typing the field is what makes that unconstructible rather
   than merely unchecked. [Shape4.t]'s own representation (a [Vec6.shape]
   with T/D pinned to the unit extent) is exactly the "repeats.T = 1,
   repeats.D = 1" condition a repeat multiplier needs -- reused as-is, not a
   coincidence of the underlying type. The domain therefore needs no
   per-axis gate of its own, the same as [Expand4]: [Graph_shape4]'s [four]
   wrap and the lowerer's own [Shape4.of_vec6] already catch it. *)
module Repeat4 = struct
  type params = { repeats : Shape4.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"repeat4_params" (fun repeats -> { repeats })
    |> Jsont.Object.mem "repeats" Shape4.jsont ~enc:(fun p -> p.repeats)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{repeats=%a}@]" Shape4.pp p.repeats

  type t = { params : params; x : Tensor_ref.t }

  let name = "Repeat4"

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
    Fmt.pf fmt "@[<hv 2>repeat4@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params
end

(* [RepeatInterleave]'s own payload, narrowed the same way [Select4]'s is:
   ONE named axis, so the field is [Axis4.t] and T/D are unsayable. Gets the
   same [check_dims]-style axis-domain rejection [Select4]/[Slice4] do, in
   [Domain.check_node] -- not strictly load-bearing the way it is for those
   two (which COLLAPSE their axis to 1 regardless of its input extent, so
   the blanket four-axis shape check cannot see a bad one; this op instead
   MULTIPLIES its axis's extent, so a T/D target > 1 is already caught
   there), but named-axis ops get one consistent diagnostic across the
   [Split]/[Repeat] family rather than a silent per-op difference in how
   good the error message is. *)
module RepeatInterleave4 = struct
  type params = { axis : Axis4.t; repeats : Op_config.Pos.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"repeat_interleave4_params" (fun axis repeats ->
        { axis; repeats = Op_config.Pos.of_int repeats })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "repeats" Jsont.int ~enc:(fun p -> (p.repeats :> int))
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a repeats=%d}@]" Axis4.pp p.axis (p.repeats :> int)

  type t = { params : params; x : Tensor_ref.t }

  let name = "RepeatInterleave4"

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
    Fmt.pf fmt "@[<hv 2>repeat_interleave4@ x=%a@ params=%a@]" pp_ref t.x
      pp_params t.params
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

(* ---- selection, joining, splitting ----------------------------------------

   [Slice4]/[Select4]/[Concat4]/[Unbind]/[Split_with_sizes4] live in
   ops4_split.ml, split out under the tracked file-size ceiling
   (scripts/check-file-size.sh); see that file's own section comments. *)
module Slice4 = Ops4_split.Slice4
module Select4 = Ops4_split.Select4
module Select_scatter4 = Ops4_split.Select_scatter4
module Concat4 = Ops4_split.Concat4
module Stack4 = Ops4_split.Stack4
module Unbind = Ops4_split.Unbind
module Split_with_sizes4 = Ops4_split.Split_with_sizes4

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
