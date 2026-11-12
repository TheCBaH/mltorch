(* The reduction family of [Ops4] payloads, split out of ops4.ml under the
   tracked file-size ceiling (scripts/check-file-size.sh). [Ops4] re-exports
   every module below by the same name, so external references such as
   [Ops4.Mean_keepdims] or [Ops4.Softmax4] are unaffected. *)

(* Keep-dimensions only: The `.ai/` native4d design §1 says reduction is represented
   ONLY in keep-dimensions form, so there is no [keepdim] field to set wrong. A
   Native [Mean keepdim=false] legalizes to this plus a [Reshape4], and only
   when the packed shape stays four-axis (correction C1). *)
module Mean_keepdims = struct
  type params = { dims : Axis4.t list; keepdim : bool }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"mean_keepdims_params" (fun dims keepdim ->
        { dims; keepdim })
    |> Jsont.Object.mem "dims" (Jsont.list Axis4.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.mem "keepdim" Jsont.bool ~enc:(fun p -> p.keepdim)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a;@ keepdim=%b}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis4.pp))
      p.dims p.keepdim

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
   representation and the same reason: The `.ai/` native4d design §1's reduction
   rule is not specific to the mean. A Native [Amax keepdim=false] legalizes to
   this plus a [Reshape4], exactly as [Mean keepdim=false] does (correction
   C1). *)
module Max_keepdims = struct
  type params = { dims : Axis4.t list; keepdim : bool }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"max_keepdims_params" (fun dims keepdim ->
        { dims; keepdim })
    |> Jsont.Object.mem "dims" (Jsont.list Axis4.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.mem "keepdim" Jsont.bool ~enc:(fun p -> p.keepdim)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a;@ keepdim=%b}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis4.pp))
      p.dims p.keepdim

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
   representation and the same reason: The `.ai/` native4d design §1's reduction
   rule is not specific to the mean. A Native [Sum keepdim=false] legalizes to
   this plus a [Reshape4], exactly as [Mean keepdim=false] does (correction
   C1). *)
module Sum_keepdims = struct
  type params = { dims : Axis4.t list; keepdim : bool }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"sum_keepdims_params" (fun dims keepdim ->
        { dims; keepdim })
    |> Jsont.Object.mem "dims" (Jsont.list Axis4.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.mem "keepdim" Jsont.bool ~enc:(fun p -> p.keepdim)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a;@ keepdim=%b}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis4.pp))
      p.dims p.keepdim

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
  type params = { dims : Axis4.t list; keepdim : bool }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"vector_norm_keepdims_params" (fun dims keepdim ->
        { dims; keepdim })
    |> Jsont.Object.mem "dims" (Jsont.list Axis4.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.mem "keepdim" Jsont.bool ~enc:(fun p -> p.keepdim)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a;@ keepdim=%b}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis4.pp))
      p.dims p.keepdim

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

(* [Reduce.Softmax]'s own payload, narrowed the same way [RepeatInterleave4]'s
   is: ONE named axis, so the field is [Axis4.t] and T/D are unnameable.
   Unlike the four keep-dimensions ops above, softmax never drops rank at
   all -- [Reduce.Softmax.output_shape] is the identity on the input shape --
   so there is no repack for the shape arm to defer to beyond carrying [x]'s
   shape straight through. Gets the same [check_dims]-style axis-domain
   rejection [Select4]/[Slice4]/[RepeatInterleave4] get in
   [Domain.check_node]. See the `.ai/` matmul softmax design §3. *)
module Softmax4 = struct
  type params = { axis : Axis4.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"softmax4_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis4.pp p.axis

  type t = { params : params; x : Tensor_ref.t }

  let name = "Softmax4"

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
    Fmt.pf fmt "@[<hv 2>softmax4@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params
end
