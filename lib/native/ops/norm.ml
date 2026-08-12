(* Normalisations over a set of axes (NHWC). [BatchNorm] (inference) and
   [RmsNorm] (ATen's `aten.rms_norm`); a category file so a future `LayerNorm`
   lands beside them, the way [Reduce] holds the reductions. *)

module BatchNorm = struct
  (* Inference batch normalisation (ATen
     `aten._native_batch_norm_legit_no_training`). Per-channel affine using the
     tracked running statistics — no batch reduction, since eval mode reads the
     running stats directly:

       y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]

     where [c] is the output pixel's index on the [channel] axis; weight, bias,
     running_mean and running_var are the per-channel [C] vectors, read at that
     channel (size 1 / index 0 on every other axis). weight and bias are optional
     (ATen `Tensor?`): an absent weight is the identity scale (1), an absent bias
     the identity shift (0), materialised by [Eval_op]. The output keeps the
     input's full shape (batch norm rescales, it does not reduce). See
     .ai/native_compute_design.md. *)
  type params = { channel : Axis.t; eps : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"batch_norm_params" (fun channel eps ->
        { channel; eps })
    |> Jsont.Object.mem "channel" Axis.jsont ~enc:(fun p -> p.channel)
    |> Jsont.Object.mem "eps" Json_util.f32_jsont ~enc:(fun p -> p.eps)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{channel=%a;@ eps=%a}@]" Axis.pp p.channel Fmt.float p.eps

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t option;
    bias : Tensor_ref.t option;
    running_mean : Tensor_ref.t;
    running_var : Tensor_ref.t;
  }

  let name = "Batch_norm"

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
          running_mean = get "running_mean" Tensor_ref.jsont;
          running_var = get "running_var" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt k = function None -> [] | Some r -> [ (k, ref_ r) ] in
        Json_util.jobj
          ([ ("params", Json_util.enc params_jsont t.params) ]
          @ opt "weight" t.weight @ opt "bias" t.bias
          @ [
              ("running_mean", ref_ t.running_mean);
              ("running_var", ref_ t.running_var);
              ("x", ref_ t.x);
            ]))
      Jsont.json

  let operands (t : t) =
    (t.x :: Option.to_list t.weight)
    @ Option.to_list t.bias
    @ [ t.running_mean; t.running_var ]

  let map_operands f (t : t) =
    {
      t with
      x = f t.x;
      weight = Option.map f t.weight;
      bias = Option.map f t.bias;
      running_mean = f t.running_mean;
      running_var = f t.running_var;
    }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt
      "@[<hv 2>batch_norm@ x=%a@ weight=%a@ bias=%a@ running_mean=%a@ \
       running_var=%a@ params=%a@]"
      pp_ref t.x
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_ref t.running_mean pp_ref t.running_var pp_params t.params

  (* Output keeps the input shape: batch norm rescales, it does not reduce. *)
  let output_shape ~(x_shape : Vec6.shape) = Err.return x_shape

  (* Walk config: just the input shape and eps; the per-channel [C] vectors are
     derived from the shape's C extent by the walk's [build] (see
     lib/native_op_walk/batch_norm_nwalk.ml), which also keeps running_var
     non-negative so sqrt is real. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; eps : float }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        eps = 1e-5;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape
    let params (c : cfg) : params = { channel = Axis.C; eps = c.eps }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "eps" [ 1e-5; 0.; 1e-3 ] (fun r v -> { r with eps = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a eps=%g}" Walk_core.Shape.pp c.shape c.eps
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~x ~weight ~bias ~running_mean ~running_var
        (out : Semantics.position S.index Vec6.t) =
      (* Each [C] vector is read at the output pixel's channel index, broadcast
         (size 1, index 0) on every other axis. *)
      let zero = Vec6.map (fun _ -> S.index_zero) out in
      let at_channel v = S.load v (Vec6.copy_axis out p.channel zero) in
      let mean = at_channel running_mean in
      let var = at_channel running_var in
      (* 1 / sqrt(running_var + eps) — ATen's rsqrt via the sqrt primitive. *)
      let inv = S.div (S.const 1.) (S.sqrt (S.add var (S.const p.eps))) in
      S.add
        (S.mul (S.mul (S.sub (S.load x out) mean) inv) (at_channel weight))
        (at_channel bias)
  end
end

module RmsNorm = struct
  (* Root-mean-square normalisation (ATen `aten.rms_norm`). Unlike a [Reduce],
     the output keeps the input's full shape: the reduction over [dims] only
     computes the per-(non-normalised-coordinate) scale, which then divides every
     element. For each output pixel,

       y = x / sqrt(mean_{dims}(x^2) + eps) * weight

     where the mean of squares is taken over the normalised axes [dims] at the
     pixel's fixed non-normalised coordinates. [weight] carries the
     normalised_shape (ATen indexes it by the trailing/normalised dims only), so
     it is read at the output coordinate on each [dims] axis and broadcast
     (size-1, index 0) on every other axis. See .ai/native_compute_design.md §2.

     [dims] is a list of frame axes (a dispatcher gets them from the ATen
     normalized_shape via [Aten_shape.axis_of_dim]); one op covers
     `rms_norm(normalized_shape=[C])` and the multi-axis case alike. *)
  type params = { dims : Axis.t list; eps : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"rms_norm_params" (fun dims eps -> { dims; eps })
    |> Jsont.Object.mem "dims" (Jsont.list Axis.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.mem "eps" Json_util.f32_jsont ~enc:(fun p -> p.eps)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a;@ eps=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis.pp))
      p.dims Fmt.float p.eps

  type t = { params : params; x : Tensor_ref.t; weight : Tensor_ref.t option }

  let name = "Rms_norm"

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
        let opt_weight =
          match t.weight with None -> [] | Some r -> [ ("weight", ref_ r) ]
        in
        Json_util.jobj
          ([ ("params", Json_util.enc params_jsont t.params) ]
          @ opt_weight
          @ [ ("x", ref_ t.x) ]))
      Jsont.json

  let operands (t : t) = t.x :: Option.to_list t.weight

  let map_operands f (t : t) =
    { t with x = f t.x; weight = Option.map f t.weight }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>rms_norm@ x=%a@ weight=%a@ params=%a@]" pp_ref t.x
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.weight pp_params t.params

  (* Output keeps the input shape: rms-norm rescales, it does not reduce. *)
  let output_shape ~(x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x ~weight
        (out : Semantics.position S.index Vec6.t) =
      let zero = Vec6.map (fun _ -> S.index_zero) out in
      (* Sum of x^2 over the normalised axes at the fixed non-normalised coords
         [out]: nest one [sum] per reduced axis (full extent), squaring the read
         at the leaf. Mirrors [Reduce.Mean]'s reduction nest. [override]
         (which reduced axis is at which reduction-variable index, so far)
         folds directly onto [out] via [Vec6.set] at the leaf, rather than an
         assoc-list lookup per axis at every leaf visited. *)
      let rec sum_sq dims override =
        match dims with
        | [] ->
            let idx =
              List.fold_left (fun v (a, i) -> Vec6.set v a i) out override
            in
            let v = S.load x idx in
            S.mul v v
        | d :: rest ->
            S.sum ~lo:S.index_zero
              ~hi:(S.index_extent (Vec6.get x_shape d))
              (fun i -> sum_sq rest ((d, i) :: override))
      in
      let count =
        List.fold_left (fun acc d -> acc * (Vec6.get x_shape d :> int)) 1 p.dims
      in
      let ms = S.div (sum_sq p.dims []) (S.const (float_of_int count)) in
      (* 1 / sqrt(mean(x^2) + eps) — ATen's rsqrt, expressed with the sqrt
         primitive. *)
      let inv = S.div (S.const 1.) (S.sqrt (S.add ms (S.const p.eps))) in
      (* weight is indexed by the normalised axes only (size 1 elsewhere):
         fold [p.dims] onto an all-zero base, copying each kept axis's value
         from [out]. *)
      let w =
        S.load weight
          (List.fold_left (fun v a -> Vec6.copy_axis out a v) zero p.dims)
      in
      S.mul (S.mul (S.load x out) inv) w
  end
end
