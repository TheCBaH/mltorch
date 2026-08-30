(* [BatchNorm] (inference, ATen `_native_batch_norm_legit_no_training`) and
   [BatchNormNoStats] (training without running statistics, ATen
   `_native_batch_norm_legit.no_stats`) -- split out of the [Norm] category
   file ([norm.ml]) under the tracked file-size ceiling. *)

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

module BatchNormNoStats = struct
  (* Training batch normalisation without running statistics (ATen
     [_native_batch_norm_legit.no_stats]).  Unlike [BatchNorm], the mean and
     variance are reductions of THIS input, and ATen exposes the mean and
     reciprocal standard deviation as real outputs:

       mean[c] = mean_{all axes except C}(x[..., c])
       invstd[c] = 1 / sqrt(mean((x[..., c] - mean[c])^2) + eps)
       y = (x - mean[c]) * invstd[c] * weight[c] + bias[c]

     The variance is biased (division by [count], not [count - 1]).  This must
     remain separate from inference [BatchNorm]: replacing it by a read of
     running statistics is a plausible, silently wrong graph. *)
  type params = { channel : Axis.t; eps : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"batch_norm_no_stats_params" (fun channel eps ->
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
  }

  let name = "Batch_norm_no_stats"

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
          ([ ("params", Json_util.enc params_jsont t.params) ]
          @ opt "weight" t.weight @ opt "bias" t.bias
          @ [ ("x", ref_ t.x) ]))
      Jsont.json

  let operands (t : t) = t.x :: (Option.to_list t.weight @ Option.to_list t.bias)

  let map_operands f (t : t) =
    {
      t with
      x = f t.x;
      weight = Option.map f t.weight;
      bias = Option.map f t.bias;
    }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt
      "@[<hv 2>batch_norm_no_stats@ x=%a@ weight=%a@ bias=%a@ params=%a@]"
      pp_ref t.x
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  let reduced_axes (p : params) =
    List.filter (fun a -> not (Axis.equal a p.channel)) Axis.all

  let stats_shape ~(x_shape : Vec6.shape) (p : params) =
    Vec6.set
      (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
      p.channel
      (Vec6.get x_shape p.channel)

  let reduction_count ~(x_shape : Vec6.shape) (p : params) =
    Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel
      (List.fold_left
         (fun s a -> Vec6.set s a (Vec6.get x_shape a))
         (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
         (reduced_axes p))

  let reduction_count_unchecked ~(x_shape : Vec6.shape) (p : params) =
    List.fold_left
      (fun n a -> n * (Vec6.get x_shape a :> int))
      1 (reduced_axes p)

  let output_shapes ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let+ (_ : int64) = reduction_count ~x_shape p in
    let stats = stats_shape ~x_shape p in
    [ x_shape; stats; stats ]

  let check_affine ~(x_shape : Vec6.shape) (p : params) ~(actual : Vec6.shape) =
    let expected = stats_shape ~x_shape p in
    if
      List.for_all
        (fun a -> Dim.equal (Vec6.get expected a) (Vec6.get actual a))
        Axis.all
    then Err.return ()
    else
      Err.fail
        (`Operand_shape
           Shape_error.Operand_shape.
             { operand = `Batch_norm_no_stats_affine; expected; actual })

  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = {
      shape : Walk_core.Shape.t;
      eps : float;
      weight : bool;
      bias : bool;
    }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 3; w = 2; c = 4 };
        eps = 1e-5;
        weight = true;
        bias = true;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape
    let params (c : cfg) = { channel = Axis.C; eps = c.eps }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c shape -> { c with shape });
          field_axis "eps" [ 1e-5; 1e-8 ] (fun c eps -> { c with eps });
          field_axis "weight" [ true; false ] (fun c weight ->
              { c with weight });
          field_axis "bias" [ true; false ] (fun c bias -> { c with bias });
        ]

    let pp fmt (c : cfg) =
      Fmt.pf fmt "{shape=%a eps=%g weight=%b bias=%b}" Walk_core.Shape.pp
        c.shape c.eps c.weight c.bias
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    let statistics (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let rec sum_over leaf axes override =
        match axes with
        | [] ->
            let idx =
              List.fold_left (fun v (a, i) -> Vec6.set v a i) out override
            in
            leaf (S.load x idx)
        | a :: rest ->
            S.sum ~lo:S.index_zero
              ~hi:(S.index_extent (Vec6.get x_shape a))
              (fun i -> sum_over leaf rest ((a, i) :: override))
      in
      let count =
        S.const (float_of_int (reduction_count_unchecked ~x_shape p))
      in
      let axes = reduced_axes p in
      let mean = S.div (sum_over (fun v -> v) axes []) count in
      let var =
        S.div
          (sum_over
             (fun v ->
               let d = S.sub v mean in
               S.mul d d)
             axes [])
          count
      in
      let invstd = S.div (S.const 1.) (S.sqrt (S.add var (S.const p.eps))) in
      (mean, invstd)

    let pixel ~output (p : params) ~(x_shape : Vec6.shape) ~x ~weight ~bias out
        =
      let mean, invstd = statistics p ~x_shape ~x out in
      match output with
      | 0 ->
          let zero = Vec6.map (fun _ -> S.index_zero) out in
          let at_channel v = S.load v (Vec6.copy_axis out p.channel zero) in
          S.add
            (S.mul
               (S.mul (S.sub (S.load x out) mean) invstd)
               (at_channel weight))
            (at_channel bias)
      | 1 -> mean
      | 2 -> invstd
      | _ -> invalid_arg "BatchNormNoStats.Compute.pixel: output ordinal"
  end
end
