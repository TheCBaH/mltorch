(* Layer normalisation (ATen `aten.layer_norm` / `aten.native_layer_norm`) --
   split out of the [Norm] category file ([norm.ml]) under the tracked
   file-size ceiling. Shares [Target] and the NORMALIZED-axes layout/count
   helpers with [Norm_rms]/[Norm_group] via [Norm_shared]. *)

open Norm_shared

module LayerNorm = struct
  (* Layer normalisation (ATen `aten.layer_norm` and the decomposed
     `aten.native_layer_norm`, which differ in their argument list and their
     return arity, not in this arithmetic). Like [RmsNorm] the output keeps the
     input's full shape; unlike it, the data is CENTRED first, so there are two
     reduction passes rather than one:

       mean = sum_{dims}(x) / count
       var  = sum_{dims}((x - mean)^2) / count
       y    = (x - mean) / sqrt(var + eps)
       y    = y * weight + bias

     all at the pixel's fixed non-normalised coordinates. Four details are
     pinned deliberately, because each is a place a plausible implementation
     differs from ATen:

     - the variance divisor is [count], not [count - 1] (ATen's layer_norm is
       the biased estimator);
     - [eps] is added INSIDE the sqrt;
     - [weight] and [bias] carry the normalized_shape, so they are read at the
       output coordinate on each [dims] axis and broadcast (size 1, index 0) on
       every other -- never at the full output coordinate;
     - the affine step is [* weight] then [+ bias], in that order.

     Two things this deliberately does NOT do. It does not compute the variance
     as [E[x^2] - mean^2]: that is one pass instead of two, and its cancellation
     behaviour on data with a large mean and a small variance is materially
     worse. And it does not route through [RmsNorm]: the formulas differ by the
     centring, which is exactly the mutation the tests have to be able to see,
     so sharing the reduction would hide it.

     [count = 1] is a legal configuration and falls out correctly: [var] is 0,
     so [y] is 0 before the affine, and [bias] alone survives.

     [dims] is a list of frame axes, which both importers derive from ATen's
     trailing [normalized_shape]. See .ai/native_compute_design.md 2. *)
  type params = { dims : Axis.t list; eps : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"layer_norm_params" (fun dims eps -> { dims; eps })
    |> Jsont.Object.mem "dims" (Jsont.list Axis.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.mem "eps" Json_util.f32_jsont ~enc:(fun p -> p.eps)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a;@ eps=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis.pp))
      p.dims Fmt.float p.eps

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t option;
    bias : Tensor_ref.t option;
  }

  let name = "Layer_norm"

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
    Fmt.pf fmt "@[<hv 2>layer_norm@ x=%a@ weight=%a@ bias=%a@ params=%a@]"
      pp_ref t.x
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  (* Output keeps the input shape: layer norm rescales, it does not reduce. The
     reduction's divisor is bounded here for the reason [RmsNorm.output_shape]
     states -- this is the one place all three graph constructors meet. *)
  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let+ (_ : int64) = normalized_count ~x_shape ~dims:p.dims in
    x_shape

  (* Both affine operands carry the normalized_shape, so both have the layout
     [normalized_shape] describes. Checked here rather than trusted: an operand
     of the wrong extent builds a graph and then raises out of [Tensor.read]'s
     strict bounds check partway through the result, which is the bug
     [RmsNorm.check_weight] exists to prevent. Exposed so [Graph_shape4] can run
     the same check on a JSON-decoded Native4D graph, which is a path Native's
     own [Graph_shape] does not cover. *)
  let check_affine ~(x_shape : Vec6.shape) ~(dims : Axis.t list)
      ~(weight : Vec6.shape option) ~(bias : Vec6.shape option) :
      (unit, Shape_error.t) Err.t =
    let expected = normalized_shape ~x_shape ~dims in
    let check operand = function
      | None -> Err.return ()
      | Some actual ->
          if
            List.for_all
              (fun a -> Dim.equal (Vec6.get expected a) (Vec6.get actual a))
              Axis.all
          then Err.return ()
          else
            Err.fail
              (`Operand_shape
                 Shape_error.Operand_shape.{ operand; expected; actual })
    in
    let open Err.Syntax in
    let* () = check `Layer_norm_weight weight in
    check `Layer_norm_bias bias

  (* [k] (how many trailing axes are normalised) and the two affine states are
     walk axes rather than fixed choices, for [RmsNorm.Walk]'s reasons and one
     more: layer_norm's weight and bias are INDEPENDENTLY optional, so all four
     combinations are structurally different graphs.

     [k] stops at 3 and the shape axis keeps the extents small on purpose. The
     symbolic program nests one reduction per normalised axis and does it twice,
     so both the expression size and the staging cost grow with [k]; the
     ceilings are [Kernel.Limits.Hard.depth]/[eval_depth]. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = {
      shape : Walk_core.Shape.t;
      k : int;
      eps : float;
      weight : bool;
      bias : bool;
    }

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 3; c = 5 };
        k = 1;
        eps = 1e-5;
        weight = true;
        bias = true;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape
    let weight_present (c : cfg) = c.weight
    let bias_present (c : cfg) = c.bias
    let dims (c : cfg) = List.filteri (fun i _ -> i >= 6 - c.k) Axis.all
    let params (c : cfg) : params = { dims = dims c; eps = c.eps }

    let affine_shape (c : cfg) =
      normalized_shape ~x_shape:(shape c) ~dims:(dims c)

    (* Both epsilons the corpus contains, plus 0 -- which is the interesting
       one, since it is where a variance of 0 (a constant normalised slice)
       divides by zero rather than by eps. *)
    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "k" [ 1; 2; 3 ] (fun c v -> { c with k = v });
          field_axis "eps" [ 1e-5; 1e-6; 0. ] (fun c v -> { c with eps = v });
          field_axis "weight" [ true; false ] (fun c v -> { c with weight = v });
          field_axis "bias" [ true; false ] (fun c v -> { c with bias = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a k=%d eps=%g weight=%b bias=%b}"
        Walk_core.Shape.pp c.shape c.k c.eps c.weight c.bias
  end

  (* Test-only scalar oracle retained for differential coverage of the
     authoritative [Computation] Region program below. *)
  module Legacy_pixel (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x ~weight ~bias
        (out : Semantics.position S.index Vec6.t) =
      let zero = Vec6.map (fun _ -> S.index_zero) out in
      (* One reduction nest, parameterised by what the leaf does with the value
         it read -- the two passes differ only there. Nest one [sum] per
         normalised axis over its full extent; [override] (which reduced axis is
         at which reduction variable, so far) folds onto [out] via [Vec6.set] at
         the leaf, rather than an assoc lookup per axis at every leaf visited.
         Mirrors [RmsNorm.Legacy_pixel]'s [sum_sq]. *)
      let sum_over leaf =
        let rec go dims override =
          match dims with
          | [] ->
              let idx =
                List.fold_left (fun v (a, i) -> Vec6.set v a i) out override
              in
              leaf (S.load x idx)
          | d :: rest ->
              S.sum ~lo:S.index_zero
                ~hi:(S.index_extent (Vec6.get x_shape d))
                (fun i -> go rest ((d, i) :: override))
        in
        go p.dims []
      in
      let count =
        S.const
          (float_of_int (normalized_count_unchecked ~x_shape ~dims:p.dims))
      in
      let mean = S.div (sum_over (fun v -> v)) count in
      (* The BIASED variance of the CENTRED data. Two passes, not
         [E[x^2] - mean^2]. *)
      let var =
        S.div
          (sum_over (fun v ->
               let d = S.sub v mean in
               S.mul d d))
          count
      in
      (* eps INSIDE the sqrt, then 1/sqrt -- ATen's rsqrt, expressed with the
         sqrt primitive as [BatchNorm] and [RmsNorm] do. *)
      let inv = S.div (S.const 1.) (S.sqrt (S.add var (S.const p.eps))) in
      (* Read at the normalised axes only (size 1, index 0 elsewhere): fold
         [p.dims] onto an all-zero base, copying each normalised axis's value
         from [out]. Indexing these by the full output coordinate is a wrong
         answer that still typechecks. *)
      let at_dims t =
        S.load t
          (List.fold_left (fun v a -> Vec6.copy_axis out a v) zero p.dims)
      in
      S.add
        (S.mul (S.mul (S.sub (S.load x out) mean) inv) (at_dims weight))
        (at_dims bias)
  end

  (* Authoritative declarative Region computation.  Inputs have already been
     resolved by role at the graph boundary. *)
  module Computation = struct
    open Region_context

    let program ~limits params ~x ~weight ~bias =
      match partition params.dims with
      | Error error -> Error error
      | Ok partition ->
          let sum =
            reduce_dims ~kind:Expr.Reduction.Sum ~dims:params.dims
              ~shape:x.Tensor_sig.shape ~leaf:(fun overrides ->
                load x (reduced_coord overrides))
          in
          let count =
            Expr.Value.const
              (float_of_int
                 (normalized_count_unchecked ~x_shape:x.Tensor_sig.shape
                    ~dims:params.dims))
          in
          let affine value = load value (affine_coord params.dims) in
          program
            (Region_program.Builder.run
               (Region_program.Builder.scalar sum (fun sum ->
                    Region_program.Builder.scalar (Expr.Value.div sum count)
                      (fun mean ->
                        let variance_sum =
                          reduce_dims ~kind:Expr.Reduction.Sum ~dims:params.dims
                            ~shape:x.Tensor_sig.shape ~leaf:(fun overrides ->
                              let delta =
                                Expr.Value.sub
                                  (load x (reduced_coord overrides))
                                  mean
                              in
                              Expr.Value.mul delta delta)
                        in
                        Region_program.Builder.scalar variance_sum
                          (fun variance_sum ->
                            let inverse =
                              Expr.Value.div (Expr.Value.const 1.)
                                (Expr.Value.sqrt
                                   (Expr.Value.add
                                      (Expr.Value.div variance_sum count)
                                      (Expr.Value.const params.eps)))
                            in
                            Region_program.Builder.scalar inverse
                              (fun inverse ->
                                let bias =
                                  load bias (affine_coord params.dims)
                                in
                                Region_program.Builder.finish
                                  ~max_size:limits.Kernel.Limits.max_size
                                  ~max_depth:limits.Kernel.Limits.max_depth
                                  ~partition
                                  ~output:
                                    (Expr.Value.add
                                       (Expr.Value.mul
                                          (Expr.Value.mul
                                             (Expr.Value.sub (load_output x)
                                                mean)
                                             inverse)
                                          (affine weight))
                                       bias)))))))
  end
end
