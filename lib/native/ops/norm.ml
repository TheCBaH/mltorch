(* Normalisations over a set of axes (NHWC). [BatchNorm] (inference),
   [RmsNorm] (ATen's `aten.rms_norm`) and [LayerNorm] (ATen's
   `aten.layer_norm` / `aten.native_layer_norm`); a category file, the way
   [Reduce] holds the reductions. *)

(* Which normalisation a shared diagnostic came from.

   Both importers report the same two [normalized_shape] faults for three ATen
   targets, and until Group 7 the message said "rms_norm" unconditionally --
   correct while rms_norm was the only caller and wrong the moment layer_norm
   reused [Op_bridge.normalized_dims]. A closed set, so a variant rather than
   the [string] that [Invalid_dim]/[Dims_count] carry (CLAUDE.md: a closed
   value set is a variant). Here rather than in either importer because both
   need it and two copies is one drift away from two vocabularies. *)
module Target = struct
  type t = Rms_norm | Layer_norm | Native_layer_norm

  let pp fmt = function
    | Rms_norm -> Fmt.string fmt "rms_norm"
    | Layer_norm -> Fmt.string fmt "layer_norm"
    | Native_layer_norm -> Fmt.string fmt "native_layer_norm"
end

(* The layout ATen gives an operand indexed by the NORMALIZED axes only: the
   input's extent on each of [dims], extent 1 (broadcast) everywhere else.
   rms_norm's weight and layer_norm's weight and bias all have it, and one
   definition is what keeps shape inference, the operand check and the walk's
   operand synthesis from disagreeing about it. *)
let normalized_shape ~(x_shape : Vec6.shape) ~(dims : Axis.t list) =
  List.fold_left
    (fun acc a -> Vec6.set acc a (Vec6.get x_shape a))
    (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
    dims

(* How many elements one normalisation reduces over -- the divisor of the mean
   and of the mean of squares.

   BOUNDED, and that is not decoration. [lib/native] is js_of_ocaml-reachable,
   where [int] is 32 bits, and this is a product of up to six extents each
   bounded only by [Kernel.Limits.Hard.extent] (2^31): the mathematical product
   reaches 2^186, so folding it into an [int] and checking afterwards is a
   post-overflow comparison, not a bound (CLAUDE.md's aggregate rule).
   [Vec6.numel_bounded] divides each factor into the ceiling BEFORE multiplying,
   so no intermediate wraps.

   Called from [output_shape], which is where [Graph_builder] and a JSON-decoded
   graph both meet it -- a rule only the importers enforced would be a rule the
   other two constructors do not have. *)
let normalized_count ~(x_shape : Vec6.shape) ~(dims : Axis.t list) =
  Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel
    (normalized_shape ~x_shape ~dims)

(* The same product as an [int], for [Compute], which has no error channel.

   Sound as a plain fold ONLY because [output_shape] has already run
   [normalized_count] on this node: its result is below [Kernel.Limits.Hard.numel]
   = 2^31, which is exactly [max_int] on the 32-bit backend, so this product
   cannot wrap there. A node whose shape rule never ran cannot be evaluated, so
   the precondition holds by construction -- but it IS a precondition, and this
   function must not be called anywhere the bound has not been established. *)
let normalized_count_unchecked ~(x_shape : Vec6.shape) ~(dims : Axis.t list) =
  List.fold_left (fun acc d -> acc * (Vec6.get x_shape d :> int)) 1 dims

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

  (* float32's machine epsilon, which is what ATen's `rms_norm` uses when [eps]
     is absent. Here rather than in either importer: both need it, and two
     copies of a literal is one drift away from two different default ops. *)
  let default_eps = 1.1920929e-07

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

  (* Output keeps the input shape: rms-norm rescales, it does not reduce. It is
     still not [Err.return x_shape], which is what this was: the reduction's
     divisor is a product of extents, and bounding it is this function's job
     because it is the one place [Graph_builder] and a JSON-decoded graph both
     reach. See [normalized_count] -- the unbounded fold that used to live in
     [Compute] wrapped on the 32-bit backend. *)
  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let+ (_ : int64) = normalized_count ~x_shape ~dims:p.dims in
    x_shape

  (* ATen indexes the weight by the NORMALIZED dims only, so it carries the
     input's extent on each of those axes and is extent-1 (broadcast) on every
     other. One definition of that layout, used by shape inference to reject a
     weight that disagrees -- which nothing checked, so a wrong one built a
     graph and then raised from [Tensor.read] partway through the result.
     Shared with [LayerNorm]'s affine operands, which have the same layout. *)
  let weight_shape = normalized_shape

  let check_weight ~(x_shape : Vec6.shape) ~(dims : Axis.t list)
      ~(actual : Vec6.shape) : (unit, Shape_error.t) Err.t =
    let expected = weight_shape ~x_shape ~dims in
    if
      List.for_all
        (fun a -> Dim.equal (Vec6.get expected a) (Vec6.get actual a))
        Axis.all
    then Err.return ()
    else
      Err.fail
        (`Operand_shape
           Shape_error.Operand_shape.
             { operand = `Rms_norm_weight; expected; actual })

  (* The normalized axes are the innermost [k] of the frame, which is what both
     importers derive from ATen's trailing [normalized_shape]. [k] is an axis of
     the walk rather than a fixed choice, because the single- and multi-axis
     cases divide by different counts and read the weight on different axes --
     and both preserve the output shape, so a walk that only ever tried one
     would look like coverage and be none.

     [weight] is an axis too: [Graph_ir]'s [Rms_norm] carries it as an option
     and the absent case is a different code path, not a ones tensor. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = {
      shape : Walk_core.Shape.t;
      k : int;
      eps : float;
      weight : bool;
    }

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 3; c = 5 };
        k = 1;
        eps = 1e-5;
        weight = true;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape
    let weight_present (c : cfg) = c.weight
    let dims (c : cfg) = List.filteri (fun i _ -> i >= 6 - c.k) Axis.all
    let params (c : cfg) : params = { dims = dims c; eps = c.eps }

    (* ATen indexes the weight by the normalized dims only, so it is extent-1
       everywhere else and broadcasts. *)
    let weight_shape (c : cfg) =
      let x = shape c in
      List.fold_left
        (fun acc a -> Vec6.set acc a (Vec6.get x a))
        (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
        (dims c)

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "k" [ 1; 2; 3 ] (fun c v -> { c with k = v });
          field_axis "eps" [ 1e-5; 0.; 1e-3 ] (fun c v -> { c with eps = v });
          field_axis "weight" [ true; false ] (fun c v -> { c with weight = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a k=%d eps=%g weight=%b}" Walk_core.Shape.pp
        c.shape c.k c.eps c.weight
  end

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
      let count = normalized_count_unchecked ~x_shape ~dims:p.dims in
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

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x ~weight ~bias
        (out : Semantics.position S.index Vec6.t) =
      let zero = Vec6.map (fun _ -> S.index_zero) out in
      (* One reduction nest, parameterised by what the leaf does with the value
         it read -- the two passes differ only there. Nest one [sum] per
         normalised axis over its full extent; [override] (which reduced axis is
         at which reduction variable, so far) folds onto [out] via [Vec6.set] at
         the leaf, rather than an assoc lookup per axis at every leaf visited.
         Mirrors [RmsNorm.Compute]'s [sum_sq]. *)
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
end

module GroupNorm = struct
  (* Group normalisation (ATen `aten.group_norm`). Reshapes [channel]'s extent
     into [groups] equal-sized chunks and normalises each (N, group) slice
     over that chunk together with every axis but [N] and [channel] -- T, D,
     H, W in the always-six-axis frame, not a caller-chosen [dims] list the
     way [LayerNorm]/[RmsNorm] take one, since ATen's group_norm always
     reduces over "everything except batch and (the group's slice of)
     channel". Unlike them, [weight]/[bias] are full PER-CHANNEL vectors
     (ATen's group_norm affine has shape [C], not the normalized-shape
     [RmsNorm]/[LayerNorm] carry) -- read at the output pixel's own channel
     index and broadcast elsewhere, the same [BatchNorm.Compute] pattern.

       count = channels_per_group * (product of every non-N/channel extent)
       group = floor(channel_index / channels_per_group)
       mean  = sum over that group's window / count
       var   = sum over (x - mean)^2 in that window / count
       y     = (x - mean) / sqrt(var + eps) * weight[channel] + bias[channel]

     [channel] is a parameter (mirroring [BatchNorm.params.channel]) rather
     than hardcoded to [Axis.C], though every importer sets it there today. *)
  type params = { channel : Axis.t; groups : Op_config.Pos.t; eps : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"group_norm_params" (fun channel groups eps ->
        { channel; groups = Op_config.Pos.of_int groups; eps })
    |> Jsont.Object.mem "channel" Axis.jsont ~enc:(fun p -> p.channel)
    |> Jsont.Object.mem "groups" Jsont.int ~enc:(fun p -> (p.groups :> int))
    |> Jsont.Object.mem "eps" Json_util.f32_jsont ~enc:(fun p -> p.eps)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{channel=%a;@ groups=%d;@ eps=%a}@]" Axis.pp p.channel
      (p.groups :> int)
      Fmt.float p.eps

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t option;
    bias : Tensor_ref.t option;
  }

  let name = "Group_norm"

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
    Fmt.pf fmt "@[<hv 2>group_norm@ x=%a@ weight=%a@ bias=%a@ params=%a@]"
      pp_ref t.x
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  (* Every axis except N and [channel]: what a group reduces over besides its
     own channel slice. Not a caller [dims] list -- ATen's group_norm has no
     axis-selection parameter, so this is fixed by the op's own semantics. *)
  let spatial_dims (channel : Axis.t) =
    List.filter
      (fun a -> (not (Axis.equal a Axis.N)) && not (Axis.equal a channel))
      Axis.all

  (* The shape one group reduces over: every [spatial_dims] axis at its full
     extent, [channel] narrowed to one group's slice, every other axis extent
     1. Used only to BOUND the reduction count -- [Vec6.numel_bounded] divides
     each factor into the ceiling before multiplying, the aggregate rule
     CLAUDE.md states and [normalized_count] above already follows. *)
  let reduce_shape ~(x_shape : Vec6.shape) ~(channel : Axis.t)
      ~(channels_per_group : Dim.extent Dim.t) =
    let base =
      List.fold_left
        (fun acc a -> Vec6.set acc a (Vec6.get x_shape a))
        (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
        (spatial_dims channel)
    in
    Vec6.set base channel channels_per_group

  (* Output keeps the input shape: group norm rescales, it does not reduce.
     [num_groups] must divide the channel count exactly -- ATen's own
     contract, checked here since it arrives as an ordinary int from a PT2
     graph rather than a value ATen has already validated -- and the
     reduction count is bounded the same way [RmsNorm]/[LayerNorm]'s are. *)
  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let channels = Vec6.get x_shape p.channel in
    let groups = (p.groups :> int) in
    if (channels :> int) mod groups <> 0 then
      Err.fail
        (`Group_norm Shape_error.Group_norm.{ channels; groups = p.groups })
    else
      let cpg = Dim.extent ((channels :> int) / groups) in
      let+ (_ : int64) =
        Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel
          (reduce_shape ~x_shape ~channel:p.channel ~channels_per_group:cpg)
      in
      x_shape

  (* The same product as an [int], for [Compute], which has no error channel.
     Sound as a plain fold ONLY because [output_shape] has already run
     [Vec6.numel_bounded] on the identical product ([reduce_shape]'s numel) --
     the same precondition [normalized_count_unchecked] documents above, one
     level up: a node whose shape rule never ran cannot be evaluated. *)
  let count_unchecked ~(x_shape : Vec6.shape) ~(channel : Axis.t) ~cpg =
    List.fold_left
      (fun acc a -> acc * (Vec6.get x_shape a :> int))
      cpg (spatial_dims channel)

  (* Both affine operands are full PER-CHANNEL vectors -- [normalized_shape]
     with a single-axis [dims] happens to be exactly that layout ([channel] at
     its full extent, every other axis 1), so this reuses it rather than
     restating it; checked for the reason [LayerNorm.check_affine]'s doc
     comment gives -- an operand of the wrong extent otherwise builds a graph
     and raises out of [Tensor.read]'s bounds check partway through the
     result. *)
  let check_affine ~(x_shape : Vec6.shape) ~(channel : Axis.t)
      ~(weight : Vec6.shape option) ~(bias : Vec6.shape option) :
      (unit, Shape_error.t) Err.t =
    let expected = normalized_shape ~x_shape ~dims:[ channel ] in
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
    let* () = check `Group_norm_weight weight in
    check `Group_norm_bias bias

  (* Walk config: [groups] must divide the channel extent, so it is derived
     from the drawn shape's C via [divisor] rather than an independent axis --
     the same "derive from the drawn extent" shape [Slice.Walk]'s [bounds] and
     [Split_with_sizes.Walk]'s [sizes_for] use, and for the same reason: an
     independent [groups] axis would draw far more configurations
     [output_shape] refuses (indivisible) than ones it accepts. The three
     candidates span group_norm's range: 1 (whole-channel), 2, and C itself
     (per-channel groups -- group_norm's instance-norm corner). *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = {
      shape : Walk_core.Shape.t;
      divisor : int;
      eps : float;
      weight : bool;
      bias : bool;
    }

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 3; w = 3; c = 6 };
        divisor = 1;
        eps = 1e-5;
        weight = true;
        bias = true;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let groups (c : cfg) =
      let c_extent = (Vec6.get (shape c) Axis.C :> int) in
      let candidates =
        List.filter (fun g -> c_extent mod g = 0) [ 1; 2; c_extent ]
      in
      List.nth candidates (c.divisor mod List.length candidates)

    let weight_present (c : cfg) = c.weight
    let bias_present (c : cfg) = c.bias

    let params (c : cfg) : params =
      {
        channel = Axis.C;
        groups = Op_config.Pos.of_int (groups c);
        eps = c.eps;
      }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "divisor" [ 0; 1; 2 ] (fun c v -> { c with divisor = v });
          field_axis "eps" [ 1e-5; 0.; 1e-3 ] (fun c v -> { c with eps = v });
          field_axis "weight" [ true; false ] (fun c v -> { c with weight = v });
          field_axis "bias" [ true; false ] (fun c v -> { c with bias = v });
        ]

    let pp fmt (c : cfg) =
      let p = params c in
      Format.fprintf fmt "{shape=%a groups=%d eps=%g weight=%b bias=%b}"
        Walk_core.Shape.pp c.shape
        (p.groups :> int)
        c.eps c.weight c.bias
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x ~weight ~bias
        (out : Semantics.position S.index Vec6.t) =
      let channel = p.channel in
      let c_extent = (Vec6.get x_shape channel :> int) in
      let groups = (p.groups :> int) in
      let cpg = c_extent / groups in
      let cpg_pos = Op_config.Pos.of_int cpg in
      (* Which group [out]'s channel coordinate falls in, and that group's
         window [lo, hi) on [channel] -- [lo] is [group * cpg], always
         non-negative since both factors are, so [clamp_low] is the SOUND
         delta->position conversion the same way every other windowed-offset
         site in this engine uses it. *)
      let group =
        S.index_floor_div_pos (S.of_index (Vec6.get out channel)) cpg_pos
      in
      let lo = S.clamp_low (S.index_scale cpg group) in
      let hi = S.index_add (S.of_index lo) (S.index_const cpg) in
      let zero = Vec6.map (fun _ -> S.index_zero) out in
      (* One reduction nest over [channel]'s WINDOW plus every spatial axis's
         FULL extent -- mirrors [LayerNorm.Compute]'s [sum_over], generalised
         to a non-zero lower bound on the one axis that needs it. *)
      let sum_over leaf =
        let rec go dims override =
          match dims with
          | [] ->
              let idx =
                List.fold_left (fun v (a, i) -> Vec6.set v a i) out override
              in
              leaf (S.load x idx)
          | d :: rest when Axis.equal d channel ->
              S.sum ~lo ~hi (fun i -> go rest ((d, i) :: override))
          | d :: rest ->
              S.sum ~lo:S.index_zero
                ~hi:(S.index_extent (Vec6.get x_shape d))
                (fun i -> go rest ((d, i) :: override))
        in
        go (channel :: spatial_dims channel) []
      in
      let count =
        S.const (float_of_int (count_unchecked ~x_shape ~channel ~cpg))
      in
      let mean = S.div (sum_over (fun v -> v)) count in
      let var =
        S.div
          (sum_over (fun v ->
               let d = S.sub v mean in
               S.mul d d))
          count
      in
      let inv = S.div (S.const 1.) (S.sqrt (S.add var (S.const p.eps))) in
      let at_channel v = S.load v (Vec6.copy_axis out channel zero) in
      S.add
        (S.mul (S.mul (S.sub (S.load x out) mean) inv) (at_channel weight))
        (at_channel bias)
  end
end
