(* Root-mean-square normalisation (ATen `aten.rms_norm`) -- split out of the
   [Norm] category file ([norm.ml]) under the tracked file-size ceiling.
   Shares [Target] and the NORMALIZED-axes layout/count helpers with
   [Norm_layer]/[Norm_group] via [Norm_shared]. *)

open Norm_shared

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

  (* Test-only scalar oracle retained for differential coverage of the
     authoritative [Computation] Region program below. *)
  module Legacy_pixel (S : Semantics.SEMANTICS) = struct
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

  (* Authoritative declarative Region computation.  Inputs have already been
     resolved by role at the graph boundary. *)
  module Computation = struct
    open Region_context

    let program ~limits params ~x ~weight =
      match partition params.dims with
      | Error error -> Error error
      | Ok partition ->
          let sumsq =
            reduce_dims ~kind:Expr.Reduction.Sum ~dims:params.dims
              ~shape:x.Tensor_sig.shape ~leaf:(fun overrides ->
                let value = load x (reduced_coord overrides) in
                Expr.Value.mul value value)
          in
          let inverse sumsq =
            Expr.Value.div (Expr.Value.const 1.)
              (Expr.Value.sqrt
                 (Expr.Value.add
                    (Expr.Value.div sumsq
                       (Expr.Value.const
                          (float_of_int
                             (normalized_count_unchecked
                                ~x_shape:x.Tensor_sig.shape ~dims:params.dims))))
                    (Expr.Value.const params.eps)))
          in
          let weight = load weight (affine_coord params.dims) in
          program
            (Region_program.Builder.run
               (Region_program.Builder.scalar sumsq (fun sumsq ->
                    Region_program.Builder.scalar (inverse sumsq)
                      (fun inverse ->
                        Region_program.Builder.finish
                          ~max_size:limits.Kernel.Limits.max_size
                          ~max_depth:limits.Kernel.Limits.max_depth ~partition
                          ~output:
                            (Expr.Value.mul
                               (Expr.Value.mul (load_output x) inverse)
                               weight)))))
  end
end
