(* Reductions over a set of axes (NHWC): [Mean] (ATen's `aten.mean.dim`),
   [Amax] (`aten.amax.default`), [Vector_norm]
   (`aten.linalg_vector_norm.default`) and [Softmax] (`aten.softmax.int`,
   which unlike the other three keeps the input's shape rather than dropping
   the reduced axis -- see its own comment); a category file so a future
   `aten.sum`/… lands beside them, the way [Pool] holds Max/AvgPool2d. *)

(* Params shared by every reduction that folds a set of named axes down with a
   [keepdim] flag: the axis list, its output-shape rule, and the (input axis,
   output axis) map a [Compute] functor folds its output coordinate through.
   One definition, so [Mean] and [Amax] cannot drift on how [keepdim]/axis
   dropping behaves — see .ai/native_add_op.md's payload-boilerplate reuse
   note. *)
module Dims_keepdim = struct
  type t = { dims : Axis.t list; keepdim : bool }

  let jsont ~kind : t Jsont.t =
    Jsont.Object.map ~kind (fun dims keepdim -> { dims; keepdim })
    |> Jsont.Object.mem "dims" (Jsont.list Axis.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.mem "keepdim" Jsont.bool ~enc:(fun p -> p.keepdim)
    |> Jsont.Object.finish

  (* Transports [dims] through an axis mapping (e.g. a permutation's
     [lookup]): every entry is remapped, in place, order preserved. Used by
     [Sink_permute_mean] to move a reduction past an input permute without
     touching [keepdim]. See .ai/native_transform_design.md §12f. *)
  let map_dims f (p : t) = { p with dims = List.map f p.dims }

  let pp fmt (p : t) =
    Fmt.pf fmt "@[<hv>{dims=%a;@ keepdim=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis.pp))
      p.dims Fmt.bool p.keepdim

  (* Maps each surviving INPUT axis to the OUTPUT axis that carries its data.
     keepdim=true: identity (collapse in place). keepdim=false: the survivors
     (all axes minus the reduced ones, in canonical order) re-pack
     right-aligned into the innermost positions. Depends only on [p], not on
     extents. *)
  let kept_map (p : t) =
    if p.keepdim then
      List.filter_map
        (fun a -> if List.mem a p.dims then None else Some (a, a))
        Axis.all
    else Aten_shape.repack_dropped ~dropped:p.dims

  (* Shape of the reduced axes alone (extent on each of [dims], 1 elsewhere) --
     the layout [Norm.normalized_shape] computes for the affine/normalized
     case, specialised here to a plain axis list. Its numel is the reduction's
     element count. *)
  let reduced_shape ~(x_shape : Vec6.shape) (p : t) =
    List.fold_left
      (fun acc a -> Vec6.set acc a (Vec6.get x_shape a))
      (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
      p.dims

  (* BOUNDED, and that is not decoration: see [Norm.normalized_count]. This is
     a product of up to six extents each bounded only by
     [Kernel.Limits.Hard.extent] (2^31), so folding it into an [int] and
     checking afterwards is a post-overflow comparison on the 32-bit
     js_of_ocaml backend, not a bound (CLAUDE.md's aggregate rule).
     [Vec6.numel_bounded] divides each factor into the ceiling before
     multiplying. Called from [output_shape], the one place [Graph_builder]
     and a JSON-decoded graph both reach. *)
  let reduced_count ~(x_shape : Vec6.shape) (p : t) =
    Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel
      (reduced_shape ~x_shape p)

  (* Same product as an [int], for [Compute], which has no error channel.
     Sound only because [output_shape] already ran [reduced_count] on this
     node -- see [Norm.normalized_count_unchecked]'s identical precondition. *)
  let reduced_count_unchecked ~(x_shape : Vec6.shape) (p : t) =
    List.fold_left (fun acc d -> acc * (Vec6.get x_shape d :> int)) 1 p.dims

  (* Output = surviving input extents packed onto their output axes;
     reduced axes (and, for keepdim=false, the vacated outer axes) are
     extent 1. *)
  let output_shape ~(x_shape : Vec6.shape) (p : t) =
    let open Err.Syntax in
    let+ (_ : int64) = reduced_count ~x_shape p in
    let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
    List.fold_left
      (fun s (kin, oax) -> Vec6.copy x_shape ~src:kin ~dst:oax s)
      ones (kept_map p)
end

module Mean = struct
  (* Arithmetic mean over [dims]: each reduced axis is summed over its full
     extent, divided by their product. resnet18's global average pool is this
     over H/W — [AvgPool2d] with the window spanning the whole axis, but
     expressed directly as a reduction (no kernel/stride/pad/clipping: the full
     axis is always in range). See .ai/native_compute_design.md §2.

     [dims] is a list of frame axes (a dispatcher gets them from an ATen dim list
     via [Aten_shape.axis_of_dim]); one op covers `mean(dim=[H,W])`.

     [keepdim] mirrors `aten.mean.dim`'s flag (.ai/native_tensor_design.md §1d):
       - keepdim=true  → each reduced axis collapses to extent 1 *in place*; the
         named layout is otherwise unchanged.
       - keepdim=false → the reduced axes are removed and the survivors re-pack
         toward the inside, so the named axis carrying a given input's data
         shifts. mean over W of [H6 W7 C8] gives [H1 W6 C8] — H's data now lives
         on W (whereas keepdim=true gives [H6 W1 C8]).
     The op stays a rank-free function of named axes: the keepdim=false shift
     needs no input rank, because under the right-aligned embedding the axes
     outside the data are already size 1, so shifting *every* surviving axis
     (real or size-1 padding) inward by the number of removed axes outer to it
     lands the real data on exactly the right-aligned axes. (Recovering the ATen
     output *rank* is a separate concern — [Aten_shape.to_aten], not the op's.) *)
  type params = Dims_keepdim.t = { dims : Axis.t list; keepdim : bool }

  let params_jsont : params Jsont.t = Dims_keepdim.jsont ~kind:"mean_params"
  let map_dims = Dims_keepdim.map_dims
  let pp_params = Dims_keepdim.pp

  type t = { params : params; x : Tensor_ref.t }

  let name = "Mean"

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
    Fmt.pf fmt "@[<hv 2>mean@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  let kept_map = Dims_keepdim.kept_map

  (* Output = surviving input extents packed onto their output axes (§1d);
     reduced axes (and, for keepdim=false, the vacated outer axes) are extent
     1. Bounds the reduction count as a side effect — see
     [Dims_keepdim.reduced_count]. *)
  let output_shape = Dims_keepdim.output_shape

  (* This op's random-walk config space: a 4D NHWC tensor reduced over a curated
     set of axis subsets with keepdim toggled. Any non-empty subset is valid, so
     [cascade] is identity. Lives with the op (per backend). *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; dims : Axis.t list; keepdim : bool }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        dims = [ Axis.H; Axis.W ];
        keepdim = false;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape
    let params (c : cfg) : params = { dims = c.dims; keepdim = c.keepdim }

    let dim_sets =
      Axis.[ [ H; W ]; [ H; W; C ]; [ C ]; [ H ]; [ W ]; [ N; H; W ] ]

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "dims" dim_sets (fun r v -> { r with dims = v });
          field_axis "keepdim" [ true; false ] (fun r v ->
              { r with keepdim = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a dims=[%a] keepdim=%b}" Walk_core.Shape.pp
        c.shape
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
           Axis.pp)
        c.dims c.keepdim
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let zero =
        Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
          ~h:S.index_zero ~w:S.index_zero ~c:S.index_zero
      in
      (* [kept_map]'s (input axis, output axis) pairs, folded once: [base]'s
         surviving axes already hold [out oax]; a reduced axis (not in
         [kept_map]) stays [index_zero] until [reduce]'s leaf overrides it.
         Folding here — not a [List.assoc_opt] lookup inside the leaf —
         means the lookup happens once per pixel, not once per (potentially
         many) leaf visited by the reduction below. *)
      let base =
        List.fold_left
          (fun v (kin, oax) -> Vec6.copy out ~src:oax ~dst:kin v)
          zero (kept_map p)
      in
      (* Nest one [sum] per reduced axis (full extent [0, extent)); the body
         reads [x] at [base] with the current reduction indices folded on
         top for the axes being summed over. *)
      let rec reduce dims override =
        match dims with
        | [] ->
            S.load x
              (List.fold_left (fun v (a, i) -> Vec6.set v a i) base override)
        | d :: rest ->
            S.sum ~lo:S.index_zero
              ~hi:(S.index_extent (Vec6.get x_shape d))
              (fun i -> reduce rest ((d, i) :: override))
      in
      let total = reduce p.dims [] in
      let count = Dims_keepdim.reduced_count_unchecked ~x_shape p in
      S.div total (S.const (float_of_int count))
  end
end

module Sum = struct
  (* Plain summation over [dims] -- ATen's `aten.sum.dim_IntList`. Same
     axis/keepdim shape rule as [Mean] ([Dims_keepdim]); the only difference
     is [Compute] skips [Mean]'s final division, the same relationship
     [Amax] has to a hypothetical "average-pool-style max". *)
  type params = Dims_keepdim.t = { dims : Axis.t list; keepdim : bool }

  let params_jsont : params Jsont.t = Dims_keepdim.jsont ~kind:"sum_params"
  let map_dims = Dims_keepdim.map_dims
  let pp_params = Dims_keepdim.pp

  type t = { params : params; x : Tensor_ref.t }

  let name = "Sum"

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
    Fmt.pf fmt "@[<hv 2>sum@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  let kept_map = Dims_keepdim.kept_map
  let output_shape = Dims_keepdim.output_shape

  (* Same config space as [Mean]'s walk, minus the divisor concern -- any
     non-empty dim subset is valid, so [cascade] is identity. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; dims : Axis.t list; keepdim : bool }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        dims = [ Axis.H; Axis.W ];
        keepdim = false;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape
    let params (c : cfg) : params = { dims = c.dims; keepdim = c.keepdim }

    let dim_sets =
      Axis.[ [ H; W ]; [ H; W; C ]; [ C ]; [ H ]; [ W ]; [ N; H; W ] ]

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "dims" dim_sets (fun r v -> { r with dims = v });
          field_axis "keepdim" [ true; false ] (fun r v ->
              { r with keepdim = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a dims=[%a] keepdim=%b}" Walk_core.Shape.pp
        c.shape
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
           Axis.pp)
        c.dims c.keepdim
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let zero =
        Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
          ~h:S.index_zero ~w:S.index_zero ~c:S.index_zero
      in
      let base =
        List.fold_left
          (fun v (kin, oax) -> Vec6.copy out ~src:oax ~dst:kin v)
          zero (kept_map p)
      in
      let rec reduce dims override =
        match dims with
        | [] ->
            S.load x
              (List.fold_left (fun v (a, i) -> Vec6.set v a i) base override)
        | d :: rest ->
            S.sum ~lo:S.index_zero
              ~hi:(S.index_extent (Vec6.get x_shape d))
              (fun i -> reduce rest ((d, i) :: override))
      in
      reduce p.dims []
  end
end

module Amax = struct
  (* Elementwise maximum over [dims] -- ATen's `aten.amax.default`. Same
     axis/keepdim shape rule as [Mean] ([Dims_keepdim]); the only difference is
     the leaf combinator ([S.max_reduce] nested per axis, no divisor). The
     bridge accepts the general `dims`/`keepdim` configuration `mean.dim`
     already does, not just CSATv2's observed `dim=[1]`/`keepdim=true`. *)
  type params = Dims_keepdim.t = { dims : Axis.t list; keepdim : bool }

  let params_jsont : params Jsont.t = Dims_keepdim.jsont ~kind:"amax_params"
  let map_dims = Dims_keepdim.map_dims
  let pp_params = Dims_keepdim.pp

  type t = { params : params; x : Tensor_ref.t }

  let name = "Amax"

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
    Fmt.pf fmt "@[<hv 2>amax@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  let kept_map = Dims_keepdim.kept_map
  let output_shape = Dims_keepdim.output_shape

  (* Same config space as [Mean]'s walk, minus the divisor -- any non-empty
     dim subset is valid, so [cascade] is identity. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; dims : Axis.t list; keepdim : bool }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        dims = [ Axis.H; Axis.W ];
        keepdim = false;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape
    let params (c : cfg) : params = { dims = c.dims; keepdim = c.keepdim }

    let dim_sets =
      Axis.[ [ H; W ]; [ H; W; C ]; [ C ]; [ H ]; [ W ]; [ N; H; W ] ]

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "dims" dim_sets (fun r v -> { r with dims = v });
          field_axis "keepdim" [ true; false ] (fun r v ->
              { r with keepdim = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a dims=[%a] keepdim=%b}" Walk_core.Shape.pp
        c.shape
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
           Axis.pp)
        c.dims c.keepdim
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let zero =
        Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
          ~h:S.index_zero ~w:S.index_zero ~c:S.index_zero
      in
      let base =
        List.fold_left
          (fun v (kin, oax) -> Vec6.copy out ~src:oax ~dst:kin v)
          zero (kept_map p)
      in
      (* Nest one [max_reduce] per reduced axis, mirroring [Mean]'s [sum]
         nest -- see .ai/native_add_op.md's "reductions over named axes". *)
      let rec reduce dims override =
        match dims with
        | [] ->
            S.load x
              (List.fold_left (fun v (a, i) -> Vec6.set v a i) base override)
        | d :: rest ->
            S.max_reduce ~lo:S.index_zero
              ~hi:(S.index_extent (Vec6.get x_shape d))
              (fun i -> reduce rest ((d, i) :: override))
      in
      reduce p.dims []
  end
end

module Vector_norm = struct
  (* L2 vector norm over [dims]: sqrt(sum of squares) -- ATen's
     `aten.linalg_vector_norm.default`, restricted to the schema's `ord=2`
     default (the only value the bridge accepts; a general `ord` needs a
     `pow` of its own and is out of scope). Same axis/keepdim
     shape rule as [Mean]/[Amax] ([Dims_keepdim]); no new [SEMANTICS]
     primitive is needed -- squaring is [S.mul] of the loaded value with
     itself, nested under one [S.sum] per reduced axis exactly as [Mean]'s
     divisor is, with a final [S.sqrt] at the leaf-fold's result rather than
     a division. *)
  type params = Dims_keepdim.t = { dims : Axis.t list; keepdim : bool }

  let params_jsont : params Jsont.t =
    Dims_keepdim.jsont ~kind:"vector_norm_params"

  let map_dims = Dims_keepdim.map_dims
  let pp_params = Dims_keepdim.pp

  type t = { params : params; x : Tensor_ref.t }

  let name = "VectorNorm"

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
    Fmt.pf fmt "@[<hv 2>vector_norm@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params

  let kept_map = Dims_keepdim.kept_map
  let output_shape = Dims_keepdim.output_shape

  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; dims : Axis.t list; keepdim : bool }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        dims = [ Axis.H; Axis.W ];
        keepdim = false;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape
    let params (c : cfg) : params = { dims = c.dims; keepdim = c.keepdim }

    let dim_sets =
      Axis.[ [ H; W ]; [ H; W; C ]; [ C ]; [ H ]; [ W ]; [ N; H; W ] ]

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "dims" dim_sets (fun r v -> { r with dims = v });
          field_axis "keepdim" [ true; false ] (fun r v ->
              { r with keepdim = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a dims=[%a] keepdim=%b}" Walk_core.Shape.pp
        c.shape
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
           Axis.pp)
        c.dims c.keepdim
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let zero =
        Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
          ~h:S.index_zero ~w:S.index_zero ~c:S.index_zero
      in
      let base =
        List.fold_left
          (fun v (kin, oax) -> Vec6.copy out ~src:oax ~dst:kin v)
          zero (kept_map p)
      in
      (* Nest one [sum] per reduced axis, mirroring [Mean]'s nest, but the
         leaf squares the loaded value ([S.mul v v]) rather than reading it
         plain -- the sum-of-squares [linalg_vector_norm]'s ord=2 needs. *)
      let rec reduce dims override =
        match dims with
        | [] ->
            let v =
              S.load x
                (List.fold_left (fun v (a, i) -> Vec6.set v a i) base override)
            in
            S.mul v v
        | d :: rest ->
            S.sum ~lo:S.index_zero
              ~hi:(S.index_extent (Vec6.get x_shape d))
              (fun i -> reduce rest ((d, i) :: override))
      in
      S.sqrt (reduce p.dims [])
  end
end

module Softmax = struct
  (* Softmax over a single axis (ATen's `aten.softmax.int`, `aten._softmax`),
     keeping the input's full shape -- unlike [Mean]/[Amax]/[Vector_norm]
     ([Dims_keepdim]), there is no axis to drop: ATen's `dim` always names
     exactly one axis and softmax never changes rank. For each output pixel,

       y = exp(x - max_axis(x)) / sum_axis(exp(x - max_axis(x)))

     the max and the denominator computed over [axis] at the pixel's fixed
     non-[axis] coordinates -- the ordinary numerically-stable softmax, NOT
     [Attention.Sdpa]'s `_safe_softmax` all-`-inf` zero-row guard: that guard
     is specific to SDPA's internal math-backend helper, not to plain
     `torch.softmax`, which is free to (and does) produce `nan` on an
     all-`-inf` row. No new [SEMANTICS] primitive is needed: [exp],
     [max_reduce] and [sum] already exist and are already exercised
     (`attention.ml`'s [Sdpa.Compute]). See .ai/matmul_softmax_design.md §3. *)
  type params = { axis : Axis.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"softmax_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) = Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis.pp p.axis

  type t = { params : params; x : Tensor_ref.t }

  let name = "Softmax"

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
    Fmt.pf fmt "@[<hv 2>softmax@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params

  (* Softmax rescales, it does not reduce: the output keeps the input's full
     shape, exactly [Norm.RmsNorm.output_shape]'s reasoning. No aggregate to
     bound here -- unlike [Dims_keepdim.reduced_count], this reduces over
     exactly ONE axis, already individually bounded by
     [Kernel.Limits.Hard.extent]; there is no product of several extents for
     a mid-fold wrap to hide in. *)
  let output_shape ~(x_shape : Vec6.shape) (_p : params) = Err.return x_shape

  (* This op's random-walk config space: a 4D NHWC tensor, softmaxed over one
     of the six frame axes. Any axis is valid at any extent (unlike
     [Mean]/[Amax]'s [dims] SUBSET, there is nothing to combine), so
     [cascade] is identity. Lives with the op (per backend). *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; axis : Axis.t }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        axis = Axis.C;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape
    let params (c : cfg) : params = { axis = c.axis }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "axis" Axis.all (fun c v -> { c with axis = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a axis=%a}" Walk_core.Shape.pp c.shape Axis.pp
        c.axis
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let extent = Vec6.get x_shape p.axis in
      (* [score_at k]: the value at [out] with [axis] overridden to [k] --
         mirrors [Attention.Sdpa.Compute]'s [score_at]. *)
      let score_at k = S.load x (Vec6.set out p.axis k) in
      let m =
        S.max_reduce ~lo:S.index_zero ~hi:(S.index_extent extent) score_at
      in
      let z =
        S.sum ~lo:S.index_zero ~hi:(S.index_extent extent) (fun k ->
            S.exp (S.sub (score_at k) m))
      in
      S.div (S.exp (S.sub (S.load x out) m)) z
  end

  (* Authoritative declarative Region computation.  Inputs have already been
     resolved by role at the graph boundary. *)
  module Computation = struct
    open Region_context

    let program ~limits params ~x =
      match partition [ params.axis ] with
      | Error error -> Error error
      | Ok partition ->
          let at axis = load x (Expr.Coord.set output_coord params.axis axis) in
          let extent = Dim.to_int (Vec6.get x.Tensor_sig.shape params.axis) in
          let max =
            Expr.Builder.run
              (Expr.Builder.reduction ~kind:Expr.Reduction.Max
                 ~lo:Expr.Index.zero ~hi:(Expr.Index.const extent) (fun axis ->
                   Expr.Builder.return (at axis)))
          in
          program
            (Region_program.Builder.run
               (Region_program.Builder.scalar max (fun max ->
                    let denominator =
                      Expr.Builder.run
                        (Expr.Builder.reduction ~kind:Expr.Reduction.Sum
                           ~lo:Expr.Index.zero ~hi:(Expr.Index.const extent)
                           (fun axis ->
                             Expr.Builder.return
                               (Expr.Value.exp (Expr.Value.sub (at axis) max))))
                    in
                    Region_program.Builder.scalar denominator
                      (fun denominator ->
                        Region_program.Builder.finish
                          ~max_size:limits.Kernel.Limits.max_size
                          ~max_depth:limits.Kernel.Limits.max_depth ~partition
                          ~output:
                            (Expr.Value.div
                               (Expr.Value.exp
                                  (Expr.Value.sub (load_output x) max))
                               denominator)))))
  end
end
