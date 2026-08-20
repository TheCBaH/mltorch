(* Scaled dot-product attention (`aten.scaled_dot_product_attention.default`),
   scoped to exactly what keeps ATen's CPU dispatch on the flash-attention
   kernel (op8-impl.md F4) -- the only oracle this row is verified against:

     - Q/K/V are rank-4, right-aligned to [D=batch, H=heads, W=sequence,
       C=head_dim] (F7). No relayout is needed: the axes the op reads are the
       axes [Aten_shape.of_aten] lands the data on.
     - the value feature dimension equals query/key's (Ev = E), so
       [output_shape] is exactly [query_shape];
     - the mask, if present, is additive f32 (never boolean, F10) and its
       broadcast shape is exactly the flash-admissible set: 2D [Wq|1, Wk|1] or
       4D [D|1, H|1, Wq|1, Wk|1] against [score_shape];
     - no dropout, causal masking, GQA or negative explicit scale -- each a
       typed rejection at the two importers (commit 3), not a case this module
       computes.

   Every one of those is a flash-ORACLE boundary, not a mathematical one: ATen
   silently falls back to its `math` backend when a gate fails (there is no
   error), and a single [atol_for_target] cannot represent two different
   kernels. Rank itself is NOT checked here and cannot be: [Tensor_sig.t]
   retains only a [Vec6.shape], and a rank-2 mask [Wq,Wk], a rank-3
   [1,Wq,Wk] and a rank-4 [1,1,Wq,Wk] all normalize to the SAME Native shape
   (op8-impl.md F13) -- only the middle one is flash-admissible, and only an
   importer reading the raw ATen rank BEFORE [Aten_shape.of_aten] can tell
   them apart. Every rank rule therefore lives in the two importers, and a
   standalone Native or JSON-decoded graph carries no ATen rank at all.

   The arithmetic is ATen's `math` backend's structure (F3), not
   op8.md's literal `dot(q,k) * scale`: neither ATen backend computes that.
   The scale is split as its own square root and applied to BOTH operands
   before the matmul (`attention.cpp`'s `_scaled_dot_product_attention_math`,
   for overflow stability under fp32/fp16), and confirmed independently by the
   corpus (vit_b_32 nodes 30/32: `mul.Scalar` by [64 ** -0.25] on both the
   query and the key, for head_dim 64). `scale * dot(q,k)` and
   `dot(q*sqrt(scale), k*sqrt(scale))` are not f32-equal, so implementing
   op8.md's formula is a required MUTATION target here, not the contract.
   Softmax is ATen's `_safe_softmax`: it runs UNCONDITIONALLY (not only when a
   mask is present), and a row whose scores are all -inf yields 0, not NaN.

   No new semantics primitive is needed (op8-impl.md F5): [exp], [max_reduce],
   [sum], [lt] and [select] already exist and are already exercised
   (`ops/window_axis.ml`'s use of [max_reduce]). *)

module Sdpa = struct
  (* The default vs an explicit scale are NOT the same computation: the
     default is [1 / sqrt(head_dim)], which depends on a shape only known at
     evaluation time, so it cannot be folded into a plain [float] the way
     [Norm]'s [eps] is. A closed variant, not a [float option] (CLAUDE.md's
     payload rule): the JSON distinction between "use the default" and
     "explicit 1.0" must be exact, and it is what makes commit 5's Model
     Explorer fixtures (default vs explicit scale) pinnable. *)
  module Scale = struct
    type t = Default | Explicit of float
  end

  (* The typed rejection boundary BOTH importers enforce (op8-impl.md commit
     3), shared for [Op_config.Bad]'s reason: the two importers must reject
     the same values, and two spellings of "dropout_p is not supported" is
     one drift away from two contracts. [Rank] is for an argument with MORE
     than one admissible rank (the mask, {2,4}) -- a single [expected] cannot
     say that. *)
  module Reject = struct
    type rank = { arg_name : string; expected : int list; got : int }

    type t =
      | Dropout of float (* dropout_p <> 0 *)
      | Causal (* is_causal = true *)
      | Gqa (* enable_gqa = true *)
      | Boolean_mask (* attn_mask is bool: additive f32 only (F10) *)
      | Rank of rank
      | Negative_scale of float (* legal in ATen, specially handled; F3 *)
      | Non_finite_scale of float

    let pp fmt = function
      | Dropout p -> Fmt.pf fmt "sdpa: dropout_p=%g is not supported (only 0)" p
      | Causal -> Fmt.string fmt "sdpa: is_causal=true is not supported"
      | Gqa -> Fmt.string fmt "sdpa: enable_gqa=true is not supported"
      | Boolean_mask ->
          Fmt.string fmt
            "sdpa: a boolean attn_mask is not supported; only an additive f32 \
             mask is (never reinterpreted from bool)"
      | Rank { arg_name; expected; got } ->
          Fmt.pf fmt "sdpa: %s has rank %d, expected %a" arg_name got
            Fmt.(list ~sep:(any " or ") int)
            expected
      | Negative_scale s ->
          Fmt.pf fmt
            "sdpa: explicit scale=%g is negative; ATen handles this specially \
             and this row does not implement it"
            s
      | Non_finite_scale s ->
          Fmt.pf fmt "sdpa: explicit scale=%g is not finite" s
  end

  type params = { scale : Scale.t }

  let scale_jsont : Scale.t Jsont.t =
    Jsont.map ~kind:"sdpa_scale"
      ~dec:
        (Json_util.union ~kind:"sdpa_scale"
           [
             ("default", fun _ -> Scale.Default);
             ( "explicit",
               fun v -> Scale.Explicit (Json_util.dec Json_util.f32_jsont v) );
           ])
      ~enc:(function
        | Scale.Default -> Json_util.single ~case:"default" Json_util.jnull
        | Scale.Explicit s ->
            Json_util.single ~case:"explicit"
              (Json_util.enc Json_util.f32_jsont s))
      Jsont.json

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"sdpa_params" (fun scale -> { scale })
    |> Jsont.Object.mem "scale" scale_jsont ~enc:(fun p -> p.scale)
    |> Jsont.Object.finish

  let pp_scale fmt = function
    | Scale.Default -> Fmt.string fmt "default"
    | Scale.Explicit s -> Fmt.pf fmt "explicit(%a)" Fmt.float s

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{scale=%a}@]" pp_scale p.scale

  (* No [dropout_p], [is_causal] or [enable_gqa] fields: the scope admits only
     [dropout_p = 0], [is_causal = false], [enable_gqa = false], each rejected
     by both importers (commit 3) before a graph carrying anything else could
     be built. Carrying a flag this module never reads would be a silent
     invitation to wire it up without also widening the verified scope. *)
  type t = {
    params : params;
    query : Tensor_ref.t;
    key : Tensor_ref.t;
    value : Tensor_ref.t;
    mask : Tensor_ref.t option;
  }

  let name = "Sdpa"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          query = get "query" Tensor_ref.jsont;
          key = get "key" Tensor_ref.jsont;
          value = get "value" Tensor_ref.jsont;
          mask = Json_util.opt_field ms "mask" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt_mask =
          match t.mask with None -> [] | Some r -> [ ("mask", ref_ r) ]
        in
        Json_util.jobj
          ([ ("params", Json_util.enc params_jsont t.params) ]
          @ opt_mask
          @ [
              ("query", ref_ t.query);
              ("key", ref_ t.key);
              ("value", ref_ t.value);
            ]))
      Jsont.json

  (* Order pinned (query, key, value, then mask if present): Model Explorer's
     operand-edge slot ids are positional (commit 5). *)
  let operands (t : t) = [ t.query; t.key; t.value ] @ Option.to_list t.mask

  let map_operands f (t : t) =
    {
      t with
      query = f t.query;
      key = f t.key;
      value = f t.value;
      mask = Option.map f t.mask;
    }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>sdpa@ query=%a@ key=%a@ value=%a@ mask=%a@ params=%a@]"
      pp_ref t.query pp_ref t.key pp_ref t.value
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.mask pp_params t.params

  (* The score's own shape [D,H,Wq,Wk]: the SAME [D,H,W=Wq] as [query_shape]
     (and hence [output_shape]), with [C] replaced by the key's sequence
     extent. NOT the query shape (whose C is the head dim, E) whenever
     [Wk <> E]. Owned here and used by three callers that must not disagree
     (op8-impl.md F11): [output_shape]'s mask validation, the total-work
     bound below, and [Compute]'s broadcast load of the mask. *)
  let score_shape ~(query_shape : Vec6.shape) ~(key_shape : Vec6.shape) =
    Vec6.set query_shape Axis.C (Vec6.get key_shape Axis.W)

  let check_extent ~check ~axis (a : Vec6.shape) (b : Vec6.shape) =
    let lhs = Vec6.get a axis and rhs = Vec6.get b axis in
    if Dim.equal lhs rhs then Err.return ()
    else
      Err.fail
        (`Sdpa (Shape_error.Sdpa.Extent_mismatch { axis; check; lhs; rhs }))

  (* Total work D*H*Wq*Wk*E*E (op8-impl.md F12): the op recomputes the score,
     row max and denominator per OUTPUT feature (stated as a deliberate choice
     in [Compute] below), so direct work is proportional to all six factors,
     not to the score count or the output numel alone -- the F12 counter-
     example ([D=H=1, Wq=Wk=E=Ev=1024]) passes both of those by six orders of
     magnitude. No single [Vec6.shape] holds all six factors ([E] appears
     twice, and none of query/key/value's shapes has both [Wq] and [Wk]), so
     this is its own divide-before-multiply fold, following
     [Kernel.Bounds.signature] -- CLAUDE.md's 32-bit-aggregate rule, and
     [lib/native] is js_of_ocaml-reachable. A check on the wrapped product
     would not be a bound. *)
  let total_work_bounded ~limit ~(query_shape : Vec6.shape)
      ~(key_shape : Vec6.shape) =
    let open Shape_error.Sdpa in
    let e = Vec6.get query_shape Axis.C in
    let factors =
      [
        (Outer_n, Vec6.get query_shape Axis.N);
        (Outer_t, Vec6.get query_shape Axis.T);
        (Batch, Vec6.get query_shape Axis.D);
        (Heads, Vec6.get query_shape Axis.H);
        (Query_len, Vec6.get query_shape Axis.W);
        (Key_len, Vec6.get key_shape Axis.W);
        (Head_dim, e);
        (Head_dim, e);
      ]
    in
    let ceiling = Int64.pred limit in
    Err.List.fold_left
      (fun prefix (factor, extent) ->
        let extent = Int64.of_int (Dim.to_int extent) in
        if Int64.compare prefix (Int64.div ceiling extent) > 0 then
          Err.fail
            (`Sdpa (Total_work_over_limit { factor; prefix; extent; limit }))
        else Err.return (Int64.mul prefix extent))
      1L factors

  (* Validates everything the six-axis frame CAN express (op8-impl.md F13):
     [N]/[T]/[D]/[H] agreement across all three operands, [Ev = E] (the flash-oracle
     constraint that makes the output shape well-defined as [query_shape]),
     key/value sharing a sequence extent, and -- when a mask is given -- its
     per-axis broadcast compatibility against [score_shape]. It does NOT check
     rank, because it cannot: [Tensor_sig.t] has no rank field, only a
     [Vec6.shape], so a rank-2, rank-3 and rank-4 mask that normalize to the
     same frame are indistinguishable here. Rank -- Q/K/V rank 4, mask rank in
     {2,4} -- is an IMPORTER obligation (commit 3), enforced on the raw ATen
     tensor before [Tensor_bridge.of_aten] erases it. A standalone Native or
     JSON-decoded graph carries no ATen rank at all, so that check would be
     meaningless here, not merely omitted. *)
  let output_shape ~(query_shape : Vec6.shape) ~(key_shape : Vec6.shape)
      ~(value_shape : Vec6.shape) ~(mask_shape : Vec6.shape option) =
    let open Err.Syntax in
    (* N/T carry no semantic role in this op (F7 names only D/H/W/C) but must
       still agree across every operand: [Compute] reads key/value at the
       OUTPUT coordinate's unchanged N/T (never reduced through
       [broadcast_coord], exactly like D/H below), so an unequal N or T
       either reads an operand out of bounds or silently drops part of it
       (op8-impl-review.md P1). Checked here, the same as D/H, rather than
       rejected outright, so a hand-built graph that genuinely wants extra
       batch-like axes on N/T still works -- it just has to agree on them. *)
    let* () =
      check_extent ~check:`Query_key ~axis:Axis.N query_shape key_shape
    in
    let* () =
      check_extent ~check:`Query_value ~axis:Axis.N query_shape value_shape
    in
    let* () =
      check_extent ~check:`Query_key ~axis:Axis.T query_shape key_shape
    in
    let* () =
      check_extent ~check:`Query_value ~axis:Axis.T query_shape value_shape
    in
    let* () =
      check_extent ~check:`Query_key ~axis:Axis.D query_shape key_shape
    in
    let* () =
      check_extent ~check:`Query_value ~axis:Axis.D query_shape value_shape
    in
    let* () =
      check_extent ~check:`Query_key ~axis:Axis.H query_shape key_shape
    in
    let* () =
      check_extent ~check:`Query_value ~axis:Axis.H query_shape value_shape
    in
    (* Ev = E (F4): a flash-oracle constraint, not a mathematical one -- the
       importers (commit 3) say so in their diagnostic. Here it is simply what
       makes the output shape well-defined as [query_shape]. *)
    let* () =
      check_extent ~check:`Query_key ~axis:Axis.C query_shape key_shape
    in
    let* () =
      check_extent ~check:`Query_value ~axis:Axis.C query_shape value_shape
    in
    let* () =
      check_extent ~check:`Key_value ~axis:Axis.W key_shape value_shape
    in
    let sshape = score_shape ~query_shape ~key_shape in
    let* () =
      match mask_shape with
      | None -> Err.return ()
      | Some ms ->
          Err.List.fold_left
            (fun () axis ->
              let mask = Vec6.get ms axis and score = Vec6.get sshape axis in
              if Dim.equal mask score || Dim.equal mask Dim.one then
                Err.return ()
              else
                Err.fail
                  (`Sdpa (Shape_error.Sdpa.Mask_shape { axis; mask; score })))
            () Axis.all
    in
    let* (_ : int64) =
      (Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel sshape
        :> (int64, [> Shape_error.t ]) Err.t)
    in
    let* (_ : int64) =
      (Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel query_shape
        :> (int64, [> Shape_error.t ]) Err.t)
    in
    let* (_ : int64) =
      total_work_bounded ~limit:Kernel.Limits.Hard.numel ~query_shape ~key_shape
    in
    Err.return query_shape

  (* Config space for [lib/native_op_walk]'s Direct-vs-Symbolic fuzz walk
     (op8-impl.md commit 1 step 8) -- NOT the ATen-oracle recipe (commit 4's
     [Recipe_sdpa]), which additionally has to keep every point on the flash
     kernel (F4). This walk has no ATen oracle to stay on, so its only
     obligation is a VALID Native graph: [batch]/[heads]/[wq]/[wk]/[e] are
     independent fields, but every shape [build] derives is computed FROM
     them (never five independently-drawn [Vec6.shape]s), so a mismatched
     query/key/value/mask shape is unrepresentable by construction and
     [cascade] has nothing to repair. *)
  module Walk (_ : Walk_core.Limits.S) = struct
    type mask_cfg = Absent | Present

    type cfg = {
      batch : int;
      heads : int;
      wq : int;
      wk : int;
      e : int;
      mask : mask_cfg;
      scale : Scale.t;
    }

    let initial =
      {
        batch = 1;
        heads = 2;
        wq = 3;
        wk = 4;
        e = 5;
        mask = Present;
        scale = Scale.Default;
      }

    let cascade c = c

    let query_shape (c : cfg) =
      Vec6.shape ~n:1 ~t:1 ~d:c.batch ~h:c.heads ~w:c.wq ~c:c.e

    let key_shape (c : cfg) =
      Vec6.shape ~n:1 ~t:1 ~d:c.batch ~h:c.heads ~w:c.wk ~c:c.e

    let mask_shape (c : cfg) =
      score_shape ~query_shape:(query_shape c) ~key_shape:(key_shape c)

    let mask_present (c : cfg) = c.mask = Present
    let params (c : cfg) : params = { scale = c.scale }

    let axes =
      Walk_core.Walk.
        [
          field_axis "batch" [ 1; 2 ] (fun c v -> { c with batch = v });
          field_axis "heads" [ 1; 2; 3 ] (fun c v -> { c with heads = v });
          field_axis "wq" [ 1; 2; 3; 5 ] (fun c v -> { c with wq = v });
          field_axis "wk" [ 1; 2; 4; 6 ] (fun c v -> { c with wk = v });
          field_axis "e" [ 1; 3; 5 ] (fun c v -> { c with e = v });
          field_axis "mask" [ Absent; Present ] (fun c v -> { c with mask = v });
          field_axis "scale"
            [ Scale.Default; Scale.Explicit 0.5; Scale.Explicit 1.0 ]
            (fun c v -> { c with scale = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{batch=%d heads=%d wq=%d wk=%d e=%d mask=%s scale=%a}"
        c.batch c.heads c.wq c.wk c.e
        (match c.mask with Absent -> "absent" | Present -> "present")
        pp_scale c.scale
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [query_shape]/[key_shape] size the two reductions ([E], [Wk]); [mask]
       is never optional here -- [Eval_op] fills an absent one at the
       all-ones shape (below), so [Compute] always has a real handle and
       [mask_shape] to broadcast it against. *)
    let pixel (p : params) ~(query_shape : Vec6.shape) ~(key_shape : Vec6.shape)
        ~(mask_shape : Vec6.shape) ~query ~key ~value ~mask
        (out : Semantics.position S.index Vec6.t) =
      let e_extent = Vec6.get query_shape Axis.C in
      let wk_extent = Vec6.get key_shape Axis.W in
      (* [scale], NOT its square root: [Scale.Default] is [1 / sqrt(E)], the
         one place this module reads the head-dim extent rather than taking it
         as a parameter, because it is genuinely a function of [query_shape].
         A negative or non-finite explicit scale is rejected upstream (the
         importers, commit 3); this module assumes [scale >= 0]. *)
      let scale =
        match p.scale with
        | Scale.Explicit s -> S.const s
        | Scale.Default ->
            S.div (S.const 1.)
              (S.sqrt (S.const (float_of_int (Dim.to_int e_extent))))
      in
      (* Split as its OWN square root and applied to BOTH operands before the
         dot (F3) -- this is what makes [dot(q,k) * scale] the wrong mutation
         and [dot(q*sf, k*sf)] the contract. *)
      let sf = S.sqrt scale in
      (* [score_at k]: the row's logit at key position [k], mask included.
         [mask] is read through [Pointwise.broadcast_coord] at the SCORE
         coordinate (out's [D,H,W=Wq], [C] replaced by [k]) -- never the
         output coordinate, and never a bare [S.load]: [load] is strict, so a
         broadcast (extent-1) mask axis must be reduced to index 0 first
         (op8-impl.md F11). *)
      let score_at k =
        let dot =
          S.sum ~lo:S.index_zero ~hi:(S.index_extent e_extent) (fun e ->
              let q_idx = Vec6.set out Axis.C e in
              let k_idx = Vec6.set (Vec6.set out Axis.C e) Axis.W k in
              S.mul
                (S.mul (S.load query q_idx) sf)
                (S.mul (S.load key k_idx) sf))
        in
        let score_coord = Vec6.set out Axis.C k in
        let m_idx =
          Pointwise.broadcast_coord ~index_zero:S.index_zero mask_shape
            score_coord
        in
        S.add dot (S.load mask m_idx)
      in
      (* Stable softmax over the key axis, recomputing [score_at] at every
         use: [Symbolic.t] is a BUILDER computation, not a memoized node
         (op7-impl.md F4), so each of [m]'s, [z]'s and the numerator's uses of
         [score_at] re-emits its whole sub-tree. Accepted deliberately (F12
         states the bound this costs, not the score count) and measured
         (op8-impl.md commit 0 step 3: depth 12, size 121 nodes at Wk=E=32,
         both flat regardless of the reduction extents -- far under
         [Kernel.Limits.Hard.depth]/[eval_depth]). *)
      let m =
        S.max_reduce ~lo:S.index_zero ~hi:(S.index_extent wk_extent) score_at
      in
      let z =
        S.sum ~lo:S.index_zero ~hi:(S.index_extent wk_extent) (fun k ->
            S.exp (S.sub (score_at k) m))
      in
      let numer =
        S.sum ~lo:S.index_zero ~hi:(S.index_extent wk_extent) (fun k ->
            let p = S.div (S.exp (S.sub (score_at k) m)) z in
            S.mul p (S.load value (Vec6.set out Axis.W k)))
      in
      (* `_safe_softmax` (F3): unconditional, not only under a mask. A row
         whose scores are all -inf has [m = -inf], and yields 0, not NaN --
         [S.div _ z] would otherwise divide 0 by 0. *)
      let neg_inf = S.const Float.neg_infinity in
      S.select (S.lt neg_inf m) numer (S.const 0.)
  end
end
