(* upsample_bilinear2d.vec: bilinear resize to an explicit output size, both
   `align_corners` values. The bridge/importer resolve a `scale_factors`
   overload to an explicit size before this op is ever built, so the op
   itself only ever sees a size -- see op_bridge.ml/native_interp.ml. The
   per-axis coordinate transform is ATen's own
   `area_pixel_compute_source_index`:

     align_corners=true:  src = out_idx * (in_extent - 1) / (out_extent - 1)
                           (0 when out_extent = 1)
     align_corners=false: src = max(0, (out_idx + 0.5) * in_extent/out_extent
                                        - 0.5)

   [Bilinear_axis] computes both WITHOUT a real division: [in_extent] and
   [out_extent] are both static at graph-construction time (ordinary shape
   extents, not tensor data), so every source coordinate is an exact
   rational number, and `floor(src)` plus its fractional part are exactly an
   INTEGER floor division and remainder -- primitives
   [Semantics.SEMANTICS] already has (the same idiom [Window_axis]/
   [Adaptive_axis] use), not a new one. This is exact rational arithmetic,
   not ATen's own float64 division, so a result is `Equivalent` (a possible
   1-ULP rounding difference), never asserted `Identical`. *)

module Bilinear_axis = struct
  let limit = Kernel.Limits.Hard.extent

  (* Both formulas' numerator is bounded by `O(in_extent * out_extent)` --
     `(in_extent-1)*(out_extent-1)` for align_corners=true, up to
     `(2*out_extent-1)*in_extent` for align_corners=false -- so one bound
     covers both: the same "bound the product before multiplying" rule
     [Pool.Adaptive_axis.check] follows for its own [input * output]
     aggregate. Both factors are non-negative ([Dim.extent]/[Op_config.Pos.t]
     are >= 1). *)
  let check ~axis ~(in_extent : Dim.extent Dim.t)
      ~(out_extent : Op_config.Pos.t) : (unit, Shape_error.t) Err.t =
    let i = Int64.of_int (in_extent :> int) in
    let o = Int64.of_int (out_extent :> int) in
    let two_o = Int64.mul 2L o in
    if i >= limit || two_o >= limit || i > Int64.div (Int64.sub limit 1L) two_o
    then
      let aggregate =
        if i >= limit || two_o >= limit then limit else Int64.mul i two_o
      in
      Err.fail
        (`Resize
           Shape_error.Resize.{ axis; in_extent; out_extent; aggregate; limit })
    else Err.return ()

  module Compute (S : sig
    type 'role index
    type t

    val index_zero : Semantics.position index
    val index_const : int -> Semantics.delta index
    val of_index : Semantics.position index -> Semantics.delta index

    val index_add :
      Semantics.delta index -> Semantics.delta index -> Semantics.delta index

    val index_scale : int -> Semantics.delta index -> Semantics.delta index

    val index_floor_div_pos :
      Semantics.delta index -> Op_config.Pos.t -> Semantics.delta index

    val index_min :
      Semantics.delta index -> Semantics.delta index -> Semantics.delta index

    val index_max :
      Semantics.delta index -> Semantics.delta index -> Semantics.delta index

    val clamp_low : Semantics.delta index -> Semantics.position index
    val value_of_index : Semantics.delta index -> t
    val const : float -> t
    val sub : t -> t -> t
    val div : t -> t -> t
  end) =
  struct
    type endpoints = {
      i0 : Semantics.position S.index;
      i1 : Semantics.position S.index;
      lambda0 : S.t;
      lambda1 : S.t;
    }

    (* [align_corners=true] with [out_extent = 1] is the one genuinely
       degenerate case: ATen's own scale is 0 (not a division by
       [out_extent - 1] = 0), so the source is always index 0 regardless of
       [in_extent]. Every other combination -- including
       [align_corners=false] at [out_extent = 1] -- divides by a positive
       denominator ([out_extent - 1] here, [2 * out_extent] below) and needs
       no special case. *)
    let endpoints ~(align_corners : bool) ~(in_extent : Dim.extent Dim.t)
        ~(out_extent : Op_config.Pos.t) (out_axis : Semantics.position S.index)
        : endpoints =
      let in_e = (in_extent :> int) and out_e = (out_extent :> int) in
      if align_corners && out_e = 1 then
        {
          i0 = S.index_zero;
          i1 = S.index_zero;
          lambda0 = S.const 1.0;
          lambda1 = S.const 0.0;
        }
      else
        let numerator, denom =
          if align_corners then
            (S.index_scale (in_e - 1) (S.of_index out_axis), out_e - 1)
          else
            (* [(2*dst+1)*in_extent - out_extent], clamped to >= 0 (ATen
               clamps the real-valued source, not the index -- clamping the
               numerator first is equivalent since the denominator is always
               positive) -- over the doubled denominator [2*out_extent],
               avoiding the half-integer offset. *)
            let two_dst_plus_1 =
              S.index_add
                (S.index_scale 2 (S.of_index out_axis))
                (S.index_const 1)
            in
            let raw =
              S.index_add
                (S.index_scale in_e two_dst_plus_1)
                (S.index_const (-out_e))
            in
            (S.index_max raw (S.index_const 0), 2 * out_e)
        in
        let d_pos = Op_config.Pos.of_int denom in
        let i0 = S.clamp_low (S.index_floor_div_pos numerator d_pos) in
        (* [i0 <= in_extent - 1] always holds by construction (provable from
           [in_extent, out_extent >= 1] for both formulas), so this [index_min]
           only guards [i1 = i0 + 1] against running one past the last row/
           column -- it never corrects [i0] itself. *)
        let i1 =
          S.clamp_low
            (S.index_min
               (S.index_const (in_e - 1))
               (S.index_add (S.of_index i0) (S.index_const 1)))
        in
        let remainder =
          S.index_add numerator
            (S.index_scale (-1) (S.index_scale denom (S.of_index i0)))
        in
        let lambda1 =
          S.div (S.value_of_index remainder) (S.const (float_of_int denom))
        in
        let lambda0 = S.sub (S.const 1.0) lambda1 in
        { i0; i1; lambda0; lambda1 }
  end
end

(* upsample_nearest2d.vec: nearest-neighbor resize to an explicit output
   size. Same "bridge/importer resolve scale_factors before this op is ever
   built" split as [Bilinear2d] -- see op_bridge.ml/native_interp.ml. ATen's
   own `nearest_neighbor_compute_source_index` (UpSample.h) has no
   [align_corners] concept at all (there is nothing to interpolate between):
   `src = floor(out_idx * in_extent / out_extent)`, an EXACT integer floor
   division over the same static graph-construction-time extents
   [Bilinear_axis]/[Window_axis]/[Adaptive_axis] already divide -- so this,
   like those, is `Identical` to ATen, not merely `Equivalent`: no float
   division or rounding is involved anywhere. *)
module Nearest_axis = struct
  let limit = Kernel.Limits.Hard.extent

  (* [out_idx * in_extent] is the aggregate: both factors are individually
     bounded ([Dim.extent]/[Op_config.Pos.t] are >= 1), and the product must
     be bounded before it is computed, the same "divide before multiply" rule
     [Bilinear_axis.check]/[Pool.Adaptive_axis.check] follow for their own
     [in_extent * out_extent] aggregate -- [out_idx < out_extent], so bounding
     the full product covers every reachable [out_idx]. *)
  let check ~axis ~(in_extent : Dim.extent Dim.t)
      ~(out_extent : Op_config.Pos.t) : (unit, Shape_error.t) Err.t =
    let i = Int64.of_int (in_extent :> int) in
    let o = Int64.of_int (out_extent :> int) in
    if i >= limit || o >= limit || i > Int64.div (Int64.sub limit 1L) o then
      let aggregate =
        if i >= limit || o >= limit then limit else Int64.mul i o
      in
      Err.fail
        (`Resize_nearest
           Shape_error.Resize_nearest.
             { axis; in_extent; out_extent; aggregate; limit })
    else Err.return ()

  module Compute (S : sig
    type 'role index

    val of_index : Semantics.position index -> Semantics.delta index
    val index_scale : int -> Semantics.delta index -> Semantics.delta index

    val index_floor_div_pos :
      Semantics.delta index -> Op_config.Pos.t -> Semantics.delta index

    val assume_index : Semantics.delta index -> Semantics.position index
  end) =
  struct
    (* [floor(out_idx * in_extent / out_extent)] is provably < [in_extent] for
       [out_idx < out_extent] (both positive): [out_idx <= out_extent - 1], so
       [out_idx * in_extent <= (out_extent - 1) * in_extent < out_extent *
       in_extent], and the floor division by [out_extent] is therefore
       strictly less than [in_extent] -- the same "bounded by construction"
       argument [Pool.Adaptive_axis.Compute.bin]'s [lo] already relies on, so
       no explicit [index_min] clamp is needed. *)
    let index ~(in_extent : Dim.extent Dim.t) ~(out_extent : Op_config.Pos.t)
        (out_axis : Semantics.position S.index) : Semantics.position S.index =
      let scale = (in_extent :> int) in
      let out = S.of_index out_axis in
      S.assume_index
        (S.index_floor_div_pos (S.index_scale scale out) out_extent)
  end
end

module Nearest2d = struct
  (* Same [output_size] shape as [Pool.AdaptiveAvgPool2d.params] -- no
     [align_corners] field, unlike [Bilinear2d.params]: nearest-neighbor
     resize has nothing for it to change (see the module doc). *)
  type params = { output_size : Op_config.Pos.t Op_config.Hw.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"upsample_nearest2d_params" (fun output_size ->
        { output_size })
    |> Jsont.Object.mem "output_size" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.output_size)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "{output_size=%a}"
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.output_size

  type t = { params : params; x : Tensor_ref.t }

  let name = "Upsample_nearest2d"

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
    Fmt.pf fmt "@[<hv 2>upsample_nearest2d@ x=%a@ params=%a@]" pp_ref t.x
      pp_params t.params

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let* () =
      Nearest_axis.check ~axis:Axis.H ~in_extent:(Vec6.get x_shape Axis.H)
        ~out_extent:p.output_size.h
    in
    let* () =
      Nearest_axis.check ~axis:Axis.W ~in_extent:(Vec6.get x_shape Axis.W)
        ~out_extent:p.output_size.w
    in
    Err.return
      (Vec6.set
         (Vec6.set x_shape Axis.H (Dim.extent (p.output_size.h :> int)))
         Axis.W
         (Dim.extent (p.output_size.w :> int)))

  module Compute (S : Semantics.SEMANTICS) = struct
    module Ne = Nearest_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let h =
        Ne.index ~in_extent:(Vec6.get x_shape Axis.H)
          ~out_extent:p.output_size.h (Vec6.get out Axis.H)
      in
      let w =
        Ne.index ~in_extent:(Vec6.get x_shape Axis.W)
          ~out_extent:p.output_size.w (Vec6.get out Axis.W)
      in
      S.load x (out |> Vec6.set_h h |> Vec6.set_w w)
  end
end

module Bilinear2d = struct
  (* Same [output_size] field shape as [Pool.AdaptiveAvgPool2d.params] (an
     H/W pair, positive -- the engine has no empty extent), plus
     [align_corners], which changes the coordinate transform itself (see
     [Bilinear_axis.endpoints]). *)
  type params = {
    output_size : Op_config.Pos.t Op_config.Hw.t;
    align_corners : bool;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"upsample_bilinear2d_params"
      (fun align_corners output_size -> { output_size; align_corners })
    |> Jsont.Object.mem "align_corners" Jsont.bool ~enc:(fun p ->
        p.align_corners)
    |> Jsont.Object.mem "output_size" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.output_size)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "{output_size=%a;@ align_corners=%b}"
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.output_size p.align_corners

  type t = { params : params; x : Tensor_ref.t }

  let name = "Upsample_bilinear2d"

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
    Fmt.pf fmt "@[<hv 2>upsample_bilinear2d@ x=%a@ params=%a@]" pp_ref t.x
      pp_params t.params

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let* () =
      Bilinear_axis.check ~axis:Axis.H ~in_extent:(Vec6.get x_shape Axis.H)
        ~out_extent:p.output_size.h
    in
    let* () =
      Bilinear_axis.check ~axis:Axis.W ~in_extent:(Vec6.get x_shape Axis.W)
        ~out_extent:p.output_size.w
    in
    Err.return
      (Vec6.set
         (Vec6.set x_shape Axis.H (Dim.extent (p.output_size.h :> int)))
         Axis.W
         (Dim.extent (p.output_size.w :> int)))

  module Compute (S : Semantics.SEMANTICS) = struct
    module Bi = Bilinear_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let eh =
        Bi.endpoints ~align_corners:p.align_corners
          ~in_extent:(Vec6.get x_shape Axis.H) ~out_extent:p.output_size.h
          (Vec6.get out Axis.H)
      in
      let ew =
        Bi.endpoints ~align_corners:p.align_corners
          ~in_extent:(Vec6.get x_shape Axis.W) ~out_extent:p.output_size.w
          (Vec6.get out Axis.W)
      in
      let read h w = S.load x (out |> Vec6.set_h h |> Vec6.set_w w) in
      let top =
        S.add
          (S.mul ew.lambda0 (read eh.i0 ew.i0))
          (S.mul ew.lambda1 (read eh.i0 ew.i1))
      in
      let bottom =
        S.add
          (S.mul ew.lambda0 (read eh.i1 ew.i0))
          (S.mul ew.lambda1 (read eh.i1 ew.i1))
      in
      S.add (S.mul eh.lambda0 top) (S.mul eh.lambda1 bottom)
  end
end

(* upsample_bicubic2d.vec: bicubic resize to an explicit output size, both
   `align_corners` values. Same "bridge/importer resolve scale_factors before
   this op is ever built" split as [Bilinear2d]/[Nearest2d] -- see
   op_bridge.ml/native_interp.ml. The per-axis coordinate transform reuses
   ATen's OWN `area_pixel_compute_source_index`, the same function
   [Bilinear_axis.endpoints] evaluates -- with one real difference, the
   reason this needs its own review rather than reusing [Bilinear_axis]:
   that function only clamps a NEGATIVE source coordinate to 0 when its
   `cubic` argument is false. Bilinear passes false (there is nothing to
   extrapolate from outside the source, so ATen clamps before dividing);
   bicubic passes true, keeping a genuinely negative rational source
   coordinate at the low boundary and instead clamping each of its four
   TAPS independently ([upsample_get_value_bounded]) -- an out-of-range tap
   reads the nearest edge element, not a synthesized zero. Getting this
   backwards (clamping the coordinate the way [Bilinear_axis] does) would
   silently double-count the edge row/column relative to real ATen at every
   boundary output pixel, not just crash or mis-round.

   The four cubic weights are ATen's own `cubic_convolution1`/
   `cubic_convolution2` (UpSampleKernel.h), Catmull-Rom-style with `A =
   -0.75`, evaluated over the fractional part [t] via the same exact-rational
   floor/remainder split [Bilinear_axis.endpoints] uses for its own [lambda]
   -- so [t] itself has no rounding error, even though the four weights
   evaluate a real cubic polynomial in it and so cannot be exact rationals
   themselves the way [Bilinear_axis]'s two linear weights are. `Equivalent`
   to ATen (rounding may differ), never `Identical`, for the same reason
   [Bilinear2d] is. *)
module Bicubic_axis = struct
  let limit = Kernel.Limits.Hard.extent

  (* Same aggregate bound as [Bilinear_axis.check], and for the same reason:
     both formulas' numerator is [O(in_extent * out_extent)] --
     [(in_extent-1)*(out_extent-1)] for align_corners=true,
     [(2*out_extent-1)*in_extent] for align_corners=false -- bounded here
     before [endpoints] ever computes it, never after. Not reused directly
     (rather than duplicated) only because the error variant must name this
     op, not [Bilinear2d]. *)
  let check ~axis ~(in_extent : Dim.extent Dim.t)
      ~(out_extent : Op_config.Pos.t) : (unit, Shape_error.t) Err.t =
    let i = Int64.of_int (in_extent :> int) in
    let o = Int64.of_int (out_extent :> int) in
    let two_o = Int64.mul 2L o in
    if i >= limit || two_o >= limit || i > Int64.div (Int64.sub limit 1L) two_o
    then
      let aggregate =
        if i >= limit || two_o >= limit then limit else Int64.mul i two_o
      in
      Err.fail
        (`Resize_bicubic
           Shape_error.Resize_bicubic.
             { axis; in_extent; out_extent; aggregate; limit })
    else Err.return ()

  module Compute (S : sig
    type 'role index
    type t

    val index_const : int -> Semantics.delta index
    val of_index : Semantics.position index -> Semantics.delta index

    val index_add :
      Semantics.delta index -> Semantics.delta index -> Semantics.delta index

    val index_scale : int -> Semantics.delta index -> Semantics.delta index

    val index_floor_div_pos :
      Semantics.delta index -> Op_config.Pos.t -> Semantics.delta index

    val index_min :
      Semantics.delta index -> Semantics.delta index -> Semantics.delta index

    val index_max :
      Semantics.delta index -> Semantics.delta index -> Semantics.delta index

    val clamp_low : Semantics.delta index -> Semantics.position index
    val value_of_index : Semantics.delta index -> t
    val const : float -> t
    val add : t -> t -> t
    val sub : t -> t -> t
    val mul : t -> t -> t
    val div : t -> t -> t
  end) =
  struct
    type endpoints = {
      i0 : Semantics.position S.index;
      i1 : Semantics.position S.index;
      i2 : Semantics.position S.index;
      i3 : Semantics.position S.index;
      w0 : S.t;
      w1 : S.t;
      w2 : S.t;
      w3 : S.t;
    }

    (* ATen's own `cubic_convolution1`/`cubic_convolution2`
       (UpSampleKernel.h), verbatim -- the SAME two polynomials evaluated in
       the SAME order, so a fuzz-oracle comparison sees the identical
       expression tree, not merely an equal function. *)
    let cubic_convolution1 ~a x =
      (* ((a + 2) * x - (a + 3)) * x * x + 1 *)
      let t = S.sub (S.mul (S.add a (S.const 2.0)) x) (S.add a (S.const 3.0)) in
      S.add (S.mul (S.mul t x) x) (S.const 1.0)

    let cubic_convolution2 ~a x =
      (* ((a * x - 5a) * x + 8a) * x - 4a *)
      let t = S.sub (S.mul a x) (S.mul (S.const 5.0) a) in
      let t = S.add (S.mul t x) (S.mul (S.const 8.0) a) in
      S.sub (S.mul t x) (S.mul (S.const 4.0) a)

    (* [get_cubic_upsample_coefficients], verbatim: [a = -0.75], weights for
       taps at relative offsets [-1, 0, 1, 2] from the floor. *)
    let coefficients (t : S.t) =
      let a = S.const (-0.75) in
      let w0 = cubic_convolution2 ~a (S.add t (S.const 1.0)) in
      let w1 = cubic_convolution1 ~a t in
      let one_minus_t = S.sub (S.const 1.0) t in
      let w2 = cubic_convolution1 ~a one_minus_t in
      let w3 = cubic_convolution2 ~a (S.add one_minus_t (S.const 1.0)) in
      (w0, w1, w2, w3)

    (* [upsample_get_value_bounded]: an out-of-range tap clamps to the
       nearest edge element, independently of every other tap -- unlike
       [Bilinear_axis.endpoints]'s [i1], no tap here is provably in range by
       construction, since the floor itself may already be negative or past
       the last element (see the module doc). *)
    let clamp_tap ~(in_extent : int) (idx : Semantics.delta S.index) :
        Semantics.position S.index =
      S.clamp_low
        (S.index_min
           (S.index_const (in_extent - 1))
           (S.index_max idx (S.index_const 0)))

    (* [align_corners=true] with [out_extent = 1] is the one genuinely
       degenerate case, exactly as in [Bilinear_axis.endpoints]: ATen's own
       scale is 0, so the source is always index 0 and [t = 0] -- weights
       [(0, 1, 0, 0)] rather than evaluating the polynomials at all. *)
    let endpoints ~(align_corners : bool) ~(in_extent : Dim.extent Dim.t)
        ~(out_extent : Op_config.Pos.t) (out_axis : Semantics.position S.index)
        : endpoints =
      let in_e = (in_extent :> int) and out_e = (out_extent :> int) in
      if align_corners && out_e = 1 then
        let i = clamp_tap ~in_extent:in_e (S.index_const 0) in
        {
          i0 = i;
          i1 = i;
          i2 = i;
          i3 = i;
          w0 = S.const 0.0;
          w1 = S.const 1.0;
          w2 = S.const 0.0;
          w3 = S.const 0.0;
        }
      else
        let numerator, denom =
          if align_corners then
            (S.index_scale (in_e - 1) (S.of_index out_axis), out_e - 1)
          else
            (* [(2*dst+1)*in_extent - out_extent] -- UNCLAMPED, unlike
               [Bilinear_axis.endpoints]'s own numerator: see the module
               doc for why bicubic keeps a negative source coordinate
               instead of flooring it to 0. *)
            let two_dst_plus_1 =
              S.index_add
                (S.index_scale 2 (S.of_index out_axis))
                (S.index_const 1)
            in
            ( S.index_add
                (S.index_scale in_e two_dst_plus_1)
                (S.index_const (-out_e)),
              2 * out_e )
        in
        let d_pos = Op_config.Pos.of_int denom in
        (* The raw floor, NOT clamped -- [S.index_floor_div_pos] is a true
           floor over a possibly-negative numerator (rounds toward negative
           infinity), so this can legitimately be [-1] here. Every read
           coordinate below clamps independently; this value only feeds the
           exact remainder/fractional-part split. *)
        let input_x = S.index_floor_div_pos numerator d_pos in
        let remainder =
          S.index_add numerator
            (S.index_scale (-1) (S.index_scale denom input_x))
        in
        let t =
          S.div (S.value_of_index remainder) (S.const (float_of_int denom))
        in
        let w0, w1, w2, w3 = coefficients t in
        {
          i0 =
            clamp_tap ~in_extent:in_e (S.index_add input_x (S.index_const (-1)));
          i1 = clamp_tap ~in_extent:in_e input_x;
          i2 = clamp_tap ~in_extent:in_e (S.index_add input_x (S.index_const 1));
          i3 = clamp_tap ~in_extent:in_e (S.index_add input_x (S.index_const 2));
          w0;
          w1;
          w2;
          w3;
        }
  end
end

module Bicubic2d = struct
  (* Same field shape as [Bilinear2d.params] -- an H/W output size plus
     [align_corners], which the coordinate transform reads exactly as
     [Bilinear_axis.endpoints] does (see [Bicubic_axis.endpoints]). *)
  type params = {
    output_size : Op_config.Pos.t Op_config.Hw.t;
    align_corners : bool;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"upsample_bicubic2d_params"
      (fun align_corners output_size -> { output_size; align_corners })
    |> Jsont.Object.mem "align_corners" Jsont.bool ~enc:(fun p ->
        p.align_corners)
    |> Jsont.Object.mem "output_size" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.output_size)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "{output_size=%a;@ align_corners=%b}"
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.output_size p.align_corners

  type t = { params : params; x : Tensor_ref.t }

  let name = "Upsample_bicubic2d"

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
    Fmt.pf fmt "@[<hv 2>upsample_bicubic2d@ x=%a@ params=%a@]" pp_ref t.x
      pp_params t.params

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let* () =
      Bicubic_axis.check ~axis:Axis.H ~in_extent:(Vec6.get x_shape Axis.H)
        ~out_extent:p.output_size.h
    in
    let* () =
      Bicubic_axis.check ~axis:Axis.W ~in_extent:(Vec6.get x_shape Axis.W)
        ~out_extent:p.output_size.w
    in
    Err.return
      (Vec6.set
         (Vec6.set x_shape Axis.H (Dim.extent (p.output_size.h :> int)))
         Axis.W
         (Dim.extent (p.output_size.w :> int)))

  module Compute (S : Semantics.SEMANTICS) = struct
    module Bi = Bicubic_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let eh =
        Bi.endpoints ~align_corners:p.align_corners
          ~in_extent:(Vec6.get x_shape Axis.H) ~out_extent:p.output_size.h
          (Vec6.get out Axis.H)
      in
      let ew =
        Bi.endpoints ~align_corners:p.align_corners
          ~in_extent:(Vec6.get x_shape Axis.W) ~out_extent:p.output_size.w
          (Vec6.get out Axis.W)
      in
      let read h w = S.load x (out |> Vec6.set_h h |> Vec6.set_w w) in
      (* Separable: interpolate each of the four rows across [W], then blend
         those four row values down [H] with the same weights. *)
      let row h =
        let open Bi in
        S.add
          (S.add (S.mul ew.w0 (read h ew.i0)) (S.mul ew.w1 (read h ew.i1)))
          (S.add (S.mul ew.w2 (read h ew.i2)) (S.mul ew.w3 (read h ew.i3)))
      in
      let open Bi in
      S.add
        (S.add (S.mul eh.w0 (row eh.i0)) (S.mul eh.w1 (row eh.i1)))
        (S.add (S.mul eh.w2 (row eh.i2)) (S.mul eh.w3 (row eh.i3)))
  end
end
