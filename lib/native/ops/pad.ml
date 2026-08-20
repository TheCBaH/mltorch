(* Boundary synthesis — the native form of `aten.pad.default`. One tensor
   operand, a per-axis (before, after) amount, and a mode saying what an output
   coordinate outside the source reads.

   The pad amounts are SIGNED, because ATen's are: a negative entry crops. That
   makes the shape rule the only place emptiness can be judged, and it is
   (`Shape_error.Pad`, fault [`Empty]) -- the engine has no zero extent, so a
   crop that consumes an axis has no Native result at all.

   The payload names FRAME axes and stores each amount against the axis it pads.
   PyTorch's serialized list is a flat, REVERSED, innermost-first sequence of
   pairs; un-reversing it is the importer's job, and the payload's type is what
   guarantees the compute never has to re-derive it. See
   .ai/native_aten_bridge_layout.md. *)

module Pad = struct
  (* What an output coordinate outside the source reads. A closed variant, not a
     string: ATen's `mode` argument has four spellings and this engine
     implements two, so the ones it does not implement must be unrepresentable
     here rather than rejected downstream. *)
  type mode =
    | Constant of float  (** every out-of-source coordinate reads this value *)
    | Reflect
        (** mirror about the boundary element, which is NOT repeated:
            [... c b | a b c d | c b ...]. Requires each positive pad below the
            extent it mirrors, which [output_shape] checks. *)

  type entry = { before : int; after : int }

  (* One entry per PADDED axis, in [Axis.all] order. Axes absent from the list
     are unpadded; an entry of [{before = 0; after = 0}] is equivalent to
     absence and is not normalized away, because the importer's list length is
     evidence a reader may want in a printed graph. *)
  type params = { pads : (Axis.t * entry) list; mode : mode }

  let entry_jsont : (Axis.t * entry) Jsont.t =
    Jsont.Object.map ~kind:"pad_entry" (fun axis before after ->
        (axis, { before; after }))
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:fst
    |> Jsont.Object.mem "before" Jsont.int ~enc:(fun (_, e) -> e.before)
    |> Jsont.Object.mem "after" Jsont.int ~enc:(fun (_, e) -> e.after)
    |> Jsont.Object.finish

  (* [{"constant": <f32>}] / [{"reflect": null}] through the same single-key
     union [Graph_ir] uses for ops, so a mode this engine does not implement
     cannot decode into something that merely misbehaves. The fill goes through
     [f32_jsont], not [Jsont.number]: an unnarrowed float64 literal would
     round-trip to a value the payload cannot store. *)
  let mode_jsont : mode Jsont.t =
    Jsont.map ~kind:"pad_mode"
      ~dec:
        (Json_util.union ~kind:"pad_mode"
           [
             ( "constant",
               fun v -> Constant (Json_util.dec Json_util.f32_jsont v) );
             ("reflect", fun _ -> Reflect);
           ])
      ~enc:(function
        | Constant v ->
            Json_util.single ~case:"constant"
              (Json_util.enc Json_util.f32_jsont v)
        | Reflect -> Json_util.single ~case:"reflect" Json_util.jnull)
      Jsont.json

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"pad_params" (fun pads mode -> { pads; mode })
    |> Jsont.Object.mem "pads" (Jsont.list entry_jsont) ~enc:(fun p -> p.pads)
    |> Jsont.Object.mem "mode" mode_jsont ~enc:(fun p -> p.mode)
    |> Jsont.Object.finish

  let pp_mode fmt = function
    | Constant v -> Fmt.pf fmt "constant(%a)" Fmt.float v
    | Reflect -> Fmt.string fmt "reflect"

  let pp_entry fmt (axis, e) =
    Fmt.pf fmt "@[<h>%a:%d,%d@]" Axis.pp axis e.before e.after

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{pads=%a mode=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_entry))
      p.pads pp_mode p.mode

  type t = { params : params; x : Tensor_ref.t }

  let name = "Pad"

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
    Fmt.pf fmt "@[<hv 2>pad@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  (* ---- the ATen serialization convention ---------------------------------

     `aten.pad.default` serializes its amounts as a flat list of pairs,
     INNERMOST DIMENSION FIRST: [left, right, top, bottom, ...]. Pair k is
     therefore ATen dim [rank - 1 - k], and the frame axis is that dim through
     the right-aligned embedding. Un-reversing it here, once, is what lets the
     payload above name frame axes and the compute never re-derive an ordering.

     Both importers call this: [Op_bridge] with a rank read off the live tensor,
     [Native_interp] with one read off the serialized metadata. Two copies of a
     reversal is one edit away from two answers. *)
  module Bad_pad_list = struct
    type fault =
      [ `Odd_length
      | `Too_many_pairs
      | `Unsupported_mode
      | `Value_with_non_constant_mode
      | `Unsupported_rank ]

    type t = { pad : int list; rank : int; mode : string; fault : fault }

    let pp ppf { pad; rank; mode; fault } =
      let pairs = List.length pad / 2 in
      match fault with
      | `Odd_length ->
          Fmt.pf ppf
            "pad list has %d entries; a pair per padded dimension is required"
            (List.length pad)
      | `Too_many_pairs ->
          Fmt.pf ppf "pad list covers %d dimensions of a rank-%d input" pairs
            rank
      | `Unsupported_mode ->
          Fmt.pf ppf
            "pad mode %S is outside the Native domain (constant and reflect \
             are supported)"
            mode
      | `Value_with_non_constant_mode ->
          Fmt.pf ppf "pad mode %S takes no non-zero value argument" mode
      | `Unsupported_rank ->
          Fmt.pf ppf
            "pad mode %S over %d dimension%s is not defined for a rank-%d input"
            mode pairs
            (if pairs = 1 then "" else "s")
            rank
  end

  (* ATen's own rules, in ATen's own order (native/PadNd.cpp's
     [_pad_enum_symint]), so this accepts exactly the nodes ATen accepts and
     refuses the rest with a row of its own rather than by diverging later:

     1. the list length is even, and covers at most [rank] dimensions;
     2. constant mode takes [value], absent meaning 0.0;
     3. every other mode refuses a non-zero [value] -- ATen's check is
        [!value.has_value() || *value == 0], so an explicit 0.0 is accepted;
     4. reflect is defined only for (2 pads, rank 2 or 3), (4, rank 3 or 4) and
        (6, rank 4 or 5) -- the per-rank kernels ATen dispatches to. Accepting a
        combination ATen rejects would turn a refusal into a walk MISMATCH,
        which reads as a wrong answer rather than as a boundary.
     5. replicate and circular are outside the Native domain entirely. *)
  let params_of_aten ~rank ~(pad : int list) ~(mode : string)
      ~(value : float option) =
    let n = List.length pad in
    let fail fault =
      Err.fail (`Bad_pad_list { Bad_pad_list.pad; rank; mode; fault })
    in
    if n mod 2 <> 0 then fail `Odd_length
    else if n > 2 * rank then fail `Too_many_pairs
    else
      let pairs = n / 2 in
      let entries =
        List.init pairs (fun k ->
            let dim = rank - 1 - k in
            let before = List.nth pad (2 * k)
            and after = List.nth pad ((2 * k) + 1) in
            (Aten_shape.axis_of_dim ~rank dim, { before; after }))
      in
      (* [Axis.all] order, so a printed payload reads outermost-first however the
         serialized list was ordered. *)
      let pads =
        List.filter_map
          (fun axis ->
            Option.map (fun e -> (axis, e)) (List.assoc_opt axis entries))
          Axis.all
      in
      match mode with
      | "constant" ->
          Err.return { pads; mode = Constant (Option.value value ~default:0.) }
      | "reflect" ->
          if Option.fold ~none:false ~some:(fun v -> v <> 0.) value then
            fail `Value_with_non_constant_mode
          else if
            not
              ((pairs = 1 && (rank = 2 || rank = 3))
              || (pairs = 2 && (rank = 3 || rank = 4))
              || (pairs = 3 && (rank = 4 || rank = 5)))
          then fail `Unsupported_rank
          else Err.return { pads; mode = Reflect }
      | _ -> fail `Unsupported_mode

  (* The per-axis amount, or a zero entry for an unpadded axis. A later
     duplicate cannot shadow an earlier one silently: [output_shape] rejects a
     repeated axis before this is ever consulted. *)
  let entry_of (p : params) axis =
    Option.value (List.assoc_opt axis p.pads) ~default:{ before = 0; after = 0 }

  (* Each factor bounded BEFORE the arithmetic, then combined in [int64] -- the
     shape [Window_axis.factor] establishes, and for the same reason: the pad
     amounts are model data bounded only in sign (and not even that: they may be
     negative), so [in + before + after] is an aggregate of individually-plausible
     numbers that can leave the range on a 32-bit backend. See
     .ai/js_backends_design.md. *)
  let bound ~what (n : int) : (int64, Shape_error.t) Err.t =
    let limit = Kernel.Limits.Hard.extent in
    let n64 = Int64.of_int n in
    if Int64.compare (Int64.abs n64) limit >= 0 then
      Err.fail
        (`Window_over_limit
           Shape_error.Window_over_limit.{ what; value = n64; limit })
    else Err.return n64

  let axis_extent ~(x_shape : Vec6.shape) (p : params) axis =
    let open Err.Syntax in
    let e = Vec6.get x_shape axis in
    let { before; after } = entry_of p axis in
    let* i = bound ~what:`Input_extent (e :> int) in
    let* b = bound ~what:`Padding before in
    let* a = bound ~what:`Padding after in
    let out = Int64.add (Int64.add i b) a in
    let fail fault =
      Err.fail
        (`Pad
           Shape_error.Pad.
             { axis; in_extent = e; before = b; after = a; out; fault })
    in
    (* Reflect mirrors about the boundary element, so a pad that reaches the far
       side has nothing to reflect off. Only the POSITIVE side can violate it: a
       negative pad shrinks the read range, so the mirror never fires there.
       PyTorch refuses the same configurations. *)
    let* () =
      match p.mode with
      | Reflect when Int64.compare b i >= 0 || Int64.compare a i >= 0 ->
          fail `Reflect_width
      | Constant _ | Reflect -> Err.return ()
    in
    if Int64.compare out 1L < 0 then fail `Empty
    else if Int64.compare out Kernel.Limits.Hard.extent >= 0 then
      Err.fail
        (`Window_over_limit
           Shape_error.Window_over_limit.
             {
               what = `Output_extent;
               value = out;
               limit = Kernel.Limits.Hard.extent;
             })
    else Err.return (Dim.extent (Int64.to_int out))

  (* Every axis is adjusted, not only the listed ones: an unlisted axis has a
     zero entry and keeps its extent, which still routes through the bound above
     so an oversized INPUT is refused here rather than at staging. *)
  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    (* Which of two entries for one axis was meant is unknowable, so a duplicate
       is refused rather than resolved last-one-wins. Neither importer can
       produce one (both build from a rank-indexed list), so this guards
       [Graph_builder] and JSON decoding. *)
    let* () =
      let axes = List.map fst p.pads in
      match
        List.find_opt
          (fun a -> List.length (List.filter (Axis.equal a) axes) > 1)
          Axis.all
      with
      | None -> Err.return ()
      | Some axis ->
          let { before; after } = List.assoc axis p.pads in
          Err.fail
            (`Pad
               Shape_error.Pad.
                 {
                   axis;
                   in_extent = Vec6.get x_shape axis;
                   before = Int64.of_int before;
                   after = Int64.of_int after;
                   out = 0L;
                   fault = `Duplicate_axis;
                 })
    in
    Err.List.fold_left
      (fun s axis ->
        let+ e = axis_extent ~x_shape p axis in
        Vec6.set s axis e)
      x_shape Axis.all

  (* This op's random-walk config space: correlated by construction. The pad
     entries are derived from the drawn shape rather than mutated beside it, so
     no step can produce a reflect pad wider than its axis or a crop that empties
     one -- an invalid config would be reported as a walk mismatch rather than as
     the rejection it is. Rank/dim and mode coverage belong to the ATEN recipe,
     where the shape is a rank-bearing list. *)
  module Walk (L : Walk_core.Limits.S) = struct
    (* ONE correlated axis of whole valid configurations, not a pattern axis
       beside a mode axis. Two independent axes would spend most steps
       re-drawing the value they already had (the walk picks uniformly, current
       value included), so reflect would appear only by luck; and any pair the
       op rejects would be reported as a walk mismatch rather than as the
       refusal it is. *)
    type pattern =
      | Pad_hw  (** symmetric spatial padding, the common convolution shape *)
      | Pad_asymmetric_w  (** before <> after, so a swap is observable *)
      | Crop_h  (** both entries negative *)
      | Mixed_hw  (** pad one axis while cropping another *)
      | Reflect_hw  (** the mirror, on both spatial axes *)
      | Reflect_asymmetric_w  (** the mirror with unequal sides *)
      | Reflect_crop_h  (** reflect mode with a negative entry *)

    type cfg = { shape : Walk_core.Shape.t; pattern : pattern }

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 6; w = 6; c = 3 };
        pattern = Pad_hw;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    (* Every amount is a fraction of the extent it applies to, which is what
       makes the config space valid at EVERY drawn shape: reflect needs each
       positive pad strictly below the extent, and a crop needs to leave at
       least one element.

       Both bounds have to be applied, and the [max 1] alone was not enough.
       The shape axis's floor is 1, and at extent 1 [max 1 (1 / 3)] is 1 -- a
       reflect of 1 on an extent-1 axis, which is exactly what
       [Pad.output_shape] refuses, so the walk raised out of [Err.or_raise]
       instead of stepping. It reproduced at seeds 9, 21, 26, 33 and 39 (12
       steps); the sweep never saw it only because its own seed happens not to
       draw a 1 in 5 steps, which is what made it a latent defect rather than a
       red test. Clamping to [n - 1] gives 0 there, and changes nothing at any
       extent >= 2. *)
    let params (c : cfg) : params =
      let h = c.shape.Walk_core.Shape.h and w = c.shape.Walk_core.Shape.w in
      let third n = Stdlib.min (Stdlib.max 1 (n / 3)) (n - 1) in
      let pads =
        match c.pattern with
        | Pad_hw | Reflect_hw ->
            [
              (Axis.H, { before = third h; after = third h });
              (Axis.W, { before = third w; after = third w });
            ]
        | Pad_asymmetric_w | Reflect_asymmetric_w ->
            [ (Axis.W, { before = third w; after = 0 }) ]
        | Crop_h -> [ (Axis.H, { before = -third h; after = 0 }) ]
        | Reflect_crop_h ->
            [
              (Axis.H, { before = -third h; after = 0 });
              (Axis.W, { before = third w; after = 0 });
            ]
        | Mixed_hw ->
            [
              (Axis.H, { before = -third h; after = 0 });
              (Axis.W, { before = 0; after = third w });
            ]
      in
      let mode =
        match c.pattern with
        | Reflect_hw | Reflect_asymmetric_w | Reflect_crop_h -> Reflect
        | Pad_hw | Pad_asymmetric_w | Crop_h | Mixed_hw -> Constant 0.5
      in
      { pads; mode }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "pattern"
            [
              Pad_hw;
              Pad_asymmetric_w;
              Crop_h;
              Mixed_hw;
              Reflect_hw;
              Reflect_asymmetric_w;
              Reflect_crop_h;
            ] (fun r v -> { r with pattern = v });
        ]

    let pp_pattern = function
      | Pad_hw -> "pad_hw"
      | Pad_asymmetric_w -> "pad_asym_w"
      | Crop_h -> "crop_h"
      | Mixed_hw -> "mixed_hw"
      | Reflect_hw -> "reflect_hw"
      | Reflect_asymmetric_w -> "reflect_asym_w"
      | Reflect_crop_h -> "reflect_crop_h"

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a pattern=%s}" Walk_core.Shape.pp c.shape
        (pp_pattern c.pattern)
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [|x| = max x (-x)]. *)
    let abs x = S.index_max x (S.index_scale (-1) x)

    (* Clamp a source coordinate into [0, extent - 1]. Sound by construction and
       NOT [assume_index]: [index_min] bounds it above and [clamp_low] below, so
       the result is a position whichever way the arithmetic went. That is what
       keeps [S.load] in bounds even in the pad region, which matters because
       [S.select] evaluates BOTH arms -- there is no lazy branch to hide an
       out-of-range read behind. *)
    let clamp_into ~(extent : Dim.extent Dim.t) j =
      S.clamp_low (S.index_min (S.index_const ((extent :> int) - 1)) j)

    (* Mirror [j] into [0, n-1] about the boundary elements, which are not
       repeated: [j' = |j|] folds the low side, [n-1 - |n-1 - j'|] folds the
       high side. One fold each is enough because [output_shape] has already
       refused a pad at least as wide as the extent. *)
    let reflect_into ~(extent : Dim.extent Dim.t) j =
      let n1 = S.index_const ((extent :> int) - 1) in
      let low = abs j in
      S.index_add n1
        (S.index_scale (-1) (abs (S.index_add n1 (S.index_scale (-1) low))))

    (* The source coordinate for one axis, plus how far it had to be moved to
       land inside. A displacement of zero means the output coordinate was
       already inside the source -- which is the region test, obtained without an
       index COMPARISON (Expr.Bool has none) and therefore without a new AST
       node. *)
    let source_axis (p : params) ~(x_shape : Vec6.shape) axis out_i =
      let extent = Vec6.get x_shape axis in
      let { before; after } = entry_of p axis in
      (* An unpadded axis maps to itself: the output extent equals the input's,
         so every clamp and every mirror below is the identity on it. Taking the
         short circuit is not an optimization of a hot path but of the SYMBOLIC
         program -- without it every graph stages the full mirror expression on
         all six axes, four of which are structurally extent 1, and the staged
         term for a two-axis pad becomes unreadable. *)
      if before = 0 && after = 0 then (out_i, S.index_const 0)
      else
        let j = S.index_add (S.of_index out_i) (S.index_const (-before)) in
        let src =
          match p.mode with
          | Reflect -> clamp_into ~extent (reflect_into ~extent j)
          | Constant _ -> clamp_into ~extent j
        in
        (src, S.index_add (S.of_index src) (S.index_scale (-1) j))

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let coord, displacement =
        List.fold_left
          (fun (coord, disp) axis ->
            let { before; after } = entry_of p axis in
            if before = 0 && after = 0 then (coord, disp)
            else
              let src, d = source_axis p ~x_shape axis (Vec6.get out axis) in
              (Vec6.set coord axis src, S.index_add disp (abs d)))
          (out, S.index_const 0)
          Axis.all
      in
      let inside = S.load x coord in
      match p.mode with
      (* Reflect reads a real element at every output coordinate, so there is
         nothing to select between -- and folding it through [select] anyway
         would stage a dead constant into every symbolic program. *)
      | Reflect -> inside
      | Constant v ->
          (* The sum of |displacement| over all axes is zero exactly when every
             axis was already inside, so ONE [index_eq] covers all six. A signed
             sum would not: two axes displaced in opposite directions cancel. *)
          S.select
            (S.index_eq displacement (S.index_const 0))
            inside (S.const v)
  end
end
