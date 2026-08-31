(* 2D pooling (NHWC): [MaxPool2d] and [AvgPool2d]. Output[n,t,d,h,w,c] reduces
   over the kernel window at the SAME channel c — no channel mixing, unlike
   conv (no [in_channels], no weight). Dilation and ceil_mode aren't modeled
   (dilation=1, ceil_mode=false implicitly) — resnet18's pool is kernel=3,
   stride=2, pad=1, dilation=1, ceil_mode=false, and nothing else needs more
   yet.

   The semantic pool primitives clip the window to the input directly. For
   [MaxPool2d] this is required for correctness: padding must not contribute 0
   and beat a real, negative activation. Symbolic semantics retain max-pool
   windows as one expression node. See .ai/native_compute_design.md. *)

(* Pooling output shape, shared by both pools (their [params] are field-
   compatible): N/T/D/C pass through from [x_shape] unchanged (pooling never
   touches channels); only H/W shrink by [Window_axis.output_extent]. All
   extent-space, no [:> int] round-trips. See .ai/native_compute_design.md §2b. *)
let window_output_shape ~(ceil_mode : bool) ~(x_shape : Vec6.shape)
    ~(kernel : Dim.extent Dim.t Op_config.Hw.t)
    ~(stride : Op_config.Pos.t Op_config.Hw.t)
    ~(pad : Op_config.Nonneg.t Op_config.Hw.t) =
  let open Err.Syntax in
  let out_extent axis ~kernel ~stride ~pad =
    Window_axis.output_extent ~ceil_mode ~in_extent:(Vec6.get x_shape axis)
      ~kernel ~stride ~pad_before:pad ~pad_after:pad
      ~dilation:(Op_config.Pos.of_int 1)
  in
  let* h = out_extent Axis.H ~kernel:kernel.h ~stride:stride.h ~pad:pad.h in
  let+ w = out_extent Axis.W ~kernel:kernel.w ~stride:stride.w ~pad:pad.w in
  Vec6.set (Vec6.set x_shape Axis.H h) Axis.W w

(* Random-walk config space shared by the two pools (their [params] are field-
   compatible). Each op's own [Walk] includes this and adds its [params] builder.
   [cascade] enforces pooling's constraints (2*pad <= kernel; input large enough
   for a >=1 output, dilation 1). Within the native backend only — not shared
   with ATen. *)
module Window_cfg (L : Walk_core.Limits.S) = struct
  type cfg = {
    shape : Walk_core.Shape.t;
    kernel : Walk_core.Walk.hw;
    stride : Walk_core.Walk.hw;
    pad : Walk_core.Walk.hw;
  }

  let l = L.limits

  let initial =
    {
      shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 8; w = 8; c = 4 };
      kernel = { Walk_core.Walk.h = 2; w = 2 };
      stride = { Walk_core.Walk.h = 2; w = 2 };
      pad = { Walk_core.Walk.h = 0; w = 0 };
    }

  let cascade c =
    let kh = c.kernel.Walk_core.Walk.h and kw = c.kernel.Walk_core.Walk.w in
    let pad_h = Walk_core.Window_math.clamp_pad_pool ~pad:c.pad.h ~kernel:kh in
    let pad_w = Walk_core.Window_math.clamp_pad_pool ~pad:c.pad.w ~kernel:kw in
    let h =
      Walk_core.Window_math.grow_input ~in_size:c.shape.Walk_core.Shape.h
        ~pad:pad_h ~kernel:kh ~dilation:1
    in
    let w =
      Walk_core.Window_math.grow_input ~in_size:c.shape.Walk_core.Shape.w
        ~pad:pad_w ~kernel:kw ~dilation:1
    in
    {
      c with
      pad = { Walk_core.Walk.h = pad_h; w = pad_w };
      shape = { c.shape with Walk_core.Shape.h; w };
    }

  let shape (c : cfg) = Walk_bridge.vec6 c.shape

  let kernel (c : cfg) : Dim.extent Dim.t Op_config.Hw.t =
    {
      Op_config.Hw.h = Dim.extent c.kernel.Walk_core.Walk.h;
      w = Dim.extent c.kernel.w;
    }

  let stride (c : cfg) : Op_config.Pos.t Op_config.Hw.t =
    {
      Op_config.Hw.h = Op_config.Pos.of_int c.stride.Walk_core.Walk.h;
      w = Op_config.Pos.of_int c.stride.w;
    }

  let pad (c : cfg) : Op_config.Nonneg.t Op_config.Hw.t =
    {
      Op_config.Hw.h = Op_config.Nonneg.of_int c.pad.Walk_core.Walk.h;
      w = Op_config.Nonneg.of_int c.pad.w;
    }

  let axes =
    Walk_core.Walk.
      [
        shape_axis "input" l
          ~get:(fun c -> c.shape)
          ~set:(fun c s -> { c with shape = s });
        hw_axis "kernel" ~lo:1 ~hi:l.max_kernel (fun c v ->
            { c with kernel = v });
        hw_axis "stride" ~lo:1 ~hi:l.max_stride (fun c v ->
            { c with stride = v });
        hw_axis "pad" ~lo:0 ~hi:l.max_pad (fun c v -> { c with pad = v });
      ]

  let pp fmt (c : cfg) =
    Format.fprintf fmt "{shape=%a kernel=%dx%d stride=%dx%d pad=%dx%d}"
      Walk_core.Shape.pp c.shape c.kernel.Walk_core.Walk.h c.kernel.w
      c.stride.Walk_core.Walk.h c.stride.w c.pad.Walk_core.Walk.h c.pad.w
end

(* [Window_cfg] plus a [ceil_mode] axis, for the two max-pool walks (not
   [AvgPool2d]: its [params] has no such field, since [avg_pool2d.default]
   isn't bridged yet -- see the module doc). A window valid under floor
   division stays valid under ceiling division (ceil_mode only ever grows or
   holds the output extent -- see [Window_axis.output_extent]), so no
   [cascade] change is needed: [Window_cfg]'s existing floor-mode growth
   already guarantees >=1 output for either mode. *)
module Ceil_window_cfg (L : Walk_core.Limits.S) = struct
  module Win = Window_cfg (L)

  type cfg = { window : Win.cfg; ceil_mode : bool }

  let initial = { window = Win.initial; ceil_mode = false }
  let cascade c = { c with window = Win.cascade c.window }
  let shape (c : cfg) = Win.shape c.window
  let kernel (c : cfg) = Win.kernel c.window
  let stride (c : cfg) = Win.stride c.window
  let pad (c : cfg) = Win.pad c.window

  let axes =
    let lift (a : Win.cfg Walk_core.Walk.axis) : cfg Walk_core.Walk.axis =
      {
        Walk_core.Walk.name = a.name;
        mutate =
          (fun pcg c ->
            let w, pcg = a.mutate pcg c.window in
            ({ c with window = w }, pcg));
      }
    in
    Walk_core.Walk.field_axis "ceil_mode" [ true; false ] (fun c v ->
        { c with ceil_mode = v })
    :: List.map lift Win.axes

  let pp fmt (c : cfg) =
    Format.fprintf fmt "%a ceil_mode=%b" Win.pp c.window c.ceil_mode
end

(* [Ceil_window_cfg] plus a [count_include_pad] axis, for [AvgPool2d]'s own
   walk -- the only pooling op whose divisor depends on it ([MaxPool2d] has no
   divisor to vary). *)
module Ceil_count_window_cfg (L : Walk_core.Limits.S) = struct
  module Base = Ceil_window_cfg (L)

  type cfg = { window : Base.cfg; count_include_pad : bool }

  let initial = { window = Base.initial; count_include_pad = true }
  let cascade c = { c with window = Base.cascade c.window }
  let shape (c : cfg) = Base.shape c.window
  let kernel (c : cfg) = Base.kernel c.window
  let stride (c : cfg) = Base.stride c.window
  let pad (c : cfg) = Base.pad c.window

  let axes =
    let lift (a : Base.cfg Walk_core.Walk.axis) : cfg Walk_core.Walk.axis =
      {
        Walk_core.Walk.name = a.name;
        mutate =
          (fun pcg c ->
            let w, pcg = a.mutate pcg c.window in
            ({ c with window = w }, pcg));
      }
    in
    Walk_core.Walk.field_axis "count_include_pad" [ true; false ] (fun c v ->
        { c with count_include_pad = v })
    :: List.map lift Base.axes

  let pp fmt (c : cfg) =
    Format.fprintf fmt "%a count_include_pad=%b" Base.pp c.window
      c.count_include_pad
end

module MaxPool2d = struct
  (* Matches ATen's `max_pool2d` value-only overload. *)

  (* Same field shape as [Conv2d.params] minus [in_channels] (pooling never
     reduces over channels), plus [ceil_mode] (dilation stays unmodeled -- see
     the module doc). See .ai/native_op_config.md, which already names the
     kernel/stride/pad carry-over as expected. [ceil_mode] is omitted from the
     encoded JSON when [false] (the pre-existing default), so every past
     fixture without the field still decodes unchanged. *)
  type params = {
    ceil_mode : bool;
    kernel : Dim.extent Dim.t Op_config.Hw.t;
    stride : Op_config.Pos.t Op_config.Hw.t;
    pad : Op_config.Nonneg.t Op_config.Hw.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"max_pool2d_params"
      (fun ceil_mode kernel pad stride -> { ceil_mode; kernel; stride; pad })
    |> Jsont.Object.mem "ceil_mode" Jsont.bool ~dec_absent:false
         ~enc:(fun p -> p.ceil_mode)
         ~enc_omit:(fun b -> not b)
    |> Jsont.Object.mem "kernel" (Op_config.Hw.jsont Dim.extent_jsont)
         ~enc:(fun p -> p.kernel)
    |> Jsont.Object.mem "pad" (Op_config.Hw.jsont Op_config.Nonneg.jsont)
         ~enc:(fun p -> p.pad)
    |> Jsont.Object.mem "stride" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.stride)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{kernel=%a;@ stride=%a;@ pad=%a;@ ceil_mode=%b}@]"
      (Op_config.Hw.pp Dim.pp) p.kernel
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.stride
      (Op_config.Hw.pp Op_config.Nonneg.pp)
      p.pad p.ceil_mode

  module Walk (L : Walk_core.Limits.S) = struct
    module W = Ceil_window_cfg (L)
    include W

    let params (c : W.cfg) : params =
      {
        kernel = W.kernel c;
        stride = W.stride c;
        pad = W.pad c;
        ceil_mode = c.ceil_mode;
      }
  end

  type t = { params : params; x : Tensor_ref.t }

  let name = "Max_pool2d"

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
    Fmt.pf fmt "@[<hv 2>max_pool2d@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    window_output_shape ~ceil_mode:p.ceil_mode ~x_shape ~kernel:p.kernel
      ~stride:p.stride ~pad:p.pad

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [ceil_mode] never reaches here: it only changes the OUTPUT EXTENT
       (above), not which input pixels a given output position reads.
       [S.max_pool2d]'s window is already clipped to the real input extent
       regardless of the nominal kernel size, and ATen's own "last pooling
       starts inside the image" correction (folded into [output_shape]'s
       [ceil_mode] handling) is exactly what keeps that clipped window
       non-empty for every output position ceil_mode admits. *)
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      S.max_pool2d x ~x_shape ~kernel:p.kernel ~stride:p.stride ~pad:p.pad out
  end
end

module MaxPool2dWithIndices = struct
  (* ATen's `max_pool2d_with_indices` — the overload resnet18's graph actually
     calls. Two outputs: out0 is the pooled max (identical to [MaxPool2d]); out1
     is the argmax *indices* — for each output pixel, the flattened input-plane
     position [ih*in_W + iw] of the max within its window. The indices output is
     unused by inference (routed to a [Discard] sink by the bridge), but the
     engine materialises it to preserve the op's full ATen arity; a later pass
     prunes it. Ties resolve to the smallest flat index. *)
  type params = MaxPool2d.params

  let params_jsont = MaxPool2d.params_jsont
  let pp_params = MaxPool2d.pp_params

  module Walk (L : Walk_core.Limits.S) = struct
    module W = Ceil_window_cfg (L)
    include W

    let params (c : W.cfg) : params =
      {
        kernel = W.kernel c;
        stride = W.stride c;
        pad = W.pad c;
        ceil_mode = c.ceil_mode;
      }
  end

  type t = { params : params; x : Tensor_ref.t }

  let name = "Max_pool2d_with_indices"

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
    Fmt.pf fmt "@[<hv 2>max_pool2d_with_indices@ x=%a@ params=%a@]" pp_ref t.x
      pp_params t.params

  (* Both outputs (values, indices) share the pooled window shape. *)
  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    window_output_shape ~ceil_mode:p.ceil_mode ~x_shape ~kernel:p.kernel
      ~stride:p.stride ~pad:p.pad

  module Compute (S : Semantics.SEMANTICS) = struct
    let value_pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      S.max_pool2d x ~x_shape ~kernel:p.kernel ~stride:p.stride ~pad:p.pad out

    (* out1: the flat input-plane index [ih*in_W + iw] of the max. The semantic
       primitive resolves ties to the smallest flat index. *)
    let index_pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      S.max_pool2d_index x ~x_shape ~kernel:p.kernel ~stride:p.stride ~pad:p.pad
        out
  end
end

module AvgPool2d = struct
  (* Same [params] shape as [MaxPool2d] (kernel/stride/pad, no [in_channels]),
     plus [ceil_mode] and [count_include_pad] -- the 100-model sweep found real
     models needing both (see the module doc's [MaxPool2d] note).
     [divisor_override] stays unrepresented: no model this
     repository can download supplies a non-default value, so a present one is
     refused at the import boundary rather than approximated.

     With [count_include_pad=true] (ATen's default) the divisor is the window
     clipped to the PADDED extent, not the real input -- position-independent
     and equal to the full kernel area for every output position floor-mode
     admits, and only smaller for the trailing positions [ceil_mode] adds (see
     [Compute.pixel]). With [count_include_pad=false] the divisor is the
     window clipped to the real input, the same range [total] already sums
     over. *)
  type params = {
    ceil_mode : bool;
    count_include_pad : bool;
    kernel : Dim.extent Dim.t Op_config.Hw.t;
    stride : Op_config.Pos.t Op_config.Hw.t;
    pad : Op_config.Nonneg.t Op_config.Hw.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"avg_pool2d_params"
      (fun ceil_mode count_include_pad kernel pad stride ->
        { ceil_mode; count_include_pad; kernel; stride; pad })
    |> Jsont.Object.mem "ceil_mode" Jsont.bool ~dec_absent:false
         ~enc:(fun p -> p.ceil_mode)
         ~enc_omit:(fun b -> not b)
    |> Jsont.Object.mem "count_include_pad" Jsont.bool ~dec_absent:true
         ~enc:(fun p -> p.count_include_pad)
         ~enc_omit:(fun b -> b)
    |> Jsont.Object.mem "kernel" (Op_config.Hw.jsont Dim.extent_jsont)
         ~enc:(fun p -> p.kernel)
    |> Jsont.Object.mem "pad" (Op_config.Hw.jsont Op_config.Nonneg.jsont)
         ~enc:(fun p -> p.pad)
    |> Jsont.Object.mem "stride" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.stride)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt
      "@[<hv>{kernel=%a;@ stride=%a;@ pad=%a;@ ceil_mode=%b;@ \
       count_include_pad=%b}@]"
      (Op_config.Hw.pp Dim.pp) p.kernel
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.stride
      (Op_config.Hw.pp Op_config.Nonneg.pp)
      p.pad p.ceil_mode p.count_include_pad

  module Walk (L : Walk_core.Limits.S) = struct
    module W = Ceil_count_window_cfg (L)
    include W

    let params (c : W.cfg) : params =
      {
        kernel = W.kernel c;
        stride = W.stride c;
        pad = W.pad c;
        ceil_mode = c.window.ceil_mode;
        count_include_pad = c.count_include_pad;
      }
  end

  type t = { params : params; x : Tensor_ref.t }

  let name = "Avg_pool2d"

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
    Fmt.pf fmt "@[<hv 2>avg_pool2d@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    window_output_shape ~ceil_mode:p.ceil_mode ~x_shape ~kernel:p.kernel
      ~stride:p.stride ~pad:p.pad

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [count_include_pad=true]'s divisor, one axis: the window clipped to the
       PADDED extent rather than the real input -- ATen's own
       `min(kernel, in_extent + 2*pad - out*stride)`. [start] (= [out*stride -
       pad]) is never less than [-pad] since [out >= 0], so this never needs a
       lower clip the way the real-input window does; it only ever shrinks
       [kernel] for the trailing output positions [ceil_mode] admits beyond
       the floor-mode bound (see [Window_axis.output_extent]'s own
       [ceil_mode] correction, which is exactly what keeps this positive). *)
    let padded_extent ~(kernel : Dim.extent Dim.t) ~(stride : Op_config.Pos.t)
        ~(pad : Op_config.Nonneg.t) ~(in_extent : Dim.extent Dim.t)
        (out_pos : Semantics.position S.index) : S.t =
      let out = S.of_index out_pos in
      let extended =
        S.index_add (S.index_extent in_extent)
          (S.index_const (2 * (pad :> int)))
      in
      let shifted =
        S.index_add extended (S.index_scale (-(stride :> int)) out)
      in
      S.value_of_index (S.index_min (S.index_extent kernel) shifted)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let module Wa = Window_axis.Compute (S) in
      let wh =
        Wa.window ~kernel:p.kernel.h ~stride:p.stride.h ~pad_before:p.pad.h
          ~dilation:(Op_config.Pos.of_int 1)
          ~in_extent:(Vec6.get x_shape Axis.H) (Vec6.get out Axis.H)
      in
      let ww =
        Wa.window ~kernel:p.kernel.w ~stride:p.stride.w ~pad_before:p.pad.w
          ~dilation:(Op_config.Pos.of_int 1)
          ~in_extent:(Vec6.get x_shape Axis.W) (Vec6.get out Axis.W)
      in
      let total =
        S.sum ~lo:wh.lo ~hi:wh.hi (fun kh ->
            S.sum ~lo:ww.lo ~hi:ww.hi (fun kw ->
                S.load x
                  (out |> Vec6.set_h (wh.src kh) |> Vec6.set_w (ww.src kw))))
      in
      let divisor =
        if (not p.ceil_mode) && p.count_include_pad then
          (* Fast path: under floor-mode output extents, every window's
             padded-clip size equals the full kernel area exactly (see
             [padded_extent]'s doc) -- the graph-construction-time constant
             this always was before [ceil_mode] existed. *)
          S.const (float_of_int ((p.kernel.h :> int) * (p.kernel.w :> int)))
        else if p.count_include_pad then
          S.mul
            (padded_extent ~kernel:p.kernel.h ~stride:p.stride.h ~pad:p.pad.h
               ~in_extent:(Vec6.get x_shape Axis.H) (Vec6.get out Axis.H))
            (padded_extent ~kernel:p.kernel.w ~stride:p.stride.w ~pad:p.pad.w
               ~in_extent:(Vec6.get x_shape Axis.W) (Vec6.get out Axis.W))
        else
          (* [count_include_pad=false]: divide by the real (non-padding)
             window area -- the same clipped range [total] already sums
             over. *)
          let real_extent (w : Wa.window) =
            S.value_of_index
              (S.index_add w.hi (S.index_scale (-1) (S.of_index w.lo)))
          in
          S.mul (real_extent wh) (real_extent ww)
      in
      S.div total divisor
  end
end

(* ATen adaptive-average-pooling bins one axis at a time:
   [start = floor(o * I / O)], [stop = ceil((o + 1) * I / O)].  The shape
   check below proves [I * O < Kernel.Limits.Hard.extent] before either
   multiplication reaches [index_scale].  That is essential under js_of_ocaml,
   where [int] is 32 bits, and makes every intermediate scale safe too because
   [o < O]. *)
module Adaptive_axis = struct
  let check ~axis ~(input_extent : Dim.extent Dim.t)
      ~(output_size : Op_config.Pos.t) : (unit, Shape_error.t) Err.t =
    let limit = Kernel.Limits.Hard.extent in
    let input = Int64.of_int (input_extent :> int) in
    let output = Int64.of_int (output_size :> int) in
    (* The factors are positive.  Divide before multiply so even a malicious
       host-[int] extent cannot wrap [int64] before the rejection. *)
    if input >= limit || output > Int64.div (Int64.sub limit 1L) input then
      let aggregate =
        if input >= limit || output >= limit then limit
        else Int64.mul input output
      in
      Err.fail
        (`Adaptive_pool
           Shape_error.Adaptive_pool.
             { axis; input_extent; output_size; aggregate; limit })
    else Err.return ()

  module Compute (S : Semantics.SEMANTICS) = struct
    type bin = {
      lo : Semantics.position S.index;
      hi : Semantics.delta S.index;
      area : S.t;
    }

    let bin ~(input_extent : Dim.extent Dim.t) ~(output_size : Op_config.Pos.t)
        out_axis =
      let scale = (input_extent :> int) in
      let out = S.of_index out_axis in
      let start = S.index_floor_div_pos (S.index_scale scale out) output_size in
      let stop =
        S.index_ceil_div_pos
          (S.index_scale scale (S.index_add out (S.index_const 1)))
          output_size
      in
      let lo = S.assume_index start in
      let width = S.index_add stop (S.index_scale (-1) start) in
      { lo; hi = stop; area = S.value_of_index width }
  end
end

(* aten.adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor,
   Tensor) always returns both value and index tensors -- unlike max_pool2d,
   ATen has no separate value-only overload for the adaptive case.
   [AdaptiveMaxPool2d] is therefore a NATIVE-internal narrowing target, not
   something either importer ever builds directly: both importers always
   construct [AdaptiveMaxPool2dWithIndices] below (mirroring
   [MaxPool2dWithIndices]'s own ATen-facing shape), and
   [Drop_pool_indices] narrows it to this op once the index output is proved
   dead -- exactly [Max_pool2d]/[Max_pool2d_with_indices]'s relationship.
   Reuses [Adaptive_axis] for the H/W bin ranges, the reduction swapped from
   [AdaptiveAvgPool2d]'s [S.sum]/[S.div] to [S.max_reduce]: no new SEMANTICS
   primitive is needed ([max_reduce] already has the same lo/hi/f shape as
   [sum]). *)
module AdaptiveMaxPool2d = struct
  type params = { output_size : Op_config.Pos.t Op_config.Hw.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"adaptive_max_pool2d_params" (fun output_size ->
        { output_size })
    |> Jsont.Object.mem "output_size" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.output_size)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "{output_size=%a}"
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.output_size

  type t = { params : params; x : Tensor_ref.t }

  let name = "Adaptive_max_pool2d"

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
    Fmt.pf fmt "@[<hv 2>adaptive_max_pool2d@ x=%a@ params=%a@]" pp_ref t.x
      pp_params t.params

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let* () =
      Adaptive_axis.check ~axis:Axis.H ~input_extent:(Vec6.get x_shape Axis.H)
        ~output_size:p.output_size.h
    in
    let* () =
      Adaptive_axis.check ~axis:Axis.W ~input_extent:(Vec6.get x_shape Axis.W)
        ~output_size:p.output_size.w
    in
    Err.return
      (Vec6.set
         (Vec6.set x_shape Axis.H (Dim.extent (p.output_size.h :> int)))
         Axis.W
         (Dim.extent (p.output_size.w :> int)))

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let module A = Adaptive_axis.Compute (S) in
      let h =
        A.bin ~input_extent:(Vec6.get x_shape Axis.H)
          ~output_size:p.output_size.h (Vec6.get out Axis.H)
      in
      let w =
        A.bin ~input_extent:(Vec6.get x_shape Axis.W)
          ~output_size:p.output_size.w (Vec6.get out Axis.W)
      in
      S.max_reduce ~lo:h.lo ~hi:h.hi (fun ih ->
          S.max_reduce ~lo:w.lo ~hi:w.hi (fun iw ->
              S.load x (out |> Vec6.set_h ih |> Vec6.set_w iw)))
  end
end

(* ATen's own `adaptive_max_pool2d.default` shape: value + flat-index argmax --
   the adaptive counterpart of [MaxPool2dWithIndices]. Same [params] as
   [AdaptiveMaxPool2d], the same reuse [MaxPool2dWithIndices.params =
   MaxPool2d.params] already establishes for the fixed-window family. *)
module AdaptiveMaxPool2dWithIndices = struct
  type params = AdaptiveMaxPool2d.params

  let params_jsont = AdaptiveMaxPool2d.params_jsont
  let pp_params = AdaptiveMaxPool2d.pp_params

  type t = { params : params; x : Tensor_ref.t }

  let name = "Adaptive_max_pool2d_with_indices"

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
    Fmt.pf fmt "@[<hv 2>adaptive_max_pool2d_with_indices@ x=%a@ params=%a@]"
      pp_ref t.x pp_params t.params

  (* Both outputs share the pooled bin shape, exactly [MaxPool2dWithIndices]'s
     own comment. *)
  let output_shape = AdaptiveMaxPool2d.output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module V = AdaptiveMaxPool2d.Compute (S)

    let value_pixel = V.pixel

    (* out1: the flat input-plane index [ih*in_W+iw] of the max within the
       adaptive bin, ties resolved to the smallest flat index -- the same
       convention [MaxPool2dWithIndices]'s own comment states, verified
       against real `at::adaptive_max_pool2d` in the bridge dispatch oracle
       test. There is no generic "arg-reduce" SEMANTICS primitive, and adding
       one would go against the module doc's [max]/[min]/[relu] rule: derive
       instead. Two passes -- the first finds the max [m] (exactly
       [value_pixel]'s own reduce), the second finds the smallest flat index
       whose value equals [m], read via [select]: [lt v m] is true only when
       [v] is NOT the max (since [v <= m] always), so that branch takes a
       sentinel strictly above every valid flat index ([in_h*in_w] -- flat
       indices only ever range over [0, in_h*in_w)), and the equal-to-max
       branch takes the real flat value -- no equality primitive needed. The
       smallest matching candidate is [-(max_reduce of the negated
       candidates)], the same max-reduce-plus-negation idiom the module doc
       uses to derive a min-like quantity from [max_reduce]/[select] rather
       than adding a primitive.

       [in_h * in_w] is an ordinary [int] product with no explicit bound
       check, the same as [direct.ml]'s own [max_pool2d_index] computing
       [(ih * w) + iw]: [x_shape] reaching [Compute] already passed a numel
       check <= [Kernel.Limits.Hard.numel] (2^31) at construction, and H/W are
       two of six factors of that bounded product (every factor >= 1), so
       their product alone cannot overflow the 32-bit js_of_ocaml [int]
       either. *)
    let index_pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let module A = Adaptive_axis.Compute (S) in
      let h =
        A.bin ~input_extent:(Vec6.get x_shape Axis.H)
          ~output_size:p.output_size.h (Vec6.get out Axis.H)
      in
      let w =
        A.bin ~input_extent:(Vec6.get x_shape Axis.W)
          ~output_size:p.output_size.w (Vec6.get out Axis.W)
      in
      let in_h = (Vec6.get x_shape Axis.H :> int) in
      let in_w = (Vec6.get x_shape Axis.W :> int) in
      let read ih iw = S.load x (out |> Vec6.set_h ih |> Vec6.set_w iw) in
      let m =
        S.max_reduce ~lo:h.lo ~hi:h.hi (fun ih ->
            S.max_reduce ~lo:w.lo ~hi:w.hi (fun iw -> read ih iw))
      in
      let sentinel = S.const (float_of_int (in_h * in_w)) in
      let candidate ih iw =
        let v = read ih iw in
        let flat =
          S.value_of_index
            (S.index_add (S.index_scale in_w (S.of_index ih)) (S.of_index iw))
        in
        S.select (S.lt v m) sentinel flat
      in
      let neg t = S.sub (S.const 0.0) t in
      neg
        (S.max_reduce ~lo:h.lo ~hi:h.hi (fun ih ->
             S.max_reduce ~lo:w.lo ~hi:w.hi (fun iw -> neg (candidate ih iw))))
  end
end

module AdaptiveAvgPool2d = struct
  type params = { output_size : Op_config.Pos.t Op_config.Hw.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"adaptive_avg_pool2d_params" (fun output_size ->
        { output_size })
    |> Jsont.Object.mem "output_size" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.output_size)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "{output_size=%a}"
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.output_size

  type t = { params : params; x : Tensor_ref.t }

  let name = "Adaptive_avg_pool2d"

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
    Fmt.pf fmt "@[<hv 2>adaptive_avg_pool2d@ x=%a@ params=%a@]" pp_ref t.x
      pp_params t.params

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let* () =
      Adaptive_axis.check ~axis:Axis.H ~input_extent:(Vec6.get x_shape Axis.H)
        ~output_size:p.output_size.h
    in
    let* () =
      Adaptive_axis.check ~axis:Axis.W ~input_extent:(Vec6.get x_shape Axis.W)
        ~output_size:p.output_size.w
    in
    Err.return
      (Vec6.set
         (Vec6.set x_shape Axis.H (Dim.extent (p.output_size.h :> int)))
         Axis.W
         (Dim.extent (p.output_size.w :> int)))

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      let module A = Adaptive_axis.Compute (S) in
      let h =
        A.bin ~input_extent:(Vec6.get x_shape Axis.H)
          ~output_size:p.output_size.h (Vec6.get out Axis.H)
      in
      let w =
        A.bin ~input_extent:(Vec6.get x_shape Axis.W)
          ~output_size:p.output_size.w (Vec6.get out Axis.W)
      in
      let total =
        S.sum ~lo:h.lo ~hi:h.hi (fun ih ->
            S.sum ~lo:w.lo ~hi:w.hi (fun iw ->
                S.load x (out |> Vec6.set_h ih |> Vec6.set_w iw)))
      in
      S.div total (S.mul h.area w.area)
  end
end
