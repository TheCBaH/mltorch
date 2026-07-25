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
let window_output_shape ~(x_shape : Vec6.shape)
    ~(kernel : Dim.extent Dim.t Op_config.Hw.t)
    ~(stride : Op_config.Pos.t Op_config.Hw.t)
    ~(pad : Op_config.Nonneg.t Op_config.Hw.t) =
  let open Core.Syntax in
  let out_extent axis ~kernel ~stride ~pad =
    Window_axis.output_extent ~in_extent:(Vec6.get x_shape axis) ~kernel ~stride
      ~pad_before:pad ~pad_after:pad ~dilation:(Op_config.Pos.of_int 1)
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

module MaxPool2d = struct
  (* Matches ATen's `max_pool2d` value-only overload. *)

  (* Same field shape as [Conv2d.params] minus [in_channels] (pooling never
     reduces over channels). See .ai/native_op_config.md, which already
     names this as the expected carry-over. *)
  type params = {
    kernel : Dim.extent Dim.t Op_config.Hw.t;
    stride : Op_config.Pos.t Op_config.Hw.t;
    pad : Op_config.Nonneg.t Op_config.Hw.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"max_pool2d_params" (fun kernel pad stride ->
        { kernel; stride; pad })
    |> Jsont.Object.mem "kernel" (Op_config.Hw.jsont Dim.extent_jsont)
         ~enc:(fun p -> p.kernel)
    |> Jsont.Object.mem "pad" (Op_config.Hw.jsont Op_config.Nonneg.jsont)
         ~enc:(fun p -> p.pad)
    |> Jsont.Object.mem "stride" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.stride)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{kernel=%a;@ stride=%a;@ pad=%a}@]"
      (Op_config.Hw.pp Dim.pp) p.kernel
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.stride
      (Op_config.Hw.pp Op_config.Nonneg.pp)
      p.pad

  module Walk (L : Walk_core.Limits.S) = struct
    module W = Window_cfg (L)
    include W

    let params (c : W.cfg) : params =
      { kernel = W.kernel c; stride = W.stride c; pad = W.pad c }
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
    window_output_shape ~x_shape ~kernel:p.kernel ~stride:p.stride ~pad:p.pad

  module Compute (S : Semantics.SEMANTICS) = struct
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
    module W = Window_cfg (L)
    include W

    let params (c : W.cfg) : params =
      { kernel = W.kernel c; stride = W.stride c; pad = W.pad c }
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
    window_output_shape ~x_shape ~kernel:p.kernel ~stride:p.stride ~pad:p.pad

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
  (* Same [params] shape as [MaxPool2d] (kernel/stride/pad, no [in_channels]).
     Matches ATen's `avg_pool2d` defaults: `count_include_pad=true`,
     `divisor_override=None`, `ceil_mode=false` — i.e. every window divides by
     the FULL kernel area, even where part of it falls in the padding region.
     That divisor is therefore a position-independent constant, which is why
     this doesn't need the [Window_axis] window for the divisor — only for the
     sum, exactly as in [MaxPool2d]: clipping the reduction range means the
     padding region simply isn't summed (contributing the same 0 it would
     under a guarded read, but never via an out-of-bounds index), while the
     divisor still counts it. *)
  type params = {
    kernel : Dim.extent Dim.t Op_config.Hw.t;
    stride : Op_config.Pos.t Op_config.Hw.t;
    pad : Op_config.Nonneg.t Op_config.Hw.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"avg_pool2d_params" (fun kernel pad stride ->
        { kernel; stride; pad })
    |> Jsont.Object.mem "kernel" (Op_config.Hw.jsont Dim.extent_jsont)
         ~enc:(fun p -> p.kernel)
    |> Jsont.Object.mem "pad" (Op_config.Hw.jsont Op_config.Nonneg.jsont)
         ~enc:(fun p -> p.pad)
    |> Jsont.Object.mem "stride" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.stride)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{kernel=%a;@ stride=%a;@ pad=%a}@]"
      (Op_config.Hw.pp Dim.pp) p.kernel
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.stride
      (Op_config.Hw.pp Op_config.Nonneg.pp)
      p.pad

  module Walk (L : Walk_core.Limits.S) = struct
    module W = Window_cfg (L)
    include W

    let params (c : W.cfg) : params =
      { kernel = W.kernel c; stride = W.stride c; pad = W.pad c }
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
    window_output_shape ~x_shape ~kernel:p.kernel ~stride:p.stride ~pad:p.pad

  module Compute (S : Semantics.SEMANTICS) = struct
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
      let area = (p.kernel.h :> int) * (p.kernel.w :> int) in
      S.div total (S.const (float_of_int area))
  end
end
