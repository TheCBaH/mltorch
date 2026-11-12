(* The random-walk config spaces of the ops in [Pool], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

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

  let shape (c : cfg) = Walk_shape.vec6 c.shape

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
  open Pool.MaxPool2d

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
end

module MaxPool2dWithIndices = struct
  open Pool.MaxPool2dWithIndices

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
end

module AvgPool2d = struct
  open Pool.AvgPool2d

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
end
