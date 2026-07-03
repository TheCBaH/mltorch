(* 2D pooling (NHWC): [MaxPool2d] and [AvgPool2d]. Output[n,t,d,h,w,c] reduces
   over the kernel window at the SAME channel c — no channel mixing, unlike
   conv (no [in_channels], no weight). Dilation and ceil_mode aren't modeled
   (dilation=1, ceil_mode=false implicitly) — resnet18's pool is kernel=3,
   stride=2, pad=1, dilation=1, ceil_mode=false, and nothing else needs more
   yet.

   Both share [Window_axis] with conv: the kh/kw reduction bounds are clipped
   to exactly the kernel offsets whose source position lands inside the
   input, so the position [load] reads is always valid by construction. For
   [MaxPool2d] this isn't just convenient — it's required for correctness: a
   guarded-read-returns-0 fallback for the padding region (as a naive [load]
   would do) would be wrong for [max_reduce] over real, possibly-negative data,
   where 0 is not a safe stand-in for "no value here." See
   .ai/native_compute_design.md §4. *)

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

module MaxPool2d = struct
  (* Matches ATen's `max_pool2d` (the value output only; `max_pool2d_with_indices`
     — what resnet18's graph actually calls — also returns the argmax indices
     for backprop, not modeled here: this engine is inference-only, the
     indices output is unused in eval mode). *)

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
    module Wa = Window_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Axis.t -> Semantics.position S.index) =
      let wh =
        Wa.window ~kernel:p.kernel.h ~stride:p.stride.h ~pad_before:p.pad.h
          ~dilation:(Op_config.Pos.of_int 1)
          ~in_extent:(Vec6.get x_shape Axis.H) (out Axis.H)
      in
      let ww =
        Wa.window ~kernel:p.kernel.w ~stride:p.stride.w ~pad_before:p.pad.w
          ~dilation:(Op_config.Pos.of_int 1)
          ~in_extent:(Vec6.get x_shape Axis.W) (out Axis.W)
      in
      S.max_reduce ~lo:wh.lo ~hi:wh.hi (fun kh ->
          S.max_reduce ~lo:ww.lo ~hi:ww.hi (fun kw ->
              S.load x (fun a ->
                  match a with
                  | Axis.H -> wh.src kh
                  | Axis.W -> ww.src kw
                  | _ -> out a)))
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
    module Wa = Window_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Axis.t -> Semantics.position S.index) =
      let wh =
        Wa.window ~kernel:p.kernel.h ~stride:p.stride.h ~pad_before:p.pad.h
          ~dilation:(Op_config.Pos.of_int 1)
          ~in_extent:(Vec6.get x_shape Axis.H) (out Axis.H)
      in
      let ww =
        Wa.window ~kernel:p.kernel.w ~stride:p.stride.w ~pad_before:p.pad.w
          ~dilation:(Op_config.Pos.of_int 1)
          ~in_extent:(Vec6.get x_shape Axis.W) (out Axis.W)
      in
      let total =
        S.sum ~lo:wh.lo ~hi:wh.hi (fun kh ->
            S.sum ~lo:ww.lo ~hi:ww.hi (fun kw ->
                S.load x (fun a ->
                    match a with
                    | Axis.H -> wh.src kh
                    | Axis.W -> ww.src kw
                    | _ -> out a)))
      in
      let area = (p.kernel.h :> int) * (p.kernel.w :> int) in
      S.div total (S.const (float_of_int area))
  end
end
