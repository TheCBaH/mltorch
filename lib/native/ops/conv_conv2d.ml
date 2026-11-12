(* 2D convolution (NHWC). Output[n,t,d,h,w,oc] reduces over input channels and the
   kernel window; the input position is the affine index [out*stride + k - pad],
   built through the SEMANTICS index ops so it specialises to either concrete ints
   (Direct) or index expressions (Symbolic). The kh/kw reduction bounds are
   clipped (via [Index_max]/[Index_min]) to exactly the kernel offsets whose source
   position lands inside the input — so that position is always a valid index,
   never a generate-then-guard read of the padding region (see
   the `.ai/` native compute design §4 "Open issue"). Weight is laid out
   [Cout,1,1,Kh,Kw,Cin_per_group]; bias is [1,1,1,1,1,Cout] (broadcast —
   extent-1 axes still go through [load]'s ordinary broadcast). See
   the `.ai/` native compute design §2. *)

let or_invalid_arg = function
  | Ok x -> x
  | Error e ->
      invalid_arg (Format.asprintf "%a" Shape_error.pp (Err.Error.kind e))

module Conv2d = struct
  (* [params]/[output_shape] are outside [Compute] so Direct/Symbolic share one
     [params] type. Field types per the `.ai/` native op config doc. *)
  type axis_window = {
    kernel : Dim.extent Dim.t;
    stride : Op_config.Pos.t;
    pad_before : Op_config.Nonneg.t;
    pad_after : Op_config.Nonneg.t;
    dilation : Op_config.Pos.t;
  }

  type params = {
    h : axis_window;
    w : axis_window;
    in_channels : Dim.extent Dim.t;
    groups : Op_config.Pos.t;
  }

  let axis_window_jsont : axis_window Jsont.t =
    Jsont.Object.map ~kind:"conv2d_axis_window"
      (fun kernel stride pad_before pad_after dilation ->
        { kernel; stride; pad_before; pad_after; dilation })
    |> Jsont.Object.mem "kernel" Dim.extent_jsont ~enc:(fun w -> w.kernel)
    |> Jsont.Object.mem "stride" Op_config.Pos.jsont ~enc:(fun w -> w.stride)
    |> Jsont.Object.mem "pad_before" Op_config.Nonneg.jsont ~enc:(fun w ->
        w.pad_before)
    |> Jsont.Object.mem "pad_after" Op_config.Nonneg.jsont ~enc:(fun w ->
        w.pad_after)
    |> Jsont.Object.mem "dilation" Op_config.Pos.jsont ~enc:(fun w ->
        w.dilation)
    |> Jsont.Object.finish

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"conv2d_params" (fun h w in_channels groups ->
        { h; w; in_channels; groups })
    |> Jsont.Object.mem "h" axis_window_jsont ~enc:(fun p -> p.h)
    |> Jsont.Object.mem "w" axis_window_jsont ~enc:(fun p -> p.w)
    |> Jsont.Object.mem "in_channels" Dim.extent_jsont ~enc:(fun p ->
        p.in_channels)
    |> Jsont.Object.mem "groups" Op_config.Pos.jsont ~enc:(fun p -> p.groups)
    |> Jsont.Object.finish

  let pp_axis_window fmt (w : axis_window) =
    Fmt.pf fmt
      "@[<hv>{kernel=%a;@ stride=%a;@ pad_before=%a;@ pad_after=%a;@ \
       dilation=%a}@]"
      Dim.pp w.kernel Op_config.Pos.pp w.stride Op_config.Nonneg.pp w.pad_before
      Op_config.Nonneg.pp w.pad_after Op_config.Pos.pp w.dilation

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{h=%a;@ w=%a;@ in_channels=%a;@ groups=%a}@]"
      pp_axis_window p.h pp_axis_window p.w Dim.pp p.in_channels
      Op_config.Pos.pp p.groups

  (* The single-pixel, no-op window: kernel 1, stride 1, no padding, dilation 1
     -- an axis this small always keeps its extent unchanged. Pins [Conv2d]'s H
     axis for [Conv1d] (conv_conv1d.ml) and, at Native4D's [Linear] ->
     [Conv2D4] legalization, both axes for a 1x1 convolution weight
     (lower_engine.ml's own former private copy of this same record). *)
  let unit_window : axis_window =
    {
      kernel = Dim.extent 1;
      stride = Op_config.Pos.of_int 1;
      pad_before = Op_config.Nonneg.of_int 0;
      pad_after = Op_config.Nonneg.of_int 0;
      dilation = Op_config.Pos.of_int 1;
    }

  (* The op payload: its params plus its operand edges. Carrying the operands here
     (rather than in [Graph_ir]'s variant) lets the op own its own serialisation,
     dataflow accessors and pretty-printer. *)
  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t;
    bias : Tensor_ref.t option;
  }

  let name = "Conv2d"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          x = get "x" Tensor_ref.jsont;
          weight = get "weight" Tensor_ref.jsont;
          bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt_bias =
          match t.bias with None -> [] | Some r -> [ ("bias", ref_ r) ]
        in
        Json_util.jobj
          (opt_bias
          @ [
              ("params", Json_util.enc params_jsont t.params);
              ("weight", ref_ t.weight);
              ("x", ref_ t.x);
            ]))
      Jsont.json

  let operands (t : t) = [ t.x; t.weight ] @ Option.to_list t.bias

  let map_operands f (t : t) =
    { t with x = f t.x; weight = f t.weight; bias = Option.map f t.bias }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>conv2d@ x=%a@ weight=%a@ bias=%a@ params=%a@]" pp_ref
      t.x pp_ref t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  let validate_channels ~(weight_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let in_channels = p.in_channels in
    let groups = p.groups in
    let out_channels = Vec6.get weight_shape Axis.N in
    let weight_in_per_group = Vec6.get weight_shape Axis.C in
    let* in_per_group =
      match Dim_arith.Extent.div_exact ~by:groups in_channels with
      | Some e -> Err.return e
      | None ->
          Err.fail
            (`Convolution
               (Shape_error.Convolution.In_channels_not_divisible_by_groups
                  Shape_error.Convolution.{ channels = in_channels; groups }))
    in
    let* out_per_group =
      match Dim_arith.Extent.div_exact ~by:groups out_channels with
      | Some e -> Err.return e
      | None ->
          Err.fail
            (`Convolution
               (Shape_error.Convolution.Out_channels_not_divisible_by_groups
                  Shape_error.Convolution.{ channels = out_channels; groups }))
    in
    let* () =
      if not (Dim.equal weight_in_per_group in_per_group) then
        Err.fail
          (`Convolution
             (Shape_error.Convolution.Weight_channels_mismatch
                Shape_error.Convolution.
                  { weight_in_per_group; expected_in_per_group = in_per_group }))
      else Err.return ()
    in
    let* () =
      if not (Dim.equal (Vec6.get weight_shape Axis.H) p.h.kernel) then
        Err.fail
          (`Convolution
             (Shape_error.Convolution.Weight_kernel_mismatch
                Shape_error.Convolution.
                  {
                    axis = Axis.H;
                    weight_extent = Vec6.get weight_shape Axis.H;
                    kernel_extent = p.h.kernel;
                  }))
      else Err.return ()
    in
    let* () =
      if not (Dim.equal (Vec6.get weight_shape Axis.W) p.w.kernel) then
        Err.fail
          (`Convolution
             (Shape_error.Convolution.Weight_kernel_mismatch
                Shape_error.Convolution.
                  {
                    axis = Axis.W;
                    weight_extent = Vec6.get weight_shape Axis.W;
                    kernel_extent = p.w.kernel;
                  }))
      else Err.return ()
    in
    Err.return (in_per_group, out_per_group)

  (* N/T/D pass through; H/W shrink via [Window_axis.output_extent]; C = Cout from
     weight_shape. *)
  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    let open Err.Syntax in
    let* _in_per_group, _out_per_group = validate_channels ~weight_shape p in
    let* () =
      if not (Dim.equal (Vec6.get x_shape Axis.C) p.in_channels) then
        Err.fail
          (`Convolution
             (Shape_error.Convolution.Input_channels_mismatch
                Shape_error.Convolution.
                  {
                    input_channels = Vec6.get x_shape Axis.C;
                    expected_in_channels = p.in_channels;
                  }))
      else Err.return ()
    in
    let out_extent axis (w : axis_window) =
      Window_axis.output_extent ~ceil_mode:false
        ~in_extent:(Vec6.get x_shape axis) ~kernel:w.kernel ~stride:w.stride
        ~pad_before:w.pad_before ~pad_after:w.pad_after ~dilation:w.dilation
    in
    (* Start from the input shape (N/T/D pass through unchanged) and replace only
       the axes the op resizes — all in extent-space, no [:> int] round-trips. *)
    let* h = out_extent Axis.H p.h in
    let+ w = out_extent Axis.W p.w in
    Vec6.set
      (Vec6.set (Vec6.set x_shape Axis.H h) Axis.W w)
      Axis.C
      (Vec6.get weight_shape Axis.N)

  module Compute (S : Semantics.SEMANTICS) = struct
    module Wa = Window_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
        ~x ~weight ~bias (out : Semantics.position S.index Vec6.t) =
      let in_per_group, out_per_group =
        or_invalid_arg (validate_channels ~weight_shape p)
      in
      let oc = Vec6.get out Axis.C in
      let group =
        if (p.groups :> int) = 1 then S.index_const 0
        else
          S.index_floor_div_pos (S.of_index oc)
            (Dim_arith.Extent.to_pos out_per_group)
      in
      let wh =
        Wa.window ~kernel:p.h.kernel ~stride:p.h.stride
          ~pad_before:p.h.pad_before ~dilation:p.h.dilation
          ~in_extent:(Vec6.get x_shape Axis.H) (Vec6.get out Axis.H)
      in
      let ww =
        Wa.window ~kernel:p.w.kernel ~stride:p.w.stride
          ~pad_before:p.w.pad_before ~dilation:p.w.dilation
          ~in_extent:(Vec6.get x_shape Axis.W) (Vec6.get out Axis.W)
      in
      (* [load6] (6 explicit indices), not [load]+a closure: this is the
         highest-call-volume site in the engine (one call per input-channel x
         kernel-h x kernel-w x output-pixel), and a closure built fresh per
         call — as [x_idx]/[w_idx] used to be here — allocates every time.
         See the `.ai/` pt2 inference perf doc. *)
      let on = Vec6.get out Axis.N
      and ot = Vec6.get out Axis.T
      and od = Vec6.get out Axis.D in
      let acc =
        S.sum ~lo:S.index_zero ~hi:(S.index_extent in_per_group)
          (fun local_ic ->
            S.sum ~lo:wh.lo ~hi:wh.hi (fun kh ->
                S.sum ~lo:ww.lo ~hi:ww.hi (fun kw ->
                    let ic =
                      S.assume_index
                        (S.index_add
                           (S.index_scale (in_per_group :> int) group)
                           (S.of_index local_ic))
                    in
                    S.mul
                      (S.load6 x ~n:on ~t:ot ~d:od ~h:(wh.src kh) ~w:(ww.src kw)
                         ~c:ic)
                      (S.load6 weight ~n:oc ~t:S.index_zero ~d:S.index_zero
                         ~h:kh ~w:kw ~c:local_ic))))
      in
      S.add acc
        (S.load bias
           (Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
              ~h:S.index_zero ~w:S.index_zero ~c:oc))
  end
end
