(* 2D convolution (NHWC). Output[n,t,d,h,w,oc] reduces over input channels and the
   kernel window; the input position is the affine index [out*stride + k - pad],
   built through the SEMANTICS index ops so it specialises to either concrete ints
   (Direct) or index expressions (Symbolic). The kh/kw reduction bounds are
   clipped (via [Index_max]/[Index_min]) to exactly the kernel offsets whose source
   position lands inside the input — so that position is always a valid index,
   never a generate-then-guard read of the padding region (see
   .ai/native_compute_design.md §4 "Open issue"). Weight is laid out
   [Cout,1,1,Kh,Kw,Cin_per_group]; bias is [1,1,1,1,1,Cout] (broadcast —
   extent-1 axes still go through [load]'s ordinary broadcast). See
   .ai/native_compute_design.md §2. *)

module Conv2d = struct
  (* [params]/[output_shape] are outside [Compute] so Direct/Symbolic share one
     [params] type. Field types per .ai/native_op_config.md. *)
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
    let in_channels = (p.in_channels :> int) in
    let groups = (p.groups :> int) in
    let out_channels = (Vec6.get weight_shape Axis.N :> int) in
    let weight_in_per_group = (Vec6.get weight_shape Axis.C :> int) in
    if in_channels mod groups <> 0 then
      invalid_arg "Conv2d: in_channels must be divisible by groups";
    if out_channels mod groups <> 0 then
      invalid_arg "Conv2d: out_channels must be divisible by groups";
    let in_per_group = in_channels / groups in
    if weight_in_per_group <> in_per_group then
      invalid_arg
        "Conv2d: weight C extent must equal in_channels divided by groups";
    if not (Dim.equal (Vec6.get weight_shape Axis.H) p.h.kernel) then
      invalid_arg "Conv2d: weight H extent must equal h.kernel";
    if not (Dim.equal (Vec6.get weight_shape Axis.W) p.w.kernel) then
      invalid_arg "Conv2d: weight W extent must equal w.kernel";
    (in_per_group, out_channels / groups)

  (* N/T/D pass through; H/W shrink via [Window_axis.output_extent]; C = Cout from
     weight_shape. *)
  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    let _in_per_group, _out_per_group = validate_channels ~weight_shape p in
    if not (Dim.equal (Vec6.get x_shape Axis.C) p.in_channels) then
      invalid_arg "Conv2d: input C extent must equal in_channels";
    let out_extent axis (w : axis_window) =
      Window_axis.output_extent ~in_extent:(Vec6.get x_shape axis)
        ~kernel:w.kernel ~stride:w.stride ~pad_before:w.pad_before
        ~pad_after:w.pad_after ~dilation:w.dilation
    in
    (* Start from the input shape (N/T/D pass through unchanged) and replace only
       the axes the op resizes — all in extent-space, no [:> int] round-trips. *)
    Vec6.set
      (Vec6.set
         (Vec6.set x_shape Axis.H (out_extent Axis.H p.h))
         Axis.W (out_extent Axis.W p.w))
      Axis.C
      (Vec6.get weight_shape Axis.N)

  module Compute (S : Semantics.SEMANTICS) = struct
    module Wa = Window_axis.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
        ~x ~weight ~bias (out : Axis.t -> Semantics.position S.index) =
      let in_per_group, out_per_group = validate_channels ~weight_shape p in
      let oc = out Axis.C in
      let group =
        if (p.groups :> int) = 1 then S.index_const 0
        else
          S.index_floor_div_pos (S.of_index oc)
            (Op_config.Pos.of_int out_per_group)
      in
      let wh =
        Wa.window ~kernel:p.h.kernel ~stride:p.h.stride
          ~pad_before:p.h.pad_before ~dilation:p.h.dilation
          ~in_extent:(Vec6.get x_shape Axis.H) (out Axis.H)
      in
      let ww =
        Wa.window ~kernel:p.w.kernel ~stride:p.w.stride
          ~pad_before:p.w.pad_before ~dilation:p.w.dilation
          ~in_extent:(Vec6.get x_shape Axis.W) (out Axis.W)
      in
      let acc =
        S.sum ~lo:S.index_zero
          ~hi:(S.index_extent (Dim.extent in_per_group))
          (fun local_ic ->
            S.sum ~lo:wh.lo ~hi:wh.hi (fun kh ->
                S.sum ~lo:ww.lo ~hi:ww.hi (fun kw ->
                    let ic =
                      S.assume_index
                        (S.index_add
                           (S.index_scale in_per_group group)
                           (S.of_index local_ic))
                    in
                    let x_idx a =
                      match a with
                      | Axis.H -> wh.src kh
                      | Axis.W -> ww.src kw
                      | Axis.C -> ic
                      | _ -> out a
                    in
                    let w_idx a =
                      match a with
                      | Axis.N -> oc
                      | Axis.H -> kh
                      | Axis.W -> kw
                      | Axis.C -> local_ic
                      | _ -> S.index_zero
                    in
                    S.mul (S.load x x_idx) (S.load weight w_idx))))
      in
      S.add acc
        (S.load bias (fun a -> match a with Axis.C -> oc | _ -> S.index_zero))
  end
end
