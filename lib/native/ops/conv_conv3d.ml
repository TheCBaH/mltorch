(* 3D convolution ([aten.conv3d.default]). Unlike [Conv1d], whose ATen schema
   genuinely has only one spatial axis (so pinning a second is the whole of
   its semantics), `conv3d` genuinely names THREE independent spatial axes --
   a delegation onto [Conv2d] would silently drop one of them. [Conv3d] is
   therefore its own [Compute], one window axis wider than [Conv2d]'s: the
   three ATen spatial axes ([D],[H],[W] in ATen's own [N,C,D,H,W] order) land
   on Native's [D]/[H]/[W] directly -- the same letters, so the relayout
   permutation at the ATen boundary (op_bridge_decode.ml's [perm_conv3d]) is
   the whole story, with no axis renaming to track here. That leaves exactly
   one Native axis, [T], with nothing to hold: `of_aten`'s right-alignment of
   a rank-5 ATen tensor already leaves the frame's outermost axis (raw [N])
   at its default extent 1, and the same permutation that moves ATen's batch
   role onto native [N] moves that unit slot onto [T] -- so [T] passes
   through unchanged here exactly the way [N]/[T]/[D] do together in
   [Conv2d]. Reuses [Conv2d.axis_window]/[Shape_error.Convolution] unchanged;
   only [output_shape]/[Compute] genuinely need a third window. *)

open Conv_conv2d

let or_invalid_arg = function
  | Ok x -> x
  | Error e ->
      invalid_arg (Format.asprintf "%a" Shape_error.pp (Err.Error.kind e))

module Conv3d = struct
  type params = {
    d : Conv2d.axis_window;
    h : Conv2d.axis_window;
    w : Conv2d.axis_window;
    in_channels : Dim.extent Dim.t;
    groups : Op_config.Pos.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"conv3d_params" (fun d h w in_channels groups ->
        { d; h; w; in_channels; groups })
    |> Jsont.Object.mem "d" Conv2d.axis_window_jsont ~enc:(fun p -> p.d)
    |> Jsont.Object.mem "h" Conv2d.axis_window_jsont ~enc:(fun p -> p.h)
    |> Jsont.Object.mem "w" Conv2d.axis_window_jsont ~enc:(fun p -> p.w)
    |> Jsont.Object.mem "in_channels" Dim.extent_jsont ~enc:(fun p ->
        p.in_channels)
    |> Jsont.Object.mem "groups" Op_config.Pos.jsont ~enc:(fun p -> p.groups)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{d=%a;@ h=%a;@ w=%a;@ in_channels=%a;@ groups=%a}@]"
      Conv2d.pp_axis_window p.d Conv2d.pp_axis_window p.h Conv2d.pp_axis_window
      p.w Dim.pp p.in_channels Op_config.Pos.pp p.groups

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t;
    bias : Tensor_ref.t option;
  }

  let name = "Conv3d"

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
    Fmt.pf fmt "@[<hv 2>conv3d@ x=%a@ weight=%a@ bias=%a@ params=%a@]" pp_ref
      t.x pp_ref t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  let validate_channels ~(weight_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let in_channels = (p.in_channels :> int) in
    let groups = (p.groups :> int) in
    let out_channels = (Vec6.get weight_shape Axis.N :> int) in
    let weight_in_per_group = (Vec6.get weight_shape Axis.C :> int) in
    let* () =
      if in_channels mod groups <> 0 then
        Err.fail
          (`Convolution
             (Shape_error.Convolution.In_channels_not_divisible_by_groups
                Shape_error.Convolution.{ channels = in_channels; groups }))
      else Err.return ()
    in
    let* () =
      if out_channels mod groups <> 0 then
        Err.fail
          (`Convolution
             (Shape_error.Convolution.Out_channels_not_divisible_by_groups
                Shape_error.Convolution.{ channels = out_channels; groups }))
      else Err.return ()
    in
    let in_per_group = in_channels / groups in
    let* () =
      if weight_in_per_group <> in_per_group then
        Err.fail
          (`Convolution
             (Shape_error.Convolution.Weight_channels_mismatch
                Shape_error.Convolution.
                  { weight_in_per_group; expected_in_per_group = in_per_group }))
      else Err.return ()
    in
    let check_kernel axis (w : Conv2d.axis_window) =
      if Dim.equal (Vec6.get weight_shape axis) w.kernel then Err.return ()
      else
        Err.fail
          (`Convolution
             (Shape_error.Convolution.Weight_kernel_mismatch
                Shape_error.Convolution.
                  {
                    axis;
                    weight_extent = Vec6.get weight_shape axis;
                    kernel_extent = w.kernel;
                  }))
    in
    let* () = check_kernel Axis.D p.d in
    let* () = check_kernel Axis.H p.h in
    let* () = check_kernel Axis.W p.w in
    Err.return (in_per_group, out_channels / groups)

  (* N/T pass through; D/H/W shrink via [Window_axis.output_extent]; C = Cout
     from weight_shape. *)
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
    let out_extent axis (w : Conv2d.axis_window) =
      Window_axis.output_extent ~ceil_mode:false
        ~in_extent:(Vec6.get x_shape axis) ~kernel:w.kernel ~stride:w.stride
        ~pad_before:w.pad_before ~pad_after:w.pad_after ~dilation:w.dilation
    in
    let* d = out_extent Axis.D p.d in
    let* h = out_extent Axis.H p.h in
    let+ w = out_extent Axis.W p.w in
    Vec6.set
      (Vec6.set (Vec6.set (Vec6.set x_shape Axis.D d) Axis.H h) Axis.W w)
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
            (Op_config.Pos.of_int out_per_group)
      in
      let wd =
        Wa.window ~kernel:p.d.kernel ~stride:p.d.stride
          ~pad_before:p.d.pad_before ~dilation:p.d.dilation
          ~in_extent:(Vec6.get x_shape Axis.D) (Vec6.get out Axis.D)
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
      let on = Vec6.get out Axis.N and ot = Vec6.get out Axis.T in
      let acc =
        S.sum ~lo:S.index_zero
          ~hi:(S.index_extent (Dim.extent in_per_group))
          (fun local_ic ->
            S.sum ~lo:wd.lo ~hi:wd.hi (fun kd ->
                S.sum ~lo:wh.lo ~hi:wh.hi (fun kh ->
                    S.sum ~lo:ww.lo ~hi:ww.hi (fun kw ->
                        let ic =
                          S.assume_index
                            (S.index_add
                               (S.index_scale in_per_group group)
                               (S.of_index local_ic))
                        in
                        S.mul
                          (S.load6 x ~n:on ~t:ot ~d:(wd.src kd) ~h:(wh.src kh)
                             ~w:(ww.src kw) ~c:ic)
                          (S.load6 weight ~n:oc ~t:S.index_zero ~d:kd ~h:kh
                             ~w:kw ~c:local_ic)))))
      in
      S.add acc
        (S.load bias
           (Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
              ~h:S.index_zero ~w:S.index_zero ~c:oc))
  end
end
