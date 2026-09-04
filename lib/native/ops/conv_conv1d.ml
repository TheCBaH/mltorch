(* 1D convolution ([aten.conv1d.default]). Native's frame has no genuinely
   1D shape -- every tensor is always 6-axis -- so [Conv1d] is a distinct
   Native graph node (its own JSON tag, its own bridge/importer arm) whose
   arithmetic delegates to [Conv2d] with the H axis pinned to
   [Conv2d.unit_window]: kernel 1, stride 1, no padding, dilation 1, which
   always keeps H's extent at 1 in and out. That is the same delegation shape
   [Conv2d_padding] already uses for [Conv2d] (its own comment: "calls into
   Conv2d's" implementation, not its node) -- one node per ATen op, per
   .ai/native_add_op.md's design goal, sharing the compute rather than
   decomposing into a [Reshape] + [Conv2d] pair. *)

open Conv_conv2d

module Conv1d = struct
  (* No [h] field at all -- unlike [Conv2d_padding]'s own [params], which still
     carries a full H window because "same"/"valid" apply to both axes, this op
     only ever has one spatial axis, so there is no H value a graph could hold
     that this record would need to reject. *)
  type params = {
    w : Conv2d.axis_window;
    in_channels : Dim.extent Dim.t;
    groups : Op_config.Pos.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"conv1d_params" (fun w in_channels groups ->
        { w; in_channels; groups })
    |> Jsont.Object.mem "w" Conv2d.axis_window_jsont ~enc:(fun p -> p.w)
    |> Jsont.Object.mem "in_channels" Dim.extent_jsont ~enc:(fun p ->
        p.in_channels)
    |> Jsont.Object.mem "groups" Op_config.Pos.jsont ~enc:(fun p -> p.groups)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{w=%a;@ in_channels=%a;@ groups=%a}@]"
      Conv2d.pp_axis_window p.w Dim.pp p.in_channels Op_config.Pos.pp p.groups

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t;
    bias : Tensor_ref.t option;
  }

  let name = "Conv1d"

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
    Fmt.pf fmt "@[<hv 2>conv1d@ x=%a@ weight=%a@ bias=%a@ params=%a@]" pp_ref
      t.x pp_ref t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  let to_conv2d_params (p : params) : Conv2d.params =
    {
      Conv2d.h = Conv2d.unit_window;
      w = p.w;
      in_channels = p.in_channels;
      groups = p.groups;
    }

  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    Conv2d.output_shape ~x_shape ~weight_shape (to_conv2d_params p)

  module Compute (S : Semantics.SEMANTICS) = struct
    module C = Conv2d.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
        ~x ~weight ~bias out =
      C.pixel (to_conv2d_params p) ~x_shape ~weight_shape ~x ~weight ~bias out
  end
end
