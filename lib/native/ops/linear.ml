(* Fully-connected / `aten.addmm` layer. Output[n,...,oc] reduces over
   in_features at the same (n,t,d,h,w) position — no spatial kernel, so this
   is [Conv2d]'s [cin] reduction alone, without the kh/kw window (resnet18's
   FC layer runs on an already globally-pooled, spatially-1x1 tensor, so a
   spatial kernel would be vacuous here anyway). Weight is laid out
   [Out,1,1,1,1,In] (matching [Conv2d]'s [Cout,1,1,Kh,Kw,Cin] minus the kernel
   axes); bias is [1,1,1,1,1,Out]. See .ai/native_compute_design.md §2. *)

module Linear = struct
  type params = { in_features : Dim.extent Dim.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"linear_params" (fun in_features -> { in_features })
    |> Jsont.Object.mem "in_features" Dim.extent_jsont ~enc:(fun p ->
        p.in_features)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{in_features=%a}@]" Dim.pp p.in_features

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t;
    bias : Tensor_ref.t option;
  }

  let name = "Linear"

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
    Fmt.pf fmt "@[<hv 2>linear@ x=%a@ weight=%a@ bias=%a@ params=%a@]" pp_ref
      t.x pp_ref t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  (* N/T/D/H/W pass through from [x_shape] (no spatial reduction); only C
     changes, to [weight_shape]'s [Out]. Extent-space, no [:> int] round-trips.
     See .ai/native_compute_design.md §2b. *)
  let output_shape (p : params) ~(x_shape : Vec6.shape)
      ~(weight_shape : Vec6.shape) =
    let expected = p.in_features in
    let x_channels = Vec6.get x_shape Axis.C in
    let weight_channels = Vec6.get weight_shape Axis.C in
    if not (Dim.equal x_channels expected) then
      Core.fail
        (`Linear
           (Shape_error.Linear.Input_channels_mismatch
              Shape_error.Linear.{ actual = x_channels; expected }))
    else if not (Dim.equal weight_channels expected) then
      Core.fail
        (`Linear
           (Shape_error.Linear.Weight_channels_mismatch
              Shape_error.Linear.{ actual = weight_channels; expected }))
    else Core.return (Vec6.copy weight_shape ~src:Axis.N ~dst:Axis.C x_shape)

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~x ~weight ~bias
        (out : Semantics.position S.index Vec6.t) =
      let oc = Vec6.get out Axis.C in
      let acc =
        S.sum ~lo:S.index_zero ~hi:(S.index_extent p.in_features) (fun ic ->
            S.mul
              (S.load x (out |> Vec6.set_c ic))
              (S.load weight
                 (Vec6.make ~n:oc ~t:S.index_zero ~d:S.index_zero
                    ~h:S.index_zero ~w:S.index_zero ~c:ic)))
      in
      S.add acc
        (S.load bias
           (Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
              ~h:S.index_zero ~w:S.index_zero ~c:oc))
  end
end
