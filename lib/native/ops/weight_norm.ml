(* `_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor`: the weight a
   `weight_norm` parametrization recomputes, [w = v * (g / ||v||)], with the norm
   taken over every dimension of [v] EXCEPT [dim]. [g] holds one magnitude per
   position along [dim] (extent 1 on every other axis), so it broadcasts against
   [v]. Both operands are weights, so the result is used as a convolution's
   weight; the norm is a reduction over most of the tensor, which is why
   [Eval_direct] computes it once per position along [dim] rather than once per
   output element, and why this generic pixel is only the definition. *)

module Weight_norm = struct
  type params = { axis : Axis.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"weight_norm_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) = Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis.pp p.axis

  type t = { params : params; v : Tensor_ref.t; g : Tensor_ref.t }

  let name = "WeightNorm"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          v = get "v" Tensor_ref.jsont;
          g = get "g" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("v", Json_util.enc Tensor_ref.jsont t.v);
            ("g", Json_util.enc Tensor_ref.jsont t.g);
          ])
      Jsont.json

  let operands (t : t) = [ t.v; t.g ]
  let map_operands f (t : t) = { t with v = f t.v; g = f t.g }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>weight_norm@ v=%a@ g=%a@ params=%a@]" pp_ref t.v pp_ref
      t.g pp_params t.params

  (* [g] must be [v]'s extent along [axis] and 1 everywhere else. *)
  let output_shape ~(v_shape : Vec6.shape) ~(g_shape : Vec6.shape) (p : params)
      =
    let open Err.Syntax in
    let+ () =
      Err.List.iter
        (fun a ->
          let want =
            if Axis.equal a p.axis then Vec6.get v_shape a else Dim.one
          in
          let got = Vec6.get g_shape a in
          if Dim.equal want got then Err.return ()
          else
            Err.fail
              (`Broadcast
                 Shape_error.Broadcast.{ axis = a; lhs = want; rhs = got }))
        Axis.all
    in
    v_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(v_shape : Vec6.shape) ~v ~g
        (out : Semantics.position S.index Vec6.t) =
      let others = List.filter (fun a -> not (Axis.equal a p.axis)) Axis.all in
      let sum_sq =
        let rec go dims override =
          match dims with
          | [] ->
              let idx =
                List.fold_left (fun c (a, i) -> Vec6.set c a i) out override
              in
              let x = S.load v idx in
              S.mul x x
          | d :: rest ->
              S.sum ~lo:S.index_zero
                ~hi:(S.index_extent (Vec6.get v_shape d))
                (fun i -> go rest ((d, i) :: override))
        in
        go others []
      in
      let zero = Vec6.map (fun _ -> S.index_zero) out in
      let g_at = S.load g (Vec6.copy_axis out p.axis zero) in
      S.mul (S.load v out) (S.div g_at (S.sqrt sum_sq))
  end
end
