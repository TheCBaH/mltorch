(* [Max_dim4]: Native's [Reduce.MaxDim] with its axis narrowed to the dialect's
   four, the same [Softmax4]/[Cumsum4] reason -- it names an axis, so the field
   is [Axis4.t] and T/D are unsayable in a four-axis graph. Two outputs, values
   and int64 indices, sharing the reduced shape like the pooling ops with
   indices. The shape rule and both pixel maps delegate whole to
   [Reduce.MaxDim]; in particular the values fold with the same predicate as
   the index, which [Max_keepdims] (a plain [Amax]) does not guarantee under
   NaN. *)
module Max_dim4 = struct
  type params = { axis : Axis4.t; keepdim : bool }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"max_dim4_params" (fun axis keepdim ->
        { axis; keepdim })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "keepdim" Jsont.bool ~enc:(fun p -> p.keepdim)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a;@ keepdim=%b}@]" Axis4.pp p.axis p.keepdim

  type t = { params : params; x : Tensor_ref.t }

  let name = "Max_dim4"

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
    Fmt.pf fmt "@[<hv 2>max_dim4@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params
end
