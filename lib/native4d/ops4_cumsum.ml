(* [Cumsum4], split out of ops4.ml (which crossed the tracked 1000-line
   ceiling, scripts/check-file-size.sh) rather than folded into it. *)

(* Native's [Reduce.Cumsum] with its axis narrowed to the dialect's four --
   the same [Softmax4]/[RepeatInterleave4] reason: it names an axis, so the
   field is [Axis4.t] and T/D are unsayable in a four-axis graph. Shape-
   preserving like [Softmax4] (no axis is dropped or rescaled), so there is
   no post-hoc [Shape4.of_vec6] re-check either. The shape rule and pixel map
   delegate whole to [Reduce.Cumsum] rather than restating them. *)
module Cumsum4 = struct
  type params = { axis : Axis4.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"cumsum4_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis4.pp p.axis

  type t = { params : params; x : Tensor_ref.t }

  let name = "Cumsum4"

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
    Fmt.pf fmt "@[<hv 2>cumsum4@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params
end
