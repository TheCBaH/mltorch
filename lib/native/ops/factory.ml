(* Tensor factories.  Unlike arithmetic ops, a factory has no operand from
 * which a result format could be inferred: its requested shape and dtype are
 * semantic parameters and therefore live in the IR payload. *)

module Zeros = struct
  type params = { shape : Vec6.shape; fmt : Payload.packed_fmt }
  type t = { params : params }

  let name = "Zeros"

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"zeros_params" (fun shape fmt -> { shape; fmt })
    |> Jsont.Object.mem "shape" Vec6.shape_jsont ~enc:(fun p -> p.shape)
    |> Jsont.Object.mem "fmt" Payload.packed_fmt_jsont ~enc:(fun p -> p.fmt)
    |> Jsont.Object.finish

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        { params = Json_util.req_field ms "params" params_jsont name })
      ~enc:(fun t ->
        Json_util.jobj [ ("params", Json_util.enc params_jsont t.params) ])
      Jsont.json

  let operands _ = []
  let map_operands _ t = t

  let pp_params fmt (p : params) =
    let (Payload.Fmt elt) = p.fmt in
    Fmt.pf fmt "@[<hv>{shape=%a;@ fmt=%s}@]" Vec6.pp_shape p.shape
      (Payload.fmt_name elt)

  let pp _ fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>zeros@ params=%a@]" pp_params t.params

  let output_shape (p : params) = Err.return p.shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel _ = S.const 0.
  end
end

(* A bounded, endpoint-exclusive range factory.  Keeping the three scalar
 * bounds in the payload (rather than lowering it to an opaque constant) lets
 * direct evaluation and symbolic compilation share the same coordinate rule.
 * The corpus only requires positive steps; refusing the other ATen branch is
 * preferable to an accidentally empty/negative Native extent. *)
module Arange = struct
  type params = {
    start : float;
    stop : float;
    step : float;
    fmt : Payload.packed_fmt;
  }

  type t = { params : params }

  let name = "Arange"

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"arange_params" (fun start stop step fmt ->
        { start; stop; step; fmt })
    |> Jsont.Object.mem "start" Json_util.f32_jsont ~enc:(fun p -> p.start)
    |> Jsont.Object.mem "stop" Json_util.f32_jsont ~enc:(fun p -> p.stop)
    |> Jsont.Object.mem "step" Json_util.f32_jsont ~enc:(fun p -> p.step)
    |> Jsont.Object.mem "fmt" Payload.packed_fmt_jsont ~enc:(fun p -> p.fmt)
    |> Jsont.Object.finish

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        { params = Json_util.req_field ms "params" params_jsont name })
      ~enc:(fun t ->
        Json_util.jobj [ ("params", Json_util.enc params_jsont t.params) ])
      Jsont.json

  let operands _ = []
  let map_operands _ t = t

  let pp_params fmt (p : params) =
    let (Payload.Fmt elt) = p.fmt in
    Fmt.pf fmt "@[<hv>{start=%g;@ stop=%g;@ step=%g;@ fmt=%s}@]" p.start p.stop
      p.step (Payload.fmt_name elt)

  let pp _ fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>arange@ params=%a@]" pp_params t.params

  let length (p : params) =
    let fault fault =
      Err.fail
        (`Arange
           Shape_error.Arange.
             { start = p.start; stop = p.stop; step = p.step; fault })
    in
    if
      not
        (Float.is_finite p.start && Float.is_finite p.stop
       && Float.is_finite p.step)
    then fault `Non_finite
    else if p.step <= 0. then fault `Non_positive_step
    else
      let count = Float.ceil ((p.stop -. p.start) /. p.step) in
      if count <= 0. then fault `Empty
      else if count >= Int64.to_float Kernel.Limits.Hard.extent then
        fault `Over_limit
      else Err.return (int_of_float count)

  let output_shape p =
    let open Err.Syntax in
    let+ count = length p in
    Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:count

  let value (p : params) i = p.start +. (float_of_int i *. p.step)

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel p out =
      S.add (S.const p.start)
        (S.mul (S.const p.step) (S.value_of_index (S.of_index out.Vec6.c)))
  end
end
