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

(* A rank-2 identity-matrix factory.  [Aten_shape.of_aten] right-aligns a
 * rank-2 ATen shape onto [W;C] (the innermost two frame axes), so [n]'s rows
 * land on [W] and [m]'s columns on [C] -- fixed axes, the same assumption
 * [Arange]'s own pixel makes for its single [C] axis. *)
module Eye = struct
  type params = { shape : Vec6.shape; fmt : Payload.packed_fmt }
  type t = { params : params }

  let name = "Eye"

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"eye_params" (fun shape fmt -> { shape; fmt })
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
    Fmt.pf fmt "@[<hv 2>eye@ params=%a@]" pp_params t.params

  let output_shape (p : params) = Err.return p.shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [1] on the diagonal ([w = c]), [0] elsewhere -- a single [index_eq],
     * the same "compare via the index domain" idiom [Pad]'s reflect region
     * test uses, needing no new [SEMANTICS] primitive. *)
    let pixel _ (out : Semantics.position S.index Vec6.t) =
      let diff =
        S.index_add (S.of_index out.Vec6.w)
          (S.index_scale (-1) (S.of_index out.Vec6.c))
      in
      S.select (S.index_eq diff (S.index_const 0)) (S.const 1.) (S.const 0.)
  end
end

(* A bounded, endpoint-exclusive range factory.  Keeping the three scalar
 * bounds in the payload (rather than lowering it to an opaque constant) lets
 * direct evaluation and symbolic compilation share the same coordinate rule.
 * The corpus only requires positive steps; refusing the other ATen branch is
 * preferable to an accidentally empty/negative Native extent. *)
module Arange = struct
  (* The exact int64 view of [params]' bounds, ALONGSIDE the float fields
     every existing caller already uses -- never replacing them, per this
     plan's "add alongside, never convert" rule. [Some] only when every
     bound was decoded from a real ATen integer scalar rather than a legacy
     float one (see [op_bridge_factory.ml]'s arange arm), so a caller that
     has this can generate exact int64 values with no float round trip at
     all. [stop] is carried for symmetry with [params]' own three fields,
     and used by [length_exact] below (the plan's own overflow-checked-count
     item) even though [value_i64_exact] itself only needs [start]/[step].
     See the implementation tracker's D02. *)
  module Exact = struct
    type t = { start : int64; stop : int64; step : int64 }

    let jsont : t Jsont.t =
      Jsont.Object.map ~kind:"arange_exact" (fun start stop step ->
          { start; stop; step })
      |> Jsont.Object.mem "start" Json_util.i64_jsont ~enc:(fun e -> e.start)
      |> Jsont.Object.mem "stop" Json_util.i64_jsont ~enc:(fun e -> e.stop)
      |> Jsont.Object.mem "step" Json_util.i64_jsont ~enc:(fun e -> e.step)
      |> Jsont.Object.finish
  end

  type params = {
    start : float;
    stop : float;
    step : float;
    fmt : Payload.packed_fmt;
    exact : Exact.t option;
  }

  type t = { params : params }

  let name = "Arange"

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"arange_params" (fun start stop step fmt exact ->
        { start; stop; step; fmt; exact })
    |> Jsont.Object.mem "start" Json_util.f32_jsont ~enc:(fun p -> p.start)
    |> Jsont.Object.mem "stop" Json_util.f32_jsont ~enc:(fun p -> p.stop)
    |> Jsont.Object.mem "step" Json_util.f32_jsont ~enc:(fun p -> p.step)
    |> Jsont.Object.mem "fmt" Payload.packed_fmt_jsont ~enc:(fun p -> p.fmt)
    |> Jsont.Object.opt_mem "exact" Exact.jsont ~enc:(fun p -> p.exact)
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
    let pp_exact fmt = function
      | None -> ()
      | Some { Exact.start; stop; step } ->
          Fmt.pf fmt ";@ exact={%Ld,%Ld,%Ld}" start stop step
    in
    Fmt.pf fmt "@[<hv>{start=%g;@ stop=%g;@ step=%g;@ fmt=%s%a}@]" p.start
      p.stop p.step (Payload.fmt_name elt) pp_exact p.exact

  let pp _ fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>arange@ params=%a@]" pp_params t.params

  (* [checked_sub] mirrors [checked_add]'s overflow check (different sign of
     the two operands, and the result's sign differs from the minuend's) --
     the classic subtraction-overflow condition, needed by [length_exact]'s
     [stop - start] below. *)
  let checked_sub a b =
    let c = Int64.sub a b in
    let same_sign x y =
      Bool.equal (Int64.compare x 0L >= 0) (Int64.compare y 0L >= 0)
    in
    if (not (same_sign a b)) && not (same_sign a c) then None else Some c

  (* [length]'s exact int64 twin: the element count computed entirely in
     [int64] ceiling division, with no float round trip -- fixes the D02
     count gap the implementation tracker documents (real ATen and a
     float-ceiling count can disagree once [start]/[stop] exceed 2^53, since
     an odd exact bound can round to the same float as a neighbor). Ceiling
     division is [q + (if r <> 0 then 1 else 0)] ([q]/[r] = [diff]/[step]'s
     floor quotient/remainder) rather than the textbook [(diff + step - 1) /
     step], which would need its own overflow check on [diff + step]: the
     [q + 1] here cannot overflow, since [step >= 2] bounds [q] well below
     [max_int], and the one [step = 1] case where [q] could reach [max_int]
     forces [r = 0] (no [+ 1] taken), as [diff <= max_int] itself. *)
  let length_exact ({ Exact.start; stop; step } as e) =
    let fault fault =
      Err.fail
        (`Arange
           Shape_error.Arange.
             {
               start = Int64.to_float e.start;
               stop = Int64.to_float e.stop;
               step = Int64.to_float e.step;
               fault;
             })
    in
    if Int64.compare step 0L <= 0 then fault `Non_positive_step
    else
      match checked_sub stop start with
      | None -> fault `Count_overflow
      | Some diff ->
          if Int64.compare diff 0L <= 0 then fault `Empty
          else
            let q = Int64.div diff step in
            let r = Int64.rem diff step in
            let count = if Int64.equal r 0L then q else Int64.add q 1L in
            if Int64.compare count Kernel.Limits.Hard.extent >= 0 then
              fault `Over_limit
            else Err.return (Int64.to_int count)

  let length (p : params) =
    match p.exact with
    | Some e -> length_exact e
    | None ->
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

  module Overflow = struct
    type t = { start : int64; step : int64; i : int }
  end

  (* [checked_mul]/[checked_add] exist because [Int64.mul]/[Int64.add] wrap
     silently on overflow -- exactly the failure mode the plan's own "do not
     compute start + i * step with unchecked overflow" rule calls out. Not a
     general-purpose checked-arithmetic module: just the two operations
     [value_i64_exact] needs, at the width it needs them. *)
  let checked_mul a b =
    if Int64.equal a 0L then Some 0L
    else
      let c = Int64.mul a b in
      if Int64.equal (Int64.div c a) b then Some c else None

  let checked_add a b =
    let c = Int64.add a b in
    let same_sign x y =
      Bool.equal (Int64.compare x 0L >= 0) (Int64.compare y 0L >= 0)
    in
    if same_sign a b && not (same_sign a c) then None else Some c

  (* [value]'s exact int64 twin: [start + i*step] computed entirely in
     [int64], with no float round trip -- what [eval_direct.ml]'s Arange arm
     uses in place of [Int64.of_float (value p i)] whenever [p.exact] is
     [Some], fixing the exact truncation bug the implementation tracker's
     D02/D09 notes name. [i] is always well under 2^31 by [length]'s own
     extent check, but [step] is an arbitrary caller-supplied [int64], so the
     product (and then the sum) genuinely can overflow -- checked, not
     assumed. *)
  let value_i64_exact { Exact.start; step; _ } i =
    let overflow () =
      Err.fail (`Arange_i64_overflow { Overflow.start; step; i })
    in
    match checked_mul step (Int64.of_int i) with
    | None -> overflow ()
    | Some offset -> (
        match checked_add start offset with
        | None -> overflow ()
        | Some result -> Err.return result)

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel p out =
      S.add (S.const p.start)
        (S.mul (S.const p.step) (S.value_of_index (S.of_index out.Vec6.c)))
  end
end
