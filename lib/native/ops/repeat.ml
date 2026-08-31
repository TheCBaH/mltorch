(* ATen's `repeat` family: `repeat.default` (tile the whole tensor) and
   `repeat_interleave.self_int` (duplicate each element along one axis).
   Own file rather than [reshape.ml] -- both reuse [Reshape.delinearize]'s
   "mod x d = x - d*(x/d)" / floor-div idioms per axis, not its
   flatten-then-redistribute strategy, since neither reinterprets the
   tensor as a different rank; every axis keeps its own identity. *)

(* [x_extent * repeats] bounded in int64 BEFORE narrowing -- the same "bound
   the product before multiplying" rule [Resize.Bilinear_axis.check] follows
   for its own [in_extent * out_extent] aggregate: both factors are
   individually plausible (each already fits an [int]), but their product is
   the aggregate CLAUDE.md's overflow rule means, and 32-bit js_of_ocaml can
   wrap a product that overflows even though neither factor alone does.
   Reuses [Window_over_limit]'s [`Output_extent] row -- the same reuse
   [Pad]/[Concat] make -- rather than a new fault, since it IS the output
   extent on the tiled axis. Shared by [Repeat] (every axis) and
   [RepeatInterleave] (one named axis) so the bound cannot drift between the
   two. *)
let bounded_axis_extent ~(x_extent : Dim.extent Dim.t) ~(repeats : int) =
  let limit = Kernel.Limits.Hard.extent in
  let x = Int64.of_int (x_extent :> int) in
  let r = Int64.of_int repeats in
  if
    x >= limit || r >= limit
    || Int64.compare x (Int64.div (Int64.sub limit 1L) r) > 0
  then
    let value = if x >= limit || r >= limit then limit else Int64.mul x r in
    Err.fail
      (`Window_over_limit
         Shape_error.Window_over_limit.{ what = `Output_extent; value; limit })
  else Err.return (Dim.extent (Int64.to_int (Int64.mul x r)))

module Repeat = struct
  type params = { repeats : Vec6.shape }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"repeat_params" (fun repeats -> { repeats })
    |> Jsont.Object.mem "repeats" Vec6.shape_jsont ~enc:(fun p -> p.repeats)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{repeats=%a}@]" Vec6.pp_shape p.repeats

  type t = { params : params; x : Tensor_ref.t }

  let name = "Repeat"

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
    Fmt.pf fmt "@[<hv 2>repeat@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    Err.List.fold_left
      (fun s axis ->
        let+ e =
          bounded_axis_extent ~x_extent:(Vec6.get x_shape axis)
            ~repeats:(Vec6.get p.repeats axis :> int)
        in
        Vec6.set s axis e)
      x_shape Axis.all

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [out]'s coordinate on axis [a] tiles modulo [x_shape]'s own extent
       there -- [Reshape.delinearize]'s per-axis remainder, without
       [Reshape]'s flatten-then-redistribute: each axis keeps its own
       identity, so no cross-axis linearization is needed. *)
    let pixel ~(x_shape : Vec6.shape) x
        (out : Semantics.position S.index Vec6.t) =
      let coord =
        Vec6.mapi
          (fun a o ->
            let ext = (Vec6.get x_shape a :> int) in
            let o = S.of_index o in
            let q = S.index_floor_div_pos o (Op_config.Pos.of_int ext) in
            S.assume_index (S.index_add o (S.index_scale (-ext) q)))
          out
      in
      S.load x coord
  end
end

(* `repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None,
   *, SymInt? output_size=None) -> Tensor`, restricted to an explicit [dim]
   -- every corpus occurrence supplies one; the [dim=None] flatten-first
   form is a genuinely different reshape+repeat composition, deferred until
   a model demonstrates it. [output_size] is read-and-discarded: it exists
   only to let a Tensor-valued [repeats] avoid a device sync, and this is
   the SymInt-scalar overload, so ATen's own [repeat_interleave_symint]
   never consults it either.

   Unlike [Repeat], which tiles the WHOLE tensor by wrapping every axis
   modulo its own extent, this DUPLICATES each element along one axis: the
   k-th group of [repeats] consecutive output positions all read source
   position k. That is [out / repeats] (floor division), not [out mod
   x_extent] -- the two ops are each other's opposite composition order,
   the same distinction numpy's own `repeat` vs `tile` draws. *)
module RepeatInterleave = struct
  type params = { axis : Axis.t; repeats : Op_config.Pos.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"repeat_interleave_params" (fun axis repeats ->
        { axis; repeats = Op_config.Pos.of_int repeats })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "repeats" Jsont.int ~enc:(fun p -> (p.repeats :> int))
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a repeats=%d}@]" Axis.pp p.axis (p.repeats :> int)

  type t = { params : params; x : Tensor_ref.t }

  let name = "RepeatInterleave"

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
    Fmt.pf fmt "@[<hv 2>repeat_interleave@ x=%a@ params=%a@]" pp_ref t.x
      pp_params t.params

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let+ e =
      bounded_axis_extent ~x_extent:(Vec6.get x_shape p.axis)
        ~repeats:(p.repeats :> int)
    in
    Vec6.set x_shape p.axis e

  module Compute (S : Semantics.SEMANTICS) = struct
    (* Every axis but [params.axis] reads straight through; on [params.axis]
       the source position is [out / repeats] (floor division) -- in bounds
       by construction, since [output_shape] set that axis's own extent to
       [x_extent * repeats], so [out] there ranges over [0, x_extent *
       repeats) and the quotient stays inside [0, x_extent). *)
    let pixel (p : params) x (out : Semantics.position S.index Vec6.t) =
      let coord =
        Vec6.mapi
          (fun a o ->
            if Axis.equal a p.axis then
              S.assume_index (S.index_floor_div_pos (S.of_index o) p.repeats)
            else o)
          out
      in
      S.load x coord
  end
end
