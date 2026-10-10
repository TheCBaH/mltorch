(* Comparison, boolean and selection pointwise ops, split out of
   [Pointwise_binary]: the ones attention-mask construction uses. See
   pointwise.ml (the facade). Every comparison here follows [Gt_scalar]'s
   two-layer split: [Compute] is [SEMANTICS]-generic and yields float 0./1.
   (shared with [Symbolic]); [Graph_builder] declares the edge [Bool] and
   [Eval_direct] writes genuine [Payload.Bool] storage from the same formula.

   The scalar comparisons read an integer operand through the float domain, as
   [Gt_scalar] does, so values past 2^53 compare inexactly; mask positions and
   token counts are far below that. *)

open Pointwise_binary

(* [and] of two bool operands (ATen's [__and__.Tensor], the bitwise form,
   restricted to bool: integer bit patterns have no float-domain encoding).
   A bool reads as 0./1., so the result is 1. only when both are nonzero. *)
module Bitwise_and = struct
  type t = Bin.t

  let name = "Bitwise_and"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"bitwise_and" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let both a b =
      S.select
        (S.eq a (S.const 0.))
        (S.const 0.)
        (S.select (S.eq b (S.const 0.)) (S.const 0.) (S.const 1.))

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:both ~a_shape ~b_shape a b out
  end
end

(* [ge.Scalar(self, other) -> self >= other]: strictly greater, or equal. IEEE
   ordering again falls out of [S.lt]/[S.eq]: a NaN operand is neither, so the
   result is false, as in ATen. *)
module Ge_scalar = struct
  type t = Scalar_bin.t

  let name = "Ge_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"ge_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    let ge v s =
      S.select (S.lt s v) (S.const 1.)
        (S.select (S.eq v s) (S.const 1.) (S.const 0.))

    let pixel ~scalar x out = B.pixel ~combine:ge ~scalar x out
  end
end

(* [le.Tensor(self, other) -> self <= other]: tensor-tensor, broadcast. *)
module Le_tensor = struct
  type t = Bin.t

  let name = "Le_tensor"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"le_tensor" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let le a b =
      S.select (S.lt a b) (S.const 1.)
        (S.select (S.eq a b) (S.const 1.) (S.const 0.))

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:le ~a_shape ~b_shape a b out
  end
end

(* [lt.Scalar(self, other) -> self < other]. *)
module Lt_scalar = struct
  type t = Scalar_bin.t

  let name = "Lt_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"lt_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    let lt v s = S.select (S.lt v s) (S.const 1.) (S.const 0.)
    let pixel ~scalar x out = B.pixel ~combine:lt ~scalar x out
  end
end

(* [where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor]:
   [self] where [condition] holds, the scalar elsewhere. [condition] and [self]
   broadcast against each other (a rank-0 [self] fans out over the mask, which
   is how the mask constant is built). The result is a float tensor; the scalar
   is the f32 value the serialized graph carries, so [-FLT_MAX] survives. *)
module Where_scalar_other = struct
  type t = { condition : Tensor_ref.t; scalar : float; x : Tensor_ref.t }

  let name = "Where_scalar_other"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        {
          condition = Json_util.req_field ms "condition" Tensor_ref.jsont name;
          scalar = Json_util.req_field ms "scalar" Json_util.f32_jsont name;
          x = Json_util.req_field ms "x" Tensor_ref.jsont name;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("condition", Json_util.enc Tensor_ref.jsont t.condition);
            ("scalar", Json_util.enc Json_util.f32_jsont t.scalar);
            ("x", Json_util.enc Tensor_ref.jsont t.x);
          ])
      Jsont.json

  let operands (t : t) = [ t.condition; t.x ]
  let map_operands f (t : t) = { t with condition = f t.condition; x = f t.x }

  let pp pp_ref fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>where_scalar_other@ condition=%a@ x=%a@ scalar=%a@]"
      pp_ref t.condition pp_ref t.x Fmt.float t.scalar

  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let pixel ~scalar ~condition_shape ~x_shape condition x out =
      B.pixel
        ~combine:(fun c v -> S.select (S.eq c (S.const 0.)) (S.const scalar) v)
        ~a_shape:condition_shape ~b_shape:x_shape condition x out
  end
end

(* [where.self(Tensor condition, Tensor self, Tensor other) -> Tensor]:
   [self] where the condition is nonzero, [other] elsewhere, all three
   broadcasting. The two branches share a format -- float32, or int64 when the
   graph selects between integer tensors, as T5's bucket table does; Direct
   evaluates the int64 case exactly. *)
module Where_self = struct
  type t = { condition : Tensor_ref.t; x : Tensor_ref.t; y : Tensor_ref.t }

  let name = "Where_self"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k = Json_util.req_field ms k Tensor_ref.jsont name in
        { condition = get "condition"; x = get "x"; y = get "y" })
      ~enc:(fun t ->
        let r = Json_util.enc Tensor_ref.jsont in
        Json_util.jobj
          [ ("condition", r t.condition); ("x", r t.x); ("y", r t.y) ])
      Jsont.json

  let operands (t : t) = [ t.condition; t.x; t.y ]

  let map_operands f (t : t) =
    { condition = f t.condition; x = f t.x; y = f t.y }

  let pp pp_ref fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>where_self@ condition=%a@ x=%a@ y=%a@]" pp_ref
      t.condition pp_ref t.x pp_ref t.y

  let output_shape condition_shape x_shape y_shape =
    let open Err.Syntax in
    let* xy = broadcast_output_shape x_shape y_shape in
    broadcast_output_shape condition_shape xy

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel ~condition_shape ~x_shape ~y_shape condition x y
        (out : Semantics.position S.index Vec6.t) =
      let read shape t =
        S.load t (broadcast_coord ~index_zero:S.index_zero shape out)
      in
      S.select
        (S.eq (read condition_shape condition) (S.const 0.))
        (read y_shape y) (read x_shape x)
  end
end

(* [min.other(Tensor self, Tensor other) -> Tensor], the elementwise minimum:
   a NaN in either operand is the result, as in ATen, and the two zeros agree.
   Int64 operands are compared exactly by Direct. *)
module Min_other = struct
  type t = Bin.t

  let name = "Min_other"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"min_other" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    (* Neither strictly below the other and not equal means a NaN: [a + b]
       carries it. *)
    let min a b =
      S.select (S.lt a b) a
        (S.select (S.lt b a) b (S.select (S.eq a b) a (S.add a b)))

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:min ~a_shape ~b_shape a b out
  end
end

(* [full_like(Tensor self, Scalar fill_value)] and [zeros_like(Tensor self)]:
   a constant with [self]'s shape and format. [zeros_like] is [full_like] with
   a zero fill, the same value it denotes. An int64 fill must be a whole number
   the f32 payload holds exactly. Reads no value of [self]. *)
module Full_like = struct
  type t = { x : Tensor_ref.t; value : float }

  let name = "Full_like"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        {
          x = Json_util.req_field ms "x" Tensor_ref.jsont name;
          value = Json_util.req_field ms "value" Json_util.f32_jsont name;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("x", Json_util.enc Tensor_ref.jsont t.x);
            ("value", Json_util.enc Json_util.f32_jsont t.value);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp pp_ref fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>full_like@ x=%a@ value=%a@]" pp_ref t.x Fmt.float
      t.value

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel ~value (_ : Semantics.position S.index Vec6.t) = S.const value
  end
end
