(* Binary pointwise ops split out of pointwise.ml see pointwise.ml (the facade). Includes the broadcasting helpers and the
   Binary/Scalar_binary functors, since only the concrete ops
   (Add/Sub/Mul/Div/.../Pow) are part of pointwise.ml's external surface. *)

(* Output shape of a binary broadcasting op. Per axis the two extents must be
   equal, or one of them must be 1 (the broadcast axis, which takes the other's
   extent); any other mismatch is incompatible and an error — NOT silently the
   larger of the two. Computed in extent-space (no [:> int] round-trips); start
   from [a_shape] and overwrite every axis. See .ai/native_compute_design.md §2b. *)
let broadcast_output_shape (a_shape : Vec6.shape) (b_shape : Vec6.shape) =
  let open Err.Syntax in
  Err.List.fold_left
    (fun s axis ->
      let a = Vec6.get a_shape axis and b = Vec6.get b_shape axis in
      let* out =
        if Dim.equal a b then Err.return a
        else if Dim.equal a Dim.one then Err.return b
        else if Dim.equal b Dim.one then Err.return a
        else
          Err.fail (`Broadcast Shape_error.Broadcast.{ axis; lhs = a; rhs = b })
      in
      Err.return (Vec6.set s axis out))
    a_shape Axis.all

(* Broadcasting for a binary op. [load] is strict — an out-of-bounds index is an
   error, never a silent fan-out — so an operand with an extent-1 (broadcast) axis
   must have the output coord reduced to a valid read of it FIRST: [broadcast_coord
   shape out_vec] maps every axis whose source extent is 1 to [index_zero] and
   keeps [out_vec]'s value elsewhere, so a single stored value is read at index 0
   on a broadcast axis regardless of where the output iterates. The decision is a
   static per-axis shape test, independent of the index value, so the same helper
   serves [Direct] (int indices) and [Symbolic] (index expressions) — it only
   needs [index_zero]. [Vec6.mapi], not a closure: the caller already has (or can
   share) a materialized [Vec6.t], and this returns one directly, ready for
   [SEMANTICS.load]. See .ai/native_tensor_design.md §1b. *)
let broadcast_coord ~(index_zero : 'i) (shape : Vec6.shape)
    (out_vec : 'i Vec6.t) : 'i Vec6.t =
  Vec6.mapi
    (fun a i -> if Dim.equal (Vec6.get shape a) Dim.one then index_zero else i)
    out_vec

(* A binary elementwise op: read each operand at the output coord reduced against
   that operand's own shape ([broadcast_coord]), so an extent-1 axis fans out
   without ever handing [load] an out-of-bounds index. [combine] is the scalar op
   (S.add, S.mul, …). *)
module Binary (S : Semantics.SEMANTICS) = struct
  let pixel ~combine ~a_shape ~b_shape a b
      (out : Semantics.position S.index Vec6.t) =
    let read shape t =
      S.load t (broadcast_coord ~index_zero:S.index_zero shape out)
    in
    combine (read a_shape a) (read b_shape b)
end

(* Exact int64 counterpart of [Binary]: reads each operand through
   [T.i64_load] instead of [S.load], so a broadcasted binary op over two I64
   tensors never round-trips through the engine's f32 compute domain (unlike
   [S.load], which reads every format via [Payload.get_float] and is
   therefore lossy above 2^53 -- the same defect class [Reshape.Compute_i64]/
   [Permute.Compute_i64] fixed for their own ops). [broadcast_coord] itself is
   carrier-independent (built from [S.index_zero]/[Vec6.mapi] alone, no
   [load]/[const] of its own), so it is reused directly rather than re-derived
   under [T]. [T] is a deliberately narrow inline signature, matching
   [Reshape.Compute_i64]'s own precedent -- see the implementation tracker's
   P5.2 note for why a full [Semantics.TYPED_SEMANTICS] functor parameter
   cannot match [Direct]/[Symbolic] here. *)
module Binary_i64
    (S : Semantics.SEMANTICS)
    (T : sig
      type 'a repr

      val i64_load : S.input -> Semantics.position S.index Vec6.t -> int64 repr
    end) =
struct
  let pixel ~combine ~a_shape ~b_shape a b
      (out : Semantics.position S.index Vec6.t) =
    let read shape t =
      T.i64_load t (broadcast_coord ~index_zero:S.index_zero shape out)
    in
    combine (read a_shape a) (read b_shape b)
end

(* Payload shared by the binary pointwise ops [Add]/[Mul]: two operand refs. Each
   op aliases its [t] to this so the [a]/[b] labels are defined once (avoiding
   cross-op label ambiguity) and the serialise/dataflow/pp boilerplate is written
   once, parameterised only by the JSON case [name] and the printed [op] keyword. *)
module Bin = struct
  type t = { a : Tensor_ref.t; b : Tensor_ref.t }

  let jsont ~name : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k = Json_util.req_field ms k Tensor_ref.jsont name in
        { a = get "a"; b = get "b" })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        Json_util.jobj [ ("a", ref_ t.a); ("b", ref_ t.b) ])
      Jsont.json

  let operands (t : t) = [ t.a; t.b ]
  let map_operands f (t : t) = { a = f t.a; b = f t.b }

  let pp ~op (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>%s@ a=%a@ b=%a@]" op pp_ref t.a pp_ref t.b

  (* Walk config space shared by the binary pointwise ops (Add/Mul): one shape
     used for both operands (equal-shape; broadcast not exercised). A functor
     over the global Limits; no constraint, so [cascade] is identity. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun _ s -> { shape = s });
        ]

    let pp fmt (c : cfg) = Walk_core.Shape.pp fmt c.shape
  end
end

(* Payload shared by the scalar-operand pointwise ops [Add_scalar]/[Div_scalar]:
   one operand ref and one compile-time scalar. The exporter serialises a
   compile-time scalar straight into a Tensor-typed slot (MobileNet-v3's
   hardsigmoid is `add(x, 3)` / `div(x, 6)` with `as_int` arguments), and the
   ATen interp materialises those with [full_like]. Here the scalar stays a
   graph parameter instead: no extra edge to bind, and the symbolic path gets a
   [const] leaf rather than a load, which is what a later fusion pass wants.

   [scalar] is f32-canonical by construction (the builder narrows it), because
   the engine's tensors are F32 and a scalar that is not f32-exact would
   otherwise round differently here than under the ATen path's [full_like] on
   an F32 tensor. *)
module Scalar_bin = struct
  type t = { x : Tensor_ref.t; scalar : float }

  let jsont ~name : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        {
          x = Json_util.req_field ms "x" Tensor_ref.jsont name;
          scalar = Json_util.req_field ms "scalar" Json_util.f32_jsont name;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("x", Json_util.enc Tensor_ref.jsont t.x);
            ("scalar", Json_util.enc Json_util.f32_jsont t.scalar);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp ~op (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>%s@ x=%a@ scalar=%a@]" op pp_ref t.x Fmt.float t.scalar

  (* Walk config shared by the scalar pointwise ops: one shape plus a scalar
     drawn from a finite candidate set. Zero is deliberately absent so the same
     space is safe for [Div_scalar] — unlike [Div], whose random *tensor*
     divisor is what makes it unwalkable. Explicit zero behaviour is pinned by
     the direct unit tests instead. The list mixes integer-like values (what
     MobileNet actually serialises), a negative, and an f32-inexact value. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; scalar : float }

    let candidates = [ 3.; 6.; -2.; 0.5; 0.1 ]

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 };
        scalar = 3.;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "scalar" candidates (fun c v -> { c with scalar = v });
        ]

    let pp fmt (c : cfg) =
      Fmt.pf fmt "%a scalar=%a" Walk_core.Shape.pp c.shape Fmt.float c.scalar
  end
end

(* A scalar elementwise op: the operand already has the output shape, so it is
   read at [out] directly (no broadcast reduction), and the scalar becomes a
   [const] in the value domain. *)
module Scalar_binary (S : Semantics.SEMANTICS) = struct
  let pixel ~combine ~scalar x (out : Semantics.position S.index Vec6.t) =
    combine (S.load x out) (S.const scalar)
end

module Add = struct
  type t = Bin.t

  let name = "Add"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"add" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:S.add ~a_shape ~b_shape a b out
  end

  (* Exact int64 counterpart of [Compute]: both operands are read through
     [T.i64_load]/[T.i64_binary] instead of [S.load]/[S.add], so a broadcasted
     I64 add never round-trips through the engine's f32 domain. Output stays
     [int64 repr], unlike [Mul_scalar.Compute_i64] (which promotes to float by
     design) -- so, unlike that op, this DOES need its output edge's declared
     format threaded to I64 by the builder; see [Graph_builder.add]. *)
  module Compute_i64
      (S : Semantics.SEMANTICS)
      (T : sig
        type 'a repr

        val i64_load :
          S.input -> Semantics.position S.index Vec6.t -> int64 repr

        val i64_binary :
          Expr.Value.i64_binary_op -> int64 repr -> int64 repr -> int64 repr
      end) =
  struct
    module B = Binary_i64 (S) (T)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel
        ~combine:(T.i64_binary Expr.Value.I64_add)
        ~a_shape ~b_shape a b out
  end
end

module Add_scalar = struct
  type t = Scalar_bin.t

  let name = "Add_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"add_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    let pixel ~scalar x out = B.pixel ~combine:S.add ~scalar x out
  end
end

module Div = struct
  type t = Bin.t

  let name = "Div"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"div" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    (* A zero divisor is the caller's business, exactly as it is for the
       underlying float division: no guard here, so the result is the IEEE
       infinity or NaN rather than a silently substituted value. *)
    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:S.div ~a_shape ~b_shape a b out
  end
end

module Div_scalar = struct
  type t = Scalar_bin.t

  let name = "Div_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"div_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    (* As in [Div], a zero divisor is the caller's business: the result is the
       IEEE infinity or NaN, not a silently substituted value. Unlike [Div] this
       op is still walkable, because its divisor comes from a fixed candidate
       list that omits zero rather than from a random tensor. *)
    let pixel ~scalar x out = B.pixel ~combine:S.div ~scalar x out
  end
end

(* [addcmul(self, tensor1, tensor2, value)] is a ternary broadcasted pointwise
   op, not a graph-level [Mul] followed by [Add].  Keeping all three loads in
   one pixel expression preserves the fused-kernel boundary. *)
module Addcmul = struct
  type t = {
    self : Tensor_ref.t;
    tensor1 : Tensor_ref.t;
    tensor2 : Tensor_ref.t;
    value : float;
  }

  let name = "Addcmul"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k = Json_util.req_field ms k Tensor_ref.jsont name in
        {
          self = get "self";
          tensor1 = get "tensor1";
          tensor2 = get "tensor2";
          value = Json_util.req_field ms "value" Json_util.f32_jsont name;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        Json_util.jobj
          [
            ("self", ref_ t.self);
            ("tensor1", ref_ t.tensor1);
            ("tensor2", ref_ t.tensor2);
            ("value", Json_util.enc Json_util.f32_jsont t.value);
          ])
      Jsont.json

  let operands t = [ t.self; t.tensor1; t.tensor2 ]

  let map_operands f t =
    { t with self = f t.self; tensor1 = f t.tensor1; tensor2 = f t.tensor2 }

  let pp pp_ref fmt t =
    Fmt.pf fmt "@[<hv 2>addcmul@ self=%a@ tensor1=%a@ tensor2=%a@ value=%a@]"
      pp_ref t.self pp_ref t.tensor1 pp_ref t.tensor2 Fmt.float t.value

  let output_shape self_shape tensor1_shape tensor2_shape =
    let open Err.Syntax in
    let* partial = broadcast_output_shape self_shape tensor1_shape in
    broadcast_output_shape partial tensor2_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel ~self_shape ~tensor1_shape ~tensor2_shape ~value self tensor1
        tensor2 (out : Semantics.position S.index Vec6.t) =
      let read shape t =
        S.load t (broadcast_coord ~index_zero:S.index_zero shape out)
      in
      S.add (read self_shape self)
        (S.mul (S.const value)
           (S.mul (read tensor1_shape tensor1) (read tensor2_shape tensor2)))
  end
end

(* [div.Tensor_mode(self, other, rounding_mode="floor")] with a compile-time
   [other] -- the corpus's only instance (EdgeNeXt's Fourier positional
   encoding divides a position index by 2, flooring). A genuinely different
   overload from [Div_scalar]: real division there, floor-of-the-quotient
   here. Not scoped for ["trunc"]/[None] rounding or a tensor [other] -- no
   corpus evidence for either, so [Op_bridge] rejects them rather than
   guessing. *)
module Floor_div_scalar = struct
  type t = Scalar_bin.t

  let name = "Floor_div_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"floor_div_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    (* [S.trunc] rounds toward zero; ATen's floor rounding mode rounds toward
       negative infinity. The two agree whenever the true quotient [q] is
       already an integer or nonnegative; the correction [- 1] fires exactly
       when truncation rounded UP relative to the floor, i.e. when [q < trunc
       q] (a negative, non-integral quotient). *)
    let floor_div v s =
      let q = S.div v s in
      let t = S.trunc q in
      S.sub t (S.select (S.lt q t) (S.const 1.) (S.const 0.))

    let pixel ~scalar x out = B.pixel ~combine:floor_div ~scalar x out
  end
end

(* [eq.Scalar(self, other) -> self == other]. Uses [SEMANTICS.eq] (P6.4),
   NOT [S.lt]-composed: [S.eq] is IEEE numerical equality -- NaN unequal to
   everything including itself, signed zeros equal -- which is exactly real
   ATen's [eq] on a float operand, and which no [S.lt]/[S.select] composition
   can express (see [Semantics.eq]'s own doc comment and the P6.3/P6.4 NaN
   finding this op deliberately avoids repeating). Same two-layer split as
   [Gt_scalar] below: the output VALUE here is still float 0./1.
   ([Compute] is [SEMANTICS]-generic, shared with [Symbolic]);
   [Graph_builder.eq_scalar] declares the edge [Bool] and [Eval_direct]
   writes genuine [Payload.Bool] storage from this same formula. *)
module Eq_scalar = struct
  type t = Scalar_bin.t

  let name = "Eq_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"eq_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    let eq v s = S.select (S.eq v s) (S.const 1.) (S.const 0.)
    let pixel ~scalar x out = B.pixel ~combine:eq ~scalar x out
  end
end

(* [ne.Scalar(self, other) -> self != other]. Real ATen's [ne] is [eq]'s
   logical negation -- reuses the same [SEMANTICS.eq] primitive [Eq_scalar]
   above uses, just a negated [S.select] arm order, rather than an
   [S.lt]-composed formula which cannot express IEEE equality's
   NaN-unequal-to-everything case (a NaN operand must read [ne] TRUE, the
   same as every other comparison against it). Same two-layer split as
   [Eq_scalar]/[Gt_scalar]: the output VALUE here is still float 0./1.
   ([Compute] is [SEMANTICS]-generic, shared with [Symbolic]);
   [Graph_builder.ne_scalar] declares the edge [Bool] and [Eval_direct]
   writes genuine [Payload.Bool] storage from this same formula. *)
module Ne_scalar = struct
  type t = Scalar_bin.t

  let name = "Ne_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"ne_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    let ne v s = S.select (S.eq v s) (S.const 0.) (S.const 1.)
    let pixel ~scalar x out = B.pixel ~combine:ne ~scalar x out
  end
end

(* [gt.Scalar(self, other) -> self > other]. Needs no new [SEMANTICS]
   primitive: built from the same [S.lt]/[S.select] pair [Bitwise_not.
   Compute]'s own nonzero-test formula already uses (see pointwise_unary.ml),
   just one ordered comparison instead of two. IEEE ordering falls out of
   [S.lt] for free -- NaN is never [<] anything, so a NaN operand on either
   side makes this false, matching real ATen's [gt] on NaN. The output
   VALUE here is still float 0./1. ([Compute] is [SEMANTICS]-generic, shared
   with [Symbolic]); [Graph_builder.gt_scalar] declares the edge [Bool] and
   [Eval_direct] writes genuine [Payload.Bool] storage from this same
   formula, mirroring [Bitwise_not]'s own two-layer split. *)
module Gt_scalar = struct
  type t = Scalar_bin.t

  let name = "Gt_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"gt_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    let gt v s = S.select (S.lt s v) (S.const 1.) (S.const 0.)
    let pixel ~scalar x out = B.pixel ~combine:gt ~scalar x out
  end
end

(* [eq.Tensor(self, other) -> self == other]. Tensor-tensor form of
   [Eq_scalar]: same [SEMANTICS.eq] primitive (P6.4's IEEE numerical
   equality -- NaN unequal to everything, signed zeros equal), broadcast via
   [Binary] instead of [Scalar_binary] since BOTH operands are runtime
   tensors rather than one tensor and one compile-time scalar. The output
   VALUE here is still float 0./1. ([Compute] is [SEMANTICS]-generic, shared
   with [Symbolic]); [Graph_builder.eq_tensor] declares the edge [Bool] and
   [Eval_direct] writes genuine [Payload.Bool] storage from this same
   formula. *)
module Eq_tensor = struct
  type t = Bin.t

  let name = "Eq_tensor"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"eq_tensor" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let eq a b = S.select (S.eq a b) (S.const 1.) (S.const 0.)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:eq ~a_shape ~b_shape a b out
  end
end

(* [ne.Tensor(self, other) -> self != other]. Tensor-tensor form of
   [Ne_scalar]: [Eq_tensor]'s own formula with the [S.select] arms swapped,
   the same negation relationship [Ne_scalar] has to [Eq_scalar]. *)
module Ne_tensor = struct
  type t = Bin.t

  let name = "Ne_tensor"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"ne_tensor" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let ne a b = S.select (S.eq a b) (S.const 0.) (S.const 1.)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:ne ~a_shape ~b_shape a b out
  end
end

module Mul = struct
  type t = Bin.t

  let name = "Mul"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"mul" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:S.mul ~a_shape ~b_shape a b out
  end

  (* Exact int64 counterpart of [Compute]; see [Add.Compute_i64] for the
     rationale, identical here down to the output-edge threading requirement.
     This is [Mul]'s tensor-tensor form -- not to be confused with
     [Mul_scalar.Compute_i64] below, whose output promotes to float by
     design. *)
  module Compute_i64
      (S : Semantics.SEMANTICS)
      (T : sig
        type 'a repr

        val i64_load :
          S.input -> Semantics.position S.index Vec6.t -> int64 repr

        val i64_binary :
          Expr.Value.i64_binary_op -> int64 repr -> int64 repr -> int64 repr
      end) =
  struct
    module B = Binary_i64 (S) (T)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel
        ~combine:(T.i64_binary Expr.Value.I64_mul)
        ~a_shape ~b_shape a b out
  end
end

module Mul_scalar = struct
  type t = Scalar_bin.t

  let name = "Mul_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"mul_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    let pixel ~scalar x out = B.pixel ~combine:S.mul ~scalar x out
  end

  (* Exact int64-input counterpart of [Compute]: the operand is read through
     [i64_load]/[i64_to_float] -- an explicit, checked promotion -- rather
     than [Compute]'s [S.load], which round-trips every format through
     [Payload.get_float] and would perform the identical promotion only
     incidentally. [Payload.get_float]'s I64 case is exactly [Int64.to_float]
     (payload.ml), so this changes no value this op ever returns; it only
     makes the cast an explicit step, per the plan's "integer-to-float is an
     explicit expression cast" invariant. No coordinate math, unlike
     [Reshape.Compute_i64]/[Permute.Compute_i64]: the operand already has the
     output shape, so it is read at [out] directly, matching [Scalar_binary]
     above. The output stays [S.t] (float), not [int64 repr]: unlike
     Reshape/Permute, [Mul_scalar]'s output format is F32 by design (ATen
     promotes an integer tensor times a float scalar to a float result), so
     there is no output-format branch to preserve here -- only the read. *)
  module Compute_i64
      (S : Semantics.SEMANTICS)
      (T : sig
        type 'a repr

        val i64_load :
          S.input -> Semantics.position S.index Vec6.t -> int64 repr

        val i64_to_float : int64 repr -> S.t
      end) =
  struct
    let pixel ~scalar x (out : Semantics.position S.index Vec6.t) =
      S.mul (T.i64_to_float (T.i64_load x out)) (S.const scalar)
  end
end

module Pow = struct
  (* [x ** scalar] -- ATen's `aten.pow.Tensor_Scalar`. [scalar] is a
     compile-time constant: an ATen [Scalar] argument is never wired from
     another node's output (a tensor-valued exponent traces to
     `pow.Tensor_Tensor`, a different node this op does not cover), and the
     bridge/importer reject anything else as a decode error. So the branch
     below runs once per graph node, in OCaml, not once per pixel -- mirroring
     ATen's own [PowKernel.cpp], which special-cases exactly these six
     exponents (sqrt/rsqrt/reciprocal/square/cube/reciprocal-of-square) via
     [sqrt_kernel]/[rsqrt_kernel]/[reciprocal_kernel]/multiplication rather
     than [std::pow], for the same accuracy reason -- not a per-element
     runtime check either engine has to make. Any other exponent falls back
     to [exp(scalar * log x)], which -- unlike the six special cases -- is
     only [Equivalent] to ATen's [std::pow], not bit-identical, and (like
     [Sqrt]) is only correct for x > 0: [log] of a non-positive base is not
     what [std::pow] returns for a negative base with a non-integer exponent
     (NaN either way) or a base of exactly 0 ([std::pow(0, e)] is
     well-defined; [log 0] is not). *)
  type t = Scalar_bin.t

  let name = "Pow"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"pow" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel ~scalar x (out : Semantics.position S.index Vec6.t) =
      let v = S.load x out in
      let reciprocal v = S.div (S.const 1.) v in
      if scalar = 2.0 then S.mul v v
      else if scalar = 3.0 then S.mul (S.mul v v) v
      else if scalar = -2.0 then reciprocal (S.mul v v)
      else if scalar = 0.5 then S.sqrt v
      else if scalar = -0.5 then reciprocal (S.sqrt v)
      else if scalar = -1.0 then reciprocal v
      else S.exp (S.mul (S.const scalar) (S.log v))
  end
end

(* [scalar ** x] -- ATen's `aten.pow.Scalar`, the reverse of [Pow]'s
   [x ** scalar]: the tensor operand is the EXPONENT, and the compile-time
   constant is the base (EdgeNeXt's Fourier positional encoding raises
   [10000.0] to a per-element frequency exponent). [exp (x * log scalar)] is
   the only general expression available, computing [log scalar] once at
   decode time rather than per pixel; correct for a positive [scalar] (the
   corpus's only instance is [10000.0]) -- a nonpositive base hits the same
   undefined-[log] case [Pow]'s own fallback already flags. *)
module Rpow_scalar = struct
  type t = Scalar_bin.t

  let name = "Rpow_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"rpow_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel ~scalar x (out : Semantics.position S.index Vec6.t) =
      let v = S.load x out in
      S.exp (S.mul v (S.const (Float.log scalar)))
  end
end

(* `rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor` computes
   [other - alpha * self] -- the REVERSE of [sub.Tensor]'s scalar form
   ([self - alpha * other]), and genuinely its own op rather than a
   legalization onto [Add_scalar]/[Mul_scalar]: composing those two would
   decompose one ATen node into two Native ones, the exact failure mode
   .ai/native_add_op.md's design goal rules out (unlike [sub.Tensor]'s own
   scalar form, which legalizes to [Add_scalar] alone with a negated scalar
   -- ONE node, not two). [other]/[alpha] are both compile-time constants,
   the same discipline [Pow]'s [scalar] follows: an ATen [Scalar] argument is
   never wired from another node's output. *)
module Rsub_scalar = struct
  type params = { other : float; alpha : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"rsub_scalar_params" (fun other alpha ->
        { other; alpha })
    |> Jsont.Object.mem "other" Json_util.f32_jsont ~enc:(fun p -> p.other)
    |> Jsont.Object.mem "alpha" Json_util.f32_jsont ~enc:(fun p -> p.alpha)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{other=%a;@ alpha=%a}@]" Fmt.float p.other Fmt.float
      p.alpha

  type t = { params : params; x : Tensor_ref.t }

  let name = "Rsub_scalar"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        {
          params = Json_util.req_field ms "params" params_jsont name;
          x = Json_util.req_field ms "x" Tensor_ref.jsont name;
        })
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
    Fmt.pf fmt "@[<hv 2>rsub_scalar@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) x (out : Semantics.position S.index Vec6.t) =
      S.sub (S.const p.other) (S.mul (S.const p.alpha) (S.load x out))
  end
end

module Sub = struct
  type t = Bin.t

  let name = "Sub"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"sub" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:S.sub ~a_shape ~b_shape a b out
  end

  (* Exact int64 counterpart of [Compute]; see [Add.Compute_i64] for the
     rationale, identical here down to the output-edge threading requirement. *)
  module Compute_i64
      (S : Semantics.SEMANTICS)
      (T : sig
        type 'a repr

        val i64_load :
          S.input -> Semantics.position S.index Vec6.t -> int64 repr

        val i64_binary :
          Expr.Value.i64_binary_op -> int64 repr -> int64 repr -> int64 repr
      end) =
  struct
    module B = Binary_i64 (S) (T)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel
        ~combine:(T.i64_binary Expr.Value.I64_sub)
        ~a_shape ~b_shape a b out
  end
end
