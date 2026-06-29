(* Pointwise ops: each output pixel reads the input(s) at the same coord. The
   functor is over SEMANTICS, so the same definition serves Direct and Symbolic.
   [out] gives the output coord as one index expression per axis.
   See .ai/native_compute_design.md §2. *)

(* Output shape of a binary broadcasting op. Per axis the two extents must be
   equal, or one of them must be 1 (the broadcast axis, which takes the other's
   extent); any other mismatch is incompatible and an error — NOT silently the
   larger of the two. Computed in extent-space (no [:> int] round-trips); start
   from [a_shape] and overwrite every axis. See .ai/native_compute_design.md §2b. *)
let broadcast_output_shape (a_shape : Vec6.shape) (b_shape : Vec6.shape) =
  List.fold_left
    (fun s axis ->
      let a = Vec6.get a_shape axis and b = Vec6.get b_shape axis in
      let out =
        if Dim.equal a b then a
        else if Dim.equal a Dim.one then b
        else if Dim.equal b Dim.one then a
        else
          invalid_arg
            (Format.asprintf
               "Pointwise.broadcast_output_shape: incompatible extents on axis \
                %a: %a vs %a"
               Axis.pp axis Dim.pp a Dim.pp b)
      in
      Vec6.set s axis out)
    a_shape Axis.all

(* Broadcasting for a binary op. [load] is strict — an out-of-bounds index is an
   error, never a silent fan-out — so an operand with an extent-1 (broadcast) axis
   must have the output coord reduced to a valid read of it FIRST: [broadcast_coord
   shape out] maps every axis whose source extent is 1 to [index_zero] and keeps
   [out a] elsewhere, so a single stored value is read at index 0 on a broadcast
   axis regardless of where the output iterates. The decision is a static per-axis
   shape test, independent of the index value, so the same helper serves [Direct]
   (int indices) and [Symbolic] (index expressions) — it only needs [index_zero].
   See .ai/native_tensor_design.md §1b. *)
let broadcast_coord ~(index_zero : 'i) (shape : Vec6.shape) (out : Axis.t -> 'i)
    : Axis.t -> 'i =
 fun a -> if Dim.equal (Vec6.get shape a) Dim.one then index_zero else out a

module Relu = struct
  type t = { x : Tensor_ref.t }

  let name = "Relu"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        { x = Json_util.req_field ms "x" Tensor_ref.jsont name })
      ~enc:(fun t ->
        Json_util.jobj [ ("x", Json_util.enc Tensor_ref.jsont t.x) ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>relu@ x=%a@]" pp_ref t.x

  (* Identity: relu doesn't change shape. See .ai/native_compute_design.md §2b. *)
  let output_shape (x_shape : Vec6.shape) = x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* relu x = (x < 0 ? 0 : x) — derived from [select]+[lt], not a primitive *)
    let pixel x (out : Axis.t -> Semantics.position S.index) =
      let v = S.load x out in
      S.select (S.lt v (S.const 0.)) (S.const 0.) v
  end
end

(* A binary elementwise op: read each operand at the output coord reduced against
   that operand's own shape ([broadcast_coord]), so an extent-1 axis fans out
   without ever handing [load] an out-of-bounds index. [combine] is the scalar op
   (S.add, S.mul, …). *)
module Binary (S : Semantics.SEMANTICS) = struct
  let pixel ~combine ~a_shape ~b_shape a b
      (out : Axis.t -> Semantics.position S.index) =
    let read shape t =
      S.load t (broadcast_coord ~index_zero:S.index_zero shape out)
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
end
