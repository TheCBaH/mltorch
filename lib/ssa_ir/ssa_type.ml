(* The scalar domains. [Index] is the language's valid 32-bit position and
   [I64] an exact two's-complement integer. [F32] and [F64] are different types
   even where an interpreter holds both in an OCaml [float]. No scalar is a byte
   offset: byte addressing is formed below this IR, by Machine IR lowering. *)
type scalar = F32 | F64 | I64 | Index | Pred

(* The lane count of a vector or mask. A domain of its own: no extent, position
   or id can be passed for it. *)
module Lanes = struct
  include
    Core.Tagged_int.Make
      (struct
        let prefix = "x"
      end)
      ()

  (* The first [n] items of a list and the rest: one register's lanes. *)
  (* The count as the decimal text a generated source embeds. *)
  let decimal n = string_of_int (to_int n)

  let split n l =
    let n = to_int n in
    (List.filteri (fun i _ -> i < n) l, List.filteri (fun i _ -> i >= n) l)
end

(* A width in bits: a scalar's, or a vector register's. *)
module Bits =
  Core.Tagged_int.Make
    (struct
      let prefix = "b"
    end)
    ()

(* One register of a vector held in several: its position among them. A lane's
   register and a register's first lane are the two directions of one
   division by the lanes a register holds. *)
module Chunk = struct
  include
    Core.Tagged_int.Make
      (struct
        let prefix = "chunk"
      end)
      ()

  (* The registers [lanes] lanes take at [per] lanes each, in order. A count
     that is not a multiple leaves the last register partial and unnamed: the
     caller refuses it first. *)
  let all lanes ~per = List.init (Lanes.to_int lanes / Lanes.to_int per) of_int
  let map lanes ~per f = List.map f (all lanes ~per)
end

(* One lane's position within a vector: a different domain from the count of
   lanes, so an extract's lane cannot be passed for a width. *)
module Lane = struct
  include
    Core.Tagged_int.Make
      (struct
        let prefix = "lane"
      end)
      ()

  let zero = of_int 0

  (* Every position of a vector of [lanes] lanes, in order. *)
  let all lanes = List.init (Lanes.to_int lanes) of_int

  (* The position [by] lanes past [k]. *)
  let add k ~by = of_int (to_int k + to_int by)

  (* The register holding lane [k], and its position inside that register. *)
  let chunk k ~per = Chunk.of_int (to_int k / Lanes.to_int per)
  let slot k ~per = of_int (to_int k mod Lanes.to_int per)

  (* The first lane of register [c]. *)
  let first c ~per = of_int (Chunk.to_int c * Lanes.to_int per)

  (* The element offset of lane [k] at [stride] elements between lanes. *)
  let offset k ~stride = Int64.mul (Int64.of_int (to_int k)) stride
end

(* The widest vector the IR names. A logical width, not a register: a target
   maps it onto however many registers it takes. *)
let max_lanes = Lanes.of_int 64

(* [Effect] is a compile-time sequencing token with no machine representation. *)
type t =
  | Effect
  | Local
  | Mask of Lanes.t
  | Scalar of scalar
  | Vec of scalar * Lanes.t

(* The width of one lane in bits, which decides how many lanes one register
   holds. A predicate is a single truth value: as a vector it is a mask, whose
   lanes take the width of the values it compares. *)
let bits = function
  | F32 -> Bits.of_int 32
  | F64 -> Bits.of_int 64
  | I64 -> Bits.of_int 64
  | Index -> Bits.of_int 32
  | Pred -> Bits.of_int 1

let scalar_name = function
  | F32 -> "f32"
  | F64 -> "f64"
  | I64 -> "i64"
  | Index -> "index"
  | Pred -> "pred"

let equal a b =
  match (a, b) with
  | Effect, Effect | Local, Local -> true
  | Mask a, Mask b -> Lanes.equal a b
  | Scalar a, Scalar b -> a = b
  | Vec (s, a), Vec (t, b) -> s = t && Lanes.equal a b
  | (Effect | Local | Mask _ | Scalar _ | Vec _), _ -> false

let pp fmt = function
  | Effect -> Fmt.string fmt "effect"
  | Local -> Fmt.string fmt "local"
  | Mask l -> Fmt.pf fmt "mask<%a>" Lanes.pp l
  | Scalar s -> Fmt.string fmt (scalar_name s)
  | Vec (s, l) -> Fmt.pf fmt "vec<%a,%s>" Lanes.pp l (scalar_name s)

(* The phantom markers of the builder's typed handles. *)
type chain = Chain_marker
type f32 = F32_marker
type f64 = F64_marker
type i64 = I64_marker
type index = Index_marker
type local = Local_marker
type pred = Pred_marker
