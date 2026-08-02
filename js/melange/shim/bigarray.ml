(* A [Bigarray] for Melange, which has none: its stdlib simply does not ship the
   module, so anything naming [Bigarray.Array1.t] in a signature cannot compile.
   Being [wrapped false] and named [bigarray.ml], this supplies that name to
   dependents without any of them being edited.

   Scope is deliberately the closed set that jsont and jsont_base touch --
   [Array1.{create,get,unsafe_get,set,dim,init,kind,layout}], the two type
   families, [c_layout], and the kind constructors. It is NOT enough for
   lib/native's [Payload]:

     the backing store here is a plain OCaml array, so a [float32] cell holds a
     full float64 and an [int8_signed] cell does not wrap. That is invisible to
     jsont (whose only Bigarray use is byte buffers, already in range) and wrong
     for [Payload], whose whole point is that storing through a kind rounds and
     truncates.

   Do NOT grow this to cover that. Real kind semantics over TypedArrays --
   including [int64], which melange represents as a pair of 32-bit words rather
   than BigInt -- is melange-re/melange#1807, which adds Bigarray to melange's
   own stdlib. When that lands this file is deleted, not extended. See
   .ai/js_backends_design.md. *)

type float32_elt
type float64_elt
type int8_signed_elt
type int8_unsigned_elt
type int16_signed_elt
type int16_unsigned_elt
type int32_elt
type int64_elt

type ('a, 'b) kind =
  | Float32 : (float, float32_elt) kind
  | Float64 : (float, float64_elt) kind
  | Int8_signed : (int, int8_signed_elt) kind
  | Int8_unsigned : (int, int8_unsigned_elt) kind
  | Int16_signed : (int, int16_signed_elt) kind
  | Int16_unsigned : (int, int16_unsigned_elt) kind
  | Int32 : (int32, int32_elt) kind
  | Int64 : (int64, int64_elt) kind

let float32 = Float32
let float64 = Float64
let int8_signed = Int8_signed
let int8_unsigned = Int8_unsigned
let int16_signed = Int16_signed
let int16_unsigned = Int16_unsigned
let int32 = Int32
let int64 = Int64

type c_layout
type fortran_layout
type 'a layout = C_layout : c_layout layout

let c_layout = C_layout

(* The zero of each kind, so [create] can fill. Matching on the GADT is what
   recovers ['a] from the kind. *)
let zero : type a b. (a, b) kind -> a = function
  | Float32 -> 0.0
  | Float64 -> 0.0
  | Int8_signed -> 0
  | Int8_unsigned -> 0
  | Int16_signed -> 0
  | Int16_unsigned -> 0
  | Int32 -> 0l
  | Int64 -> 0L

module Array1 = struct
  type ('a, 'b, 'c) t = { k : ('a, 'b) kind; l : 'c layout; d : 'a array }

  let create : type a b c. (a, b) kind -> c layout -> int -> (a, b, c) t =
   fun k l n -> { k; l; d = Array.make n (zero k) }

  let init : type a b c.
      (a, b) kind -> c layout -> int -> (int -> a) -> (a, b, c) t =
   fun k l n f -> { k; l; d = Array.init n f }

  let dim t = Array.length t.d
  let kind t = t.k
  let layout t = t.l
  let get t i = t.d.(i)
  let unsafe_get t i = Array.unsafe_get t.d i
  let set t i v = t.d.(i) <- v
end

(* Deliberately absent: [sub], [fill], [unsafe_set], [blit], [of_array], the
   Array2/Array3/Genarray families. There is no .mli here, so anything defined
   is public and a dependency bump could start using it -- and [sub] in
   particular cannot be written correctly over an OCaml array. Real
   [Bigarray.Array1.sub] returns a *view sharing* the parent's storage;
   [Array.sub] copies, so writes through the result would silently stop
   reaching the parent. Better a compile error naming the missing function than
   a shim that quietly means something else. If a future dependency needs one,
   add it then, over Js.Typed_array if sharing matters. *)
