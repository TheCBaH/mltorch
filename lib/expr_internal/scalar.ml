(* Carrier witnesses for the typed scalar language: ['a t] denotes the OCaml
   type a typed expression evaluates to. Exhaustive over exactly the carriers
   this delivery admits -- Bool, Float, I64 (Int32/half are future
   milestones, see .ai/) -- so adding one forces every match here and at each
   call site to be revisited, rather than silently falling through a
   wildcard. [Float] is the working-float domain; a storage width (F16, F32,
   BF16, F64) is a separate witness that decodes to it, not a fourth carrier
   of its own. *)

type _ t = Bool : bool t | Float : float t | I64 : int64 t

let name : type a. a t -> string = function
  | Bool -> "bool"
  | Float -> "float"
  | I64 -> "i64"

let pp fmt s = Fmt.string fmt (name s)

(* Leibniz equality: matching [Some Refl] tells the compiler that two
   previously independent type variables are the same type, refining every
   subsequent use of either. A boolean name comparison cannot provide this
   proof -- see .ai/ for why the distinction matters at heterogeneous
   boundaries below. *)
type (_, _) eq = Refl : ('a, 'a) eq

let equal : type a b. a t -> b t -> (a, b) eq option =
 fun a b ->
  match (a, b) with
  | Bool, Bool -> Some Refl
  | Float, Float -> Some Refl
  | I64, I64 -> Some Refl
  | Bool, _ | Float, _ | I64, _ -> None

(* Existential package for heterogeneous collections (graph bindings, packed
   expressions) whose entries can each have a different carrier. [unpack] is
   the only way to recover a concrete ['a]: without a matched [Refl] the
   existential's own type variable cannot be related to the caller's
   requested ['a], so there is no ill-typed way to "unpack without checking". *)
type packed = Pack : 'a t * 'a -> packed

let unpack : type a. a t -> packed -> a option =
 fun expected (Pack (actual, value)) ->
  match equal expected actual with Some Refl -> Some value | None -> None
