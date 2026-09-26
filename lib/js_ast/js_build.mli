(** Typed builders over the untyped {!Js_ast}. The AST stays simple so the
    printer can; the helpers the lowering calls carry a phantom kind, so mixing
    representations is an OCaml type error. {!expr} reads a value back out. No
    function here turns a raw [Js_ast.expr] into a ['k t], except
    {!Unsafe.call}, the boundary to a named runtime helper. *)

type num
(** A JS [Number] working value: binary64. *)

type idx
(** A JS [Number] holding an index proven to lie in [[-2^31, 2^31)]. Only these
    go in an array subscript, and no operation on them prints a bitwise
    operator. *)

type big
(** A JS [BigInt]: the int64 carrier. *)

type bool_

type bits16
(** A small unsigned bit pattern: a raw f16/bf16 half, or a field cut from one.
*)

type bits32
(** A 32-bit pattern, the result of [Bits.shl16], stored into a [Uint32Array].
*)

type 'k arr
(** A typed array (or array literal) whose elements read as ['k]. *)

type 'k t = private Js_ast.expr

val expr : 'k t -> Js_ast.expr
(** The untyped expression: the only way out, and it loses the kind. *)

(** {1 Float working values}

    [Num] folds nothing: [x + 0] and [x * 1] are not identities on binary64
    ([-0 + 0 = +0]), and the interpreter is the oracle. *)

module Num : sig
  val abs : num t -> num t
  val add : num t -> num t -> num t
  val const : float -> num t
  val cos : num t -> num t
  val div : num t -> num t -> num t
  val eq : num t -> num t -> bool_ t
  val exp : num t -> num t
  val fround : num t -> num t
  val is_finite : num t -> bool_ t
  val is_nan : num t -> bool_ t
  val log : num t -> num t
  val le : num t -> num t -> bool_ t
  val lt : num t -> num t -> bool_ t
  val gt : num t -> num t -> bool_ t
  val max : num t -> num t -> num t
  val mul : num t -> num t -> num t
  val ne : num t -> num t -> bool_ t
  val neg : num t -> num t

  val of_big : big t -> num t
  (** [Number(b)]: rounds to nearest, as [Int64.to_float]. *)

  val of_bits : bits16 t -> num t

  val of_idx : idx t -> num t
  (** [i + 0]: an index [Number] can be [-0], a float value must be [+0]. A
      literal other than [-0] is already its own value and is left bare. *)

  val pow : num t -> num t -> num t
  val sin : num t -> num t
  val sqrt : num t -> num t
  val sub : num t -> num t -> num t
  val trunc : num t -> num t
  val var : Js_ident.t -> num t
end

(** {1 Indices}

    Folding is exactness-preserving only: [add] drops a constant [0] and prints
    [a + -k * b] as [a - k * b] (and [a + -k] as [a - k]), [scale 1 a] is [a],
    and [scale k] of a constant [0] is [0]. [scale 0 a] is {i not} folded to
    [0]: [a] may carry an overflow guard's subexpression the program still
    checks. *)

module Idx : sig
  val add : idx t -> idx t -> idx t
  val ceil_div_pos : idx t -> int -> idx t
  val clamp_low : idx t -> idx t
  val const : int -> idx t
  val eq : idx t -> idx t -> bool_ t
  val floor_div_pos : idx t -> int -> idx t
  val ge : idx t -> idx t -> bool_ t
  val lt : idx t -> idx t -> bool_ t
  val max : idx t -> idx t -> idx t
  val min : idx t -> idx t -> idx t

  val of_big_bounded : big t -> idx t
  (** [Number(b)], the one crossing from [BigInt] to an index. Named for its
      precondition: only after a range check. *)

  val out_of_range : idx t -> int -> bool_ t
  (** [i < 0 || i >= n]. *)

  val outside_int32 : idx t -> bool_ t
  (** [i < -2^31 || i >= 2^31]: the domain [Loop_range] proves an index in. *)

  val scale : int -> idx t -> idx t
  val var : Js_ident.t -> idx t
end

(** {1 int64}

    [add_wrap], [sub_wrap] and [mul_wrap] are the only [big] arithmetic, so a
    missing [BigInt.asIntN] is not a possible state. *)

module Big : sig
  val add_wrap : big t -> big t -> big t
  val const : int64 -> big t

  val div_unchecked : big t -> big t -> big t
  (** The caller emits the zero and [min / -1] guards first. *)

  val eq : big t -> big t -> bool_ t
  val lt : big t -> big t -> bool_ t
  val mul_wrap : big t -> big t -> big t
  val of_idx : idx t -> big t

  val of_num_trunc : num t -> big t
  (** [BigInt(Math.trunc(x))], after its range check. *)

  val sub_wrap : big t -> big t -> big t
  val var : Js_ident.t -> big t
end

(** {1 Bit patterns}

    The only producer of a bitwise operator. *)

module Bits : sig
  val and_ : bits16 t -> bits16 t -> bits16 t
  val const : int -> bits16 t
  val eq : bits16 t -> bits16 t -> bool_ t
  val or_ : bits16 t -> bits16 t -> bits16 t
  val shl16 : bits16 t -> bits32 t
  val shr : bits16 t -> int -> bits16 t
  val var : Js_ident.t -> bits16 t
end

(** {1 Booleans} *)

module Pred : sig
  val false_ : bool_ t
  val not_ : bool_ t -> bool_ t
  val or_ : bool_ t -> bool_ t -> bool_ t
end

val select : bool_ t -> 'k t -> 'k t -> 'k t

(** {1 Arrays} *)

module Arr : sig
  val bits : Js_ident.t -> bits16 arr t
  val big : Js_ident.t -> big arr t
  val idx : Js_ident.t -> idx arr t

  val literal : 'k t list -> 'k arr t
  (** An array literal, e.g. a per-channel quantization table. *)

  val new_float64 : int -> num arr t
  val new_uint32 : int -> bits32 arr t
  val num : Js_ident.t -> num arr t
  val uint32 : Js_ident.t -> bits32 arr t

  val view_float32 : bits32 arr t -> num arr t
  (** [new Float32Array(a.buffer)]: reinterprets the same bytes. *)
end

val load : 'k arr t -> idx t -> 'k t
val store : 'k arr t -> idx t -> 'k t -> Js_ast.stmt

(** {1 Statements}

    Bindings and control flow over kinded values. The kind of a variable is
    fixed by the binder that introduces it. *)

module Stmt : sig
  val assign_big : Js_ident.t -> big t -> Js_ast.stmt
  val assign_idx : Js_ident.t -> idx t -> Js_ast.stmt
  val assign_num : Js_ident.t -> num t -> Js_ast.stmt
  val const_arr : Js_ident.t -> 'k arr t -> Js_ast.stmt
  val const_bits : Js_ident.t -> bits16 t -> Js_ast.stmt
  val const_idx : Js_ident.t -> idx t -> Js_ast.stmt
  val const_num : Js_ident.t -> num t -> Js_ast.stmt

  val decr_num : Js_ident.t -> num t -> Js_ast.stmt
  (** [v -= e]. *)

  val for_ :
    Js_ident.t -> lo:idx t -> hi:idx t -> Js_ast.stmt list -> Js_ast.stmt
  (** [for (let v = lo; v < hi; v++)]. *)

  val if_ : bool_ t -> Js_ast.stmt list -> Js_ast.stmt list -> Js_ast.stmt

  val incr_num : Js_ident.t -> num t -> Js_ast.stmt
  (** [v += e]. *)

  val let_big : Js_ident.t -> big t -> Js_ast.stmt
  val let_idx : Js_ident.t -> idx t -> Js_ast.stmt
  val let_num : Js_ident.t -> num t -> Js_ast.stmt
  val return_ : Js_ast.expr -> Js_ast.stmt
  val return_null : Js_ast.stmt
end

(** {1 Records and calls} *)

val record : (string * Js_ast.expr) list -> Js_ast.expr
(** An object literal, e.g. a failure record. Keys are checked as identifiers.
*)

val string : string -> Js_ast.expr
val bool : bool -> Js_ast.expr

val to_string : 'k t -> Js_ast.expr
(** [(x).toString()] on a [BigInt] or [Number]. *)

module Unsafe : sig
  val call : Js_ident.t -> Js_ast.expr list -> 'k t
  (** A call to a named runtime helper. The caller states the result kind: the
      helper's own typed wrapper is the one place that says it. *)
end
