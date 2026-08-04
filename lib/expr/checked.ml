(* Overflow-checked integer arithmetic for the index domain. Private to the
   library: it is reached through [Expr.Eval] and [Expr.Intrinsic], never
   directly. See .ai/native_expr_refactoring_design.md.

   ---- why [int] and not [int64] ----

   The design's motivation for a wider domain is that UNCHECKED [int] arithmetic
   can wrap before a later bound check. That is answered by checking, not by
   widening: [int64] would not remove the narrowing, only relocate it to every
   host that must produce a Bigarray offset -- the exact defect class CLAUDE.md
   logs six instances of. And [Int64] is an allocating three-field emulation
   under js_of_ocaml, on a path evaluated once per output pixel.

   The predicates below are width-agnostic and therefore self-adapting: under
   js_of_ocaml, where [Sys.int_size] is 32, [max_int] is 2^31-1 and the same
   comparisons fire. That is a STRONGER guarantee than [int64] gives, because it
   rejects at the operation that would wrap rather than at a boundary narrow.
   The consequence -- an index reaching 2^40 is accepted natively and rejected
   under jsoo -- is honest: the engine cannot index such a tensor there anyway.

   ---- every check is a PRE-operation bound ----

   CLAUDE.md's rule is flat: a check on a wrapped result is not a bound. So no
   operation here computes its result and then inspects it, and every arithmetic
   expression inside a check is itself safe. In particular:

     - [add] does not compute [a + b] and compare operand signs to the result
       sign. That test happens to be sound under two's-complement wrapping on
       both backends, but it is not a bound, and carving out an exception one
       line after forbidding the technique is how the rule erodes.

     - [mul] does not use [r / a = b]. At [a = -1, b = min_int] the product
       wraps to [min_int] and [min_int / -1] is also [min_int], so that check
       passes on a genuine overflow.

   There is no [neg]: once division stops negating its dividend (below), nothing
   in the language negates anything -- [scale (-1) x] goes through [mul]. *)

type error =
  [ `Index_overflow of string * int * int | `Non_positive_divisor of int ]

let pp_error fmt : [< error ] -> unit = function
  | `Index_overflow (op, a, b) ->
      (* [Sys.int_size] IS the domain width: 63 natively, 32 under
         js_of_ocaml. *)
      Fmt.pf fmt "index overflow: %s %d %d exceeds the %d-bit int domain" op a b
        Sys.int_size
  | `Non_positive_divisor d -> Fmt.pf fmt "divisor must be > 0, got %d" d

let add a b =
  if b > 0 && a > Stdlib.max_int - b then
    Core.fail (`Index_overflow ("add", a, b))
  else if b < 0 && a < Stdlib.min_int - b then
    Core.fail (`Index_overflow ("add", a, b))
  else Core.return (a + b)

let sub a b =
  if b < 0 && a > Stdlib.max_int + b then
    Core.fail (`Index_overflow ("sub", a, b))
  else if b > 0 && a < Stdlib.min_int + b then
    Core.fail (`Index_overflow ("sub", a, b))
  else Core.return (a - b)

(* Quadrant by sign, after peeling the two cases that would make a division
   inside the check itself unsafe. Every remaining [max_int / b] and
   [min_int / a] has a divisor that is neither 0 nor -1. *)
let mul a b =
  let overflow () = Core.fail (`Index_overflow ("mul", a, b)) in
  let ok () = Core.return (a * b) in
  if a = 0 || b = 0 then Core.return 0
  else if a = -1 then
    if b = Stdlib.min_int then overflow () else Core.return (-b)
  else if b = -1 then
    if a = Stdlib.min_int then overflow () else Core.return (-a)
  else if a > 0 then
    if b > 0 then if a > Stdlib.max_int / b then overflow () else ok ()
    else if b < Stdlib.min_int / a then overflow ()
    else ok ()
  else if b > 0 then if a < Stdlib.min_int / b then overflow () else ok ()
  else if a < Stdlib.max_int / b then overflow ()
  else ok ()

(* Floor and ceiling with a POSITIVE divisor, by remainder adjustment.

   Deliberately not [-floor_div (-n) d]: for a positive divisor every floor and
   ceiling quotient of every representable dividend is itself representable
   ([ceil_div min_int 1 = min_int]), so negating the dividend would manufacture
   an overflow the operation does not have. This form has no error case beyond
   the divisor, a positive divisor rules out the trapping [min_int / -1], and
   the adjustment cannot overflow either: the only dividend at a representation
   boundary is reached when [d = 1], where the remainder is zero and no
   adjustment happens.

   That makes this MORE defined than the form it replaces, which negates the
   dividend and so already misbehaves near [min_int]. *)
let floor_div_pos n d =
  if d <= 0 then Core.fail (`Non_positive_divisor d)
  else
    let q = n / d and r = n mod d in
    Core.return (if r < 0 then q - 1 else q)

let ceil_div_pos n d =
  if d <= 0 then Core.fail (`Non_positive_divisor d)
  else
    let q = n / d and r = n mod d in
    Core.return (if r > 0 then q + 1 else q)
