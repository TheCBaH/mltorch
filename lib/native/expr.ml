(* The symbolic value IR built by [Symbolic]. A pixel is a pure expression over
   [Load]s of its inputs; index expressions are affine over the output-coord
   variables ([Ivar]) and reduction variables ([Rvar]). [pp] renders it infix so
   the dependency structure is legible (the most important printer in the engine);
   [eval] interprets it against a binding at a concrete output coord, which is how
   the Symbolic instance is checked to agree with Direct.
   See .ai/native_symbolic_language.md §2.3, §2.4. *)

type binop = Add | Sub | Mul | Div
type unop = Exp | Sqrt
type cmpop = Lt
type redkind = Sum | Max_r

type iexpr =
  | Ivar of Axis.t (* an output-coord component *)
  | Rvar of int (* a reduction variable *)
  | Iconst of int
  | Iadd of iexpr * iexpr
  | Iscale of int * iexpr
  | Imax of iexpr * iexpr (* clamp a windowed reduction's bounds in-range *)
  | Imin of iexpr * iexpr

(* [bexpr] is the boolean domain ([Symbolic.b]); it only appears as a [Select]
   guard, never as a standalone value — hence no [Bool] value constructor. *)
type bexpr = Cmp of cmpop * t * t

and t =
  | Const of float
  | Bin of binop * t * t
  | Un of unop * t
  | Select of bexpr * t * t (* [select c a b]: a when c holds, else b *)
  | Load of Tensor_sig.t * iexpr array (* 6 indices, Axis order *)
  | Reduce of {
      kind : redkind;
      var : int;
      lo : iexpr; (* affine in the output coord for a windowed reduction *)
      hi : iexpr;
      body : t;
    }

(* ---- printers ---- *)

let binop_sym = function Add -> "+" | Sub -> "-" | Mul -> "*" | Div -> "/"
let unop_name = function Exp -> "exp" | Sqrt -> "sqrt"
let cmpop_sym = function Lt -> "<"
let redkind_name = function Sum -> "sum" | Max_r -> "maxr"

let rec pp_iexpr fmt = function
  | Ivar a -> Axis.pp fmt a
  | Rvar v -> Format.fprintf fmt "r%d" v
  | Iconst n -> Format.fprintf fmt "%d" n
  | Iadd (a, b) -> Format.fprintf fmt "%a+%a" pp_iexpr a pp_iexpr b
  | Iscale (k, a) -> Format.fprintf fmt "%d*%a" k pp_iexpr a
  | Imax (a, b) -> Format.fprintf fmt "max(%a,%a)" pp_iexpr a pp_iexpr b
  | Imin (a, b) -> Format.fprintf fmt "min(%a,%a)" pp_iexpr a pp_iexpr b

let pp_idx fmt arr =
  Array.iteri
    (fun i ie ->
      if i > 0 then Format.pp_print_char fmt ',';
      pp_iexpr fmt ie)
    arr

let rec pp fmt = function
  | Const x -> Format.fprintf fmt "%g" x
  | Bin (op, a, b) -> Format.fprintf fmt "(%a %s %a)" pp a (binop_sym op) pp b
  | Un (op, a) -> Format.fprintf fmt "%s(%a)" (unop_name op) pp a
  | Select (c, a, b) ->
      Format.fprintf fmt "select(%a, %a, %a)" pp_bexpr c pp a pp b
  | Load (s, idx) -> Format.fprintf fmt "%a[%a]" Tensor_sig.pp s pp_idx idx
  | Reduce { kind; var; lo; hi; body } ->
      Format.fprintf fmt "%s(r%d=%a..%a: %a)" (redkind_name kind) var pp_iexpr
        lo pp_iexpr hi pp body

and pp_bexpr fmt (Cmp (op, a, b)) =
  Format.fprintf fmt "(%a %s %a)" pp a (cmpop_sym op) pp b

(* ---- evaluation (verification against Direct) ---- *)

let apply_binop = function
  | Add -> ( +. )
  | Sub -> ( -. )
  | Mul -> ( *. )
  | Div -> ( /. )

let apply_unop = function Exp -> Stdlib.exp | Sqrt -> Stdlib.sqrt
let apply_cmpop = function Lt -> ( < )

let rec eval_iexpr ~coord ~rvars = function
  | Ivar a -> coord a
  | Rvar v -> List.assoc v rvars
  | Iconst n -> n
  | Iadd (a, b) -> eval_iexpr ~coord ~rvars a + eval_iexpr ~coord ~rvars b
  | Iscale (k, a) -> k * eval_iexpr ~coord ~rvars a
  | Imax (a, b) ->
      Stdlib.max (eval_iexpr ~coord ~rvars a) (eval_iexpr ~coord ~rvars b)
  | Imin (a, b) ->
      Stdlib.min (eval_iexpr ~coord ~rvars a) (eval_iexpr ~coord ~rvars b)

(* [binding] maps an input signature to a real tensor; [coord] gives the concrete
   output coordinate to evaluate at. *)
let rec eval ~binding ~coord ?(rvars = []) e =
  let recur = eval ~binding ~coord ~rvars in
  match e with
  | Const x -> x
  | Bin (op, a, b) -> apply_binop op (recur a) (recur b)
  | Un (op, a) -> apply_unop op (recur a)
  | Select (Cmp (op, ca, cb), a, b) ->
      if apply_cmpop op (recur ca) (recur cb) then recur a else recur b
  | Load (s, idx) ->
      Tensor.read_guarded (binding s) (fun a ->
          eval_iexpr ~coord ~rvars idx.(Axis.to_int a))
  | Reduce { kind; var; lo; hi; body } ->
      let combine, init =
        match kind with
        | Sum -> (( +. ), 0.)
        | Max_r -> (Float.max, neg_infinity)
      in
      let lo = eval_iexpr ~coord ~rvars lo
      and hi = eval_iexpr ~coord ~rvars hi in
      let rec loop i acc =
        if i >= hi then acc
        else
          loop (i + 1)
            (combine acc (eval ~binding ~coord ~rvars:((var, i) :: rvars) body))
      in
      loop lo init
