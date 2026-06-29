(* The symbolic value IR built by [Symbolic]. A pixel is a pure expression over
   [Load]s of its inputs; index expressions are affine over the output-coord
   variables ([Index_var]) and reduction variables ([Reduce_var]). [pp] renders it
   infix so the dependency structure is legible (the most important printer in the
   engine); [eval] interprets it against a binding at a concrete output coord, which
   is how the Symbolic instance is checked to agree with Direct.
   See .ai/native_symbolic_language.md §2.3, §2.4. *)

type binary_op = Add | Sub | Mul | Div
type unary_op = Exp | Sqrt
type compare_op = Lt
type reduction_kind = Sum | Max_reduce

type index_expr =
  | Index_var of Axis.t (* an output-coord component *)
  | Reduce_var of int (* a reduction variable *)
  | Index_const of int
  | Index_add of index_expr * index_expr
  | Index_scale of int * index_expr
  | Index_floor_div_pos of index_expr * int
  | Index_ceil_div_pos of index_expr * int
  | Index_max of
      index_expr * index_expr (* clamp a windowed reduction's bounds in-range *)
  | Index_min of index_expr * index_expr

(* [bool_expr] is the boolean domain ([Symbolic.b]); it only appears as a [Select]
   guard, never as a standalone value — hence no [Bool] value constructor. *)
type bool_expr = Cmp of compare_op * t * t

and t =
  | Const of float
  | Binary of binary_op * t * t
  | Unary of unary_op * t
  | Select of bool_expr * t * t (* [select c a b]: a when c holds, else b *)
  | Load of Tensor_sig.t * index_expr array (* 6 indices, Axis order *)
  | Reduce of {
      kind : reduction_kind;
      var : int;
      lo : index_expr; (* affine in the output coord for a windowed reduction *)
      hi : index_expr;
      body : t;
    }

(* ---- printers ---- *)

let binary_op_sym = function Add -> "+" | Sub -> "-" | Mul -> "*" | Div -> "/"
let unary_op_name = function Exp -> "exp" | Sqrt -> "sqrt"
let compare_op_sym = function Lt -> "<"
let reduction_kind_name = function Sum -> "sum" | Max_reduce -> "max_reduce"

let rec pp_index_expr fmt = function
  | Index_var a -> Axis.pp fmt a
  | Reduce_var v -> Format.fprintf fmt "r%d" v
  | Index_const n -> Format.fprintf fmt "%d" n
  | Index_add (a, b) ->
      Format.fprintf fmt "%a+%a" pp_index_expr a pp_index_expr b
  | Index_scale (k, a) -> Format.fprintf fmt "%d*%a" k pp_index_expr a
  | Index_floor_div_pos (a, d) ->
      Format.fprintf fmt "floor_div(%a,%d)" pp_index_expr a d
  | Index_ceil_div_pos (a, d) ->
      Format.fprintf fmt "ceil_div(%a,%d)" pp_index_expr a d
  | Index_max (a, b) ->
      Format.fprintf fmt "max(%a,%a)" pp_index_expr a pp_index_expr b
  | Index_min (a, b) ->
      Format.fprintf fmt "min(%a,%a)" pp_index_expr a pp_index_expr b

let pp_index_array fmt arr =
  Array.iteri
    (fun i ie ->
      if i > 0 then Format.pp_print_char fmt ',';
      pp_index_expr fmt ie)
    arr

let rec pp fmt = function
  | Const x -> Format.fprintf fmt "%g" x
  | Binary (op, a, b) ->
      Format.fprintf fmt "(%a %s %a)" pp a (binary_op_sym op) pp b
  | Unary (op, a) -> Format.fprintf fmt "%s(%a)" (unary_op_name op) pp a
  | Select (c, a, b) ->
      Format.fprintf fmt "select(%a, %a, %a)" pp_bool_expr c pp a pp b
  | Load (s, index) ->
      Format.fprintf fmt "%a[%a]" Tensor_sig.pp s pp_index_array index
  | Reduce { kind; var; lo; hi; body } ->
      Format.fprintf fmt "%s(r%d=%a..%a: %a)" (reduction_kind_name kind) var
        pp_index_expr lo pp_index_expr hi pp body

and pp_bool_expr fmt (Cmp (op, a, b)) =
  Format.fprintf fmt "(%a %s %a)" pp a (compare_op_sym op) pp b

(* ---- evaluation (verification against Direct) ---- *)

let apply_binary_op = function
  | Add -> ( +. )
  | Sub -> ( -. )
  | Mul -> ( *. )
  | Div -> ( /. )

let apply_unary_op = function Exp -> Stdlib.exp | Sqrt -> Stdlib.sqrt
let apply_compare_op = function Lt -> ( < )

let rec eval_index_expr ~coord ~rvars = function
  | Index_var a -> coord a
  | Reduce_var v -> List.assoc v rvars
  | Index_const n -> n
  | Index_add (a, b) ->
      eval_index_expr ~coord ~rvars a + eval_index_expr ~coord ~rvars b
  | Index_scale (k, a) -> k * eval_index_expr ~coord ~rvars a
  | Index_floor_div_pos (a, d) ->
      let n = eval_index_expr ~coord ~rvars a in
      if n >= 0 then n / d else -((-n + d - 1) / d)
  | Index_ceil_div_pos (a, d) ->
      let n = eval_index_expr ~coord ~rvars a in
      let floor_neg = if -n >= 0 then -n / d else -((n + d - 1) / d) in
      -floor_neg
  | Index_max (a, b) ->
      Stdlib.max
        (eval_index_expr ~coord ~rvars a)
        (eval_index_expr ~coord ~rvars b)
  | Index_min (a, b) ->
      Stdlib.min
        (eval_index_expr ~coord ~rvars a)
        (eval_index_expr ~coord ~rvars b)

(* [binding] maps an input signature to a real tensor; [coord] gives the concrete
   output coordinate to evaluate at. *)
let rec eval ~binding ~coord ?(rvars = []) e =
  let recur = eval ~binding ~coord ~rvars in
  match e with
  | Const x -> x
  | Binary (op, a, b) -> apply_binary_op op (recur a) (recur b)
  | Unary (op, a) -> apply_unary_op op (recur a)
  | Select (Cmp (op, ca, cb), a, b) ->
      if apply_compare_op op (recur ca) (recur cb) then recur a else recur b
  | Load (s, index) ->
      Tensor.read_at (binding s) (fun a ->
          eval_index_expr ~coord ~rvars index.(Axis.to_int a))
  | Reduce { kind; var; lo; hi; body } ->
      let combine, init =
        match kind with
        | Sum -> (( +. ), 0.)
        | Max_reduce -> (Float.max, neg_infinity)
      in
      let lo = eval_index_expr ~coord ~rvars lo
      and hi = eval_index_expr ~coord ~rvars hi in
      let rec loop i acc =
        if i >= hi then acc
        else
          loop (i + 1)
            (combine acc (eval ~binding ~coord ~rvars:((var, i) :: rvars) body))
      in
      loop lo init
