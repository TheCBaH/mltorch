(* A dependency-free reproduction of Expr.Eval's recursive control flow.  The
   syntax and operations are intentionally small, but the relevant shapes are
   preserved:

   - binary/unary nodes leave work pending after a recursive value call;
   - Select asks a mutually recursive guard evaluator for a result;
   - Value_lt asks the value evaluator for two operands;
   - an indirect call through a ref models the unsupported call site discussed
     in Expr.Eval's source comment.

   Nothing in this directory links or copies the repository's Expr code. *)

type binary_op = Add
type unary_op = Neg

type expr =
  | Const of int
  | Binary of binary_op * expr * expr
  | Unary of unary_op * expr
  | Select of guard * expr * expr

and guard = Value_lt of expr * expr

let apply_binary Add a b = a + b
let apply_unary Neg a = -a

let binary_chain depth =
  let e = ref (Const 0) in
  for _ = 1 to depth do
    e := Binary (Add, !e, Const 1)
  done;
  (!e, depth)

let unary_chain depth =
  let e = ref (Const 1) in
  for _ = 1 to depth do
    e := Unary (Neg, !e)
  done;
  (!e, if depth land 1 = 0 then 1 else -1)

(* Each wrapper is the standalone equivalent of
   [select (value_lt previous 1) 1 0].  Starting at zero, the result alternates
   between one and zero. *)
let guard_chain depth =
  let e = ref (Const 0) in
  for _ = 1 to depth do
    e := Select (Value_lt (!e, Const 1), Const 1, Const 0)
  done;
  (!e, depth land 1)

(* Controls for the compiler documentation rather than the Expr reproduction. *)
let rec self_tail n acc =
  if n = 0 then acc else (self_tail [@tailcall]) (n - 1) (acc + 1)

let rec mutual_tail_a n acc =
  if n = 0 then acc else (mutual_tail_b [@tailcall]) (n - 1) (acc + 1)

and mutual_tail_b n acc =
  if n = 0 then acc else (mutual_tail_a [@tailcall]) (n - 1) (acc + 2)

let mutual_expected n = n + (n / 2)

(* The proposed rewrite for mutually recursive TAIL calls.  This is deliberately
   separate from [dispatch] below: here every transition carries all remaining
   work in the state and the recursive call is in tail position. *)
type mutual_state = Foo of int * int | Bar of int * int

let mutual_variant n acc =
  let rec loop = function
    | Foo (0, acc) | Bar (0, acc) -> acc
    | Foo (n, acc) -> (loop [@tailcall]) (Bar (n - 1, acc + 1))
    | Bar (n, acc) -> (loop [@tailcall]) (Foo (n - 1, acc + 2))
  in
  loop (Foo (n, acc))

(* If the functions' arguments can share a representation, keep them as loop
   parameters and make the variant a nullary control tag.  This retains the
   single-function dispatch while avoiding a fresh payload object per step. *)
type mutual_tag = Foo_tag | Bar_tag

let mutual_tag n acc =
  let rec loop tag n acc =
    if n = 0 then acc
    else
      match tag with
      | Foo_tag -> (loop [@tailcall]) Bar_tag (n - 1) (acc + 1)
      | Bar_tag -> (loop [@tailcall]) Foo_tag (n - 1) (acc + 2)
  in
  loop Foo_tag n acc

let indirect_cell = ref (fun (_ : int) (_ : int) -> assert false)
let indirect_tail n acc = if n = 0 then acc else !indirect_cell (n - 1) (acc + 1)
let () = indirect_cell := indirect_tail

(* The direct evaluator follows Expr.Eval.value's [go]/[guard] structure.  In
   particular, being in one statically known recursive group does not help the
   Binary, Unary, or Value_lt calls: arithmetic/comparison remains to be done
   after each recursive call returns. *)
let rec eval_direct = function
  | Const i -> i
  | Binary (op, a, b) -> apply_binary op (eval_direct a) (eval_direct b)
  | Unary (op, a) -> apply_unary op (eval_direct a)
  | Select (condition, if_true, if_false) ->
      if guard_direct condition then eval_direct if_true
      else eval_direct if_false

and guard_direct = function Value_lt (a, b) -> eval_direct a < eval_direct b

(* The proposed "one recursive function receiving a variant", applied
   literally to the same complete standalone evaluator.  It removes the [and]
   binding, but every recursive call that had pending work still has it. *)
type _ request = Eval : expr -> int request | Guard : guard -> bool request

let rec dispatch : type a. a request -> a = function
  | Eval (Const i) -> i
  | Eval (Binary (op, a, b)) ->
      apply_binary op (dispatch (Eval a)) (dispatch (Eval b))
  | Eval (Unary (op, a)) -> apply_unary op (dispatch (Eval a))
  | Eval (Select (condition, if_true, if_false)) ->
      if dispatch (Guard condition) then dispatch (Eval if_true)
      else dispatch (Eval if_false)
  | Guard (Value_lt (a, b)) -> dispatch (Eval a) < dispatch (Eval b)

let eval_dispatch e = dispatch (Eval e)

(* A closure trampoline for the same non-tail evaluator. [depth] counts calls
   across evaluation, guard evaluation, and continuation application. At the
   threshold, [More] is returned unchanged through the tail-position callers
   to [drive]; only the root invokes the suspension, after the old stack has
   unwound, and the resumed segment starts at depth zero. Threshold 1 is the
   eager form; a larger threshold amortizes root bounces over a bounded stack
   segment. *)
type bounce = Done of int | More of (unit -> bounce)
type trampoline_result = { value : int; bounces : int; max_depth : int }

let eval_trampoline ~threshold e =
  if threshold < 1 then
    invalid_arg "eval_trampoline: threshold must be positive";
  let maximum = ref 0 in
  let note depth = if depth > !maximum then maximum := depth in
  let rec continue : type a. int -> (int -> a -> bounce) -> a -> bounce =
   fun depth k value ->
    if depth >= threshold then More (fun () -> (continue [@tailcall]) 0 k value)
    else
      let depth = depth + 1 in
      note depth;
      (k [@tailcall]) depth value
  in
  let rec eval depth expression k =
    if depth >= threshold then
      More (fun () -> (eval [@tailcall]) 0 expression k)
    else
      let depth = depth + 1 in
      note depth;
      match expression with
      | Const i -> (continue [@tailcall]) depth k i
      | Binary (op, a, b) ->
          (eval [@tailcall]) depth a (fun depth av ->
              (eval [@tailcall]) depth b (fun depth bv ->
                  (continue [@tailcall]) depth k (apply_binary op av bv)))
      | Unary (op, a) ->
          (eval [@tailcall]) depth a (fun depth av ->
              (continue [@tailcall]) depth k (apply_unary op av))
      | Select (condition, if_true, if_false) ->
          (guard [@tailcall]) depth condition (fun depth selected ->
              (eval [@tailcall]) depth
                (if selected then if_true else if_false)
                k)
  and guard depth condition k =
    if depth >= threshold then
      More (fun () -> (guard [@tailcall]) 0 condition k)
    else
      let depth = depth + 1 in
      note depth;
      match condition with
      | Value_lt (a, b) ->
          (eval [@tailcall]) depth a (fun depth av ->
              (eval [@tailcall]) depth b (fun depth bv ->
                  (continue [@tailcall]) depth k (av < bv)))
  in
  let bounces = ref 0 in
  let rec drive = function
    | Done value -> { value; bounces = !bounces; max_depth = !maximum }
    | More resume ->
        incr bounces;
        (drive [@tailcall]) (resume ())
  in
  drive (eval 0 e (fun _ value -> Done value))

let eval_trampoline_eager e = (eval_trampoline ~threshold:1 e).value
let eval_trampoline_delayed e = (eval_trampoline ~threshold:512 e).value

(* Defunctionalized continuations for the same evaluator.  The frames are the
   work the direct and naive versions leave on the JavaScript call stack.  All
   recursive transitions in [loop] are self tail calls, so the JS compilers can
   emit a loop.  This has constant CALL-STACK usage and O(depth) live heap
   state, which is the general guarantee available for an immutable tree. *)
type frame =
  | Binary_left of binary_op * expr
  | Binary_right of binary_op * int
  | Unary_result of unary_op
  | Select_result of expr * expr
  | Value_lt_left of expr
  | Value_lt_right of int

type state =
  | Eval_state of expr
  | Guard_state of guard
  | Int_result of int
  | Bool_result of bool

let eval_machine e =
  let rec loop state frames =
    match (state, frames) with
    | Eval_state (Const i), _ -> loop (Int_result i) frames
    | Eval_state (Binary (op, a, b)), _ ->
        loop (Eval_state a) (Binary_left (op, b) :: frames)
    | Eval_state (Unary (op, a)), _ ->
        loop (Eval_state a) (Unary_result op :: frames)
    | Eval_state (Select (condition, if_true, if_false)), _ ->
        loop (Guard_state condition)
          (Select_result (if_true, if_false) :: frames)
    | Guard_state (Value_lt (a, b)), _ ->
        loop (Eval_state a) (Value_lt_left b :: frames)
    | Int_result a, Binary_left (op, b) :: rest ->
        loop (Eval_state b) (Binary_right (op, a) :: rest)
    | Int_result b, Binary_right (op, a) :: rest ->
        loop (Int_result (apply_binary op a b)) rest
    | Int_result a, Unary_result op :: rest ->
        loop (Int_result (apply_unary op a)) rest
    | Int_result a, Value_lt_left b :: rest ->
        loop (Eval_state b) (Value_lt_right a :: rest)
    | Int_result b, Value_lt_right a :: rest -> loop (Bool_result (a < b)) rest
    | Bool_result condition, Select_result (if_true, if_false) :: rest ->
        loop (Eval_state (if condition then if_true else if_false)) rest
    | Int_result i, [] -> i
    | Bool_result _, []
    | ( Bool_result _,
        ( Binary_left _ | Binary_right _ | Unary_result _ | Value_lt_left _
        | Value_lt_right _ )
        :: _ )
    | Int_result _, Select_result _ :: _ ->
        assert false
  in
  loop (Eval_state e) []

(* The same explicit-frame algorithm with storage amortized across evaluations.
   Parallel arrays avoid allocating a list cell and payload variant for every
   push. [ensure_reusable_frames] grows geometrically; after a warm-up at the
   maximum expression depth, evaluation only overwrites existing slots. *)
type reusable_frames = {
  mutable tags : int array;
  mutable expr1 : expr array;
  mutable expr2 : expr array;
  mutable values : int array;
  mutable binary_ops : binary_op array;
  mutable unary_ops : unary_op array;
}

let create_reusable_frames capacity =
  let capacity = max 1 capacity in
  {
    tags = Array.make capacity 0;
    expr1 = Array.make capacity (Const 0);
    expr2 = Array.make capacity (Const 0);
    values = Array.make capacity 0;
    binary_ops = Array.make capacity Add;
    unary_ops = Array.make capacity Neg;
  }

let ensure_reusable_frames frames needed =
  let old_capacity = Array.length frames.tags in
  if needed > old_capacity then (
    let capacity = ref old_capacity in
    while !capacity < needed do
      capacity := !capacity * 2
    done;
    let grow old init =
      let fresh = Array.make !capacity init in
      Array.blit old 0 fresh 0 old_capacity;
      fresh
    in
    frames.tags <- grow frames.tags 0;
    frames.expr1 <- grow frames.expr1 (Const 0);
    frames.expr2 <- grow frames.expr2 (Const 0);
    frames.values <- grow frames.values 0;
    frames.binary_ops <- grow frames.binary_ops Add;
    frames.unary_ops <- grow frames.unary_ops Neg)

type machine_mode = Eval_mode | Guard_mode | Int_mode | Bool_mode

let dummy_guard = Value_lt (Const 0, Const 0)

let eval_machine_reuse frames e =
  let rec loop mode expression condition int_value bool_value top =
    match mode with
    | Eval_mode -> (
        match expression with
        | Const i ->
            (loop [@tailcall]) Int_mode expression condition i bool_value top
        | Binary (op, a, b) ->
            ensure_reusable_frames frames (top + 1);
            frames.tags.(top) <- 0;
            frames.expr1.(top) <- b;
            frames.binary_ops.(top) <- op;
            (loop [@tailcall]) Eval_mode a condition int_value bool_value
              (top + 1)
        | Unary (op, a) ->
            ensure_reusable_frames frames (top + 1);
            frames.tags.(top) <- 2;
            frames.unary_ops.(top) <- op;
            (loop [@tailcall]) Eval_mode a condition int_value bool_value
              (top + 1)
        | Select (guard, if_true, if_false) ->
            ensure_reusable_frames frames (top + 1);
            frames.tags.(top) <- 3;
            frames.expr1.(top) <- if_true;
            frames.expr2.(top) <- if_false;
            (loop [@tailcall]) Guard_mode expression guard int_value bool_value
              (top + 1))
    | Guard_mode -> (
        match condition with
        | Value_lt (a, b) ->
            ensure_reusable_frames frames (top + 1);
            frames.tags.(top) <- 4;
            frames.expr1.(top) <- b;
            (loop [@tailcall]) Eval_mode a condition int_value bool_value
              (top + 1))
    | Int_mode ->
        if top = 0 then int_value
        else
          let slot = top - 1 in
          let tag = frames.tags.(slot) in
          if tag = 0 then (
            frames.tags.(slot) <- 1;
            frames.values.(slot) <- int_value;
            (loop [@tailcall]) Eval_mode frames.expr1.(slot) condition int_value
              bool_value top)
          else if tag = 1 then
            let value =
              apply_binary frames.binary_ops.(slot) frames.values.(slot)
                int_value
            in
            (loop [@tailcall]) Int_mode expression condition value bool_value
              slot
          else if tag = 2 then
            let value = apply_unary frames.unary_ops.(slot) int_value in
            (loop [@tailcall]) Int_mode expression condition value bool_value
              slot
          else if tag = 4 then (
            frames.tags.(slot) <- 5;
            frames.values.(slot) <- int_value;
            (loop [@tailcall]) Eval_mode frames.expr1.(slot) condition int_value
              bool_value top)
          else if tag = 5 then
            (loop [@tailcall]) Bool_mode expression condition int_value
              (frames.values.(slot) < int_value)
              slot
          else assert false
    | Bool_mode ->
        if top = 0 then assert false
        else
          let slot = top - 1 in
          if frames.tags.(slot) <> 3 then assert false
          else
            let selected =
              if bool_value then frames.expr1.(slot) else frames.expr2.(slot)
            in
            (loop [@tailcall]) Eval_mode selected condition int_value bool_value
              slot
  in
  loop Eval_mode e dummy_guard 0 false 0

(* Preserve direct evaluation until [threshold], then evaluate that entire
   subtree with the constant-call-stack reusable-frame implementation. The
   existing direct frames remain live but do not grow any deeper, so maximum
   host-stack use is the threshold plus the machine's fixed overhead. *)
let eval_hybrid ~threshold frames e =
  if threshold < 1 then invalid_arg "eval_hybrid: threshold must be positive";
  let rec eval depth expression =
    if depth >= threshold then eval_machine_reuse frames expression
    else
      let depth = depth + 1 in
      match expression with
      | Const i -> i
      | Binary (op, a, b) -> apply_binary op (eval depth a) (eval depth b)
      | Unary (op, a) -> apply_unary op (eval depth a)
      | Select (condition, if_true, if_false) ->
          if guard depth condition then eval depth if_true
          else eval depth if_false
  and guard depth = function
    | Value_lt (a, b) ->
        if depth >= threshold then
          eval_machine_reuse frames a < eval_machine_reuse frames b
        else
          let depth = depth + 1 in
          eval depth a < eval depth b
  in
  eval 0 e

type outcome = Ok | Wrong | Raised

let observe expected f =
  try if Int.equal (f ()) expected then Ok else Wrong with _ -> Raised

let outcome_name = function Ok -> "ok" | Wrong -> "wrong" | Raised -> "raised"
