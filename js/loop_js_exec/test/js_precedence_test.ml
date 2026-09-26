open Js_of_ocaml
open Js_ast

(* The precedence differential. A wrong entry in the printer's precedence table
   changes what an expression means, and a golden only catches the pairs someone
   thought to write down. So: seeded random expression trees over every operator,
   printed once with the minimum parentheses and once with every compound child
   parenthesised (the second reading cannot depend on the table), each evaluated
   by the engine, and the two results compared with [Object.is] plus their type,
   which tells -0 from +0 and a number from a boolean. *)

let binops =
  [|
    Add;
    And;
    Bit_and;
    Bit_or;
    Div;
    Eq_strict;
    Ge;
    Gt;
    Le;
    Lt;
    Mul;
    Ne_strict;
    Or;
    Shl;
    Shr;
    Sub;
  |]

let unops = [| Neg; Not; Plus |]

let leaves =
  [|
    0.;
    -0.;
    1.;
    -1.;
    2.;
    3.;
    7.;
    0.5;
    -2.5;
    1e21;
    2147483648.;
    Float.nan;
    Float.infinity;
    Float.neg_infinity;
  |]

let draw pcg n =
  let x, pcg = Walk_core.Pcg.next pcg in
  (Int64.to_int (Int64.rem x (Int64.of_int n)), pcg)

let rec gen pcg depth =
  let pick, pcg = draw pcg 10 in
  if depth = 0 || pick < 2 then
    let i, pcg = draw pcg (Array.length leaves) in
    (Number leaves.(i), pcg)
  else if pick < 8 then
    let i, pcg = draw pcg (Array.length binops) in
    let a, pcg = gen pcg (depth - 1) in
    let b, pcg = gen pcg (depth - 1) in
    (Binary (binops.(i), a, b), pcg)
  else if pick < 9 then
    let i, pcg = draw pcg (Array.length unops) in
    let a, pcg = gen pcg (depth - 1) in
    (Unary (unops.(i), a), pcg)
  else
    let c, pcg = gen pcg (depth - 1) in
    let a, pcg = gen pcg (depth - 1) in
    let b, pcg = gen pcg (depth - 1) in
    (Cond (c, a, b), pcg)

let value parens e =
  match
    Err.payload
      (Loop_js_exec.compile_source ("return " ^ Js_print.expr ~parens e ^ ";"))
  with
  | Ok v -> v
  | Error (`Js_compile m) -> failwith ("printed source does not parse: " ^ m)

let same =
  Js.Unsafe.js_expr
    "(function (a, b) { return Object.is(a, b) && typeof a === typeof b; })"

let differential ~count ~seed =
  let bad = ref [] in
  let pcg = ref (Walk_core.Pcg.seed ~seed ~seq:1L) in
  for _ = 1 to count do
    let e, next = gen !pcg 5 in
    pcg := next;
    let minimal = value `Minimal e and all = value `All e in
    if not (Js.to_bool (Js.Unsafe.fun_call same [| minimal; all |])) then
      bad := e :: !bad
  done;
  List.rev !bad

let%expect_test "minimal and fully parenthesised printing evaluate alike" =
  let bad = differential ~count:3000 ~seed:20260921L in
  Fmt.pr "%d of 3000 trees disagree@." (List.length bad);
  List.iteri
    (fun i e ->
      if i < 3 then
        Fmt.pr "  %s@.  %s@." (Js_print.expr e) (Js_print.expr ~parens:`All e))
    bad;
  [%expect {| 0 of 3000 trees disagree |}]
