(* Section: the JavaScript printer. Reachable from Melange (js_ast depends on
   fmt alone), which is the point: [Printf "%.17g"] and [Int64.to_string] are
   different implementations on each backend, and the printer's promise is the
   same bytes on all of them. Prints a fixed corpus: the literal edges, every
   binary-operator pair in both child positions, the unary spacing rule, and one
   runtime helper rebuilt from constructors (Loop_js_runtime itself is in
   loop_ir, out of Melange's reach). *)

open Js_ast

let id = Ident.v
let v s = Var (id s)

let binops =
  [
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
  ]

let numbers =
  [
    0.;
    -0.;
    1.;
    -1.;
    0.5;
    0.1;
    -0.1;
    16777217.;
    16777216.;
    4.9e-324;
    Float.max_float;
    Float.min_float;
    1e21;
    1e-7;
    123456789012345678.;
    1. /. 3.;
    Float.pi;
    Float.nan;
    Float.infinity;
    Float.neg_infinity;
    2147483647.;
    -2147483648.;
  ]

let bigints =
  [
    0L;
    1L;
    -1L;
    Int64.min_int;
    Int64.max_int;
    9007199254740993L;
    -9007199254740993L;
    4294967296L;
    -4294967295L;
  ]

let strings = [ "kind"; "a\"b"; "a\\b"; "a\nb"; "\001"; "\xe2\x80\xa8" ]

(* [Loop_js_runtime.f16_to_float], as constructors. *)
let f16_to_float =
  let h = v "h" and sign = v "sign" and exp = v "exp" and mant = v "mant" in
  let num n = Number n in
  let pow2 e = Call (Member (Global Global.Math, id "pow"), [ num 2.; e ]) in
  Stmt.Function
    {
      Func.name = id "f16_to_float";
      params = [ id "h" ];
      body =
        [
          Stmt.Const
            (id "sign", Binary (Bit_and, Binary (Shr, h, num 15.), num 1.));
          Stmt.Const
            (id "exp", Binary (Bit_and, Binary (Shr, h, num 10.), num 31.));
          Stmt.Const (id "mant", Binary (Bit_and, h, num 1023.));
          Stmt.Let (id "m", num 0.);
          Stmt.If
            ( Binary (Eq_strict, exp, num 0.),
              [
                Stmt.Assign
                  (Lvar (id "m"), Eq, Binary (Mul, mant, pow2 (num (-24.))));
              ],
              [
                Stmt.If
                  ( Binary (Eq_strict, exp, num 31.),
                    [
                      Stmt.Assign
                        ( Lvar (id "m"),
                          Eq,
                          Cond
                            ( Binary (Eq_strict, mant, num 0.),
                              num Float.infinity,
                              num Float.nan ) );
                    ],
                    [
                      Stmt.Assign
                        ( Lvar (id "m"),
                          Eq,
                          Binary
                            ( Mul,
                              Binary (Bit_or, mant, num 1024.),
                              pow2 (Binary (Sub, exp, num 25.)) ) );
                    ] );
              ] );
          Stmt.Return
            (Some
               (Cond
                  (Binary (Eq_strict, sign, num 1.), Unary (Neg, v "m"), v "m")));
        ];
    }

let run () =
  print_endline "=== js_ast ===";
  List.iter (fun x -> print_endline (Js_print.expr (Number x))) numbers;
  List.iter (fun n -> print_endline (Js_print.expr (Bigint n))) bigints;
  List.iter (fun s -> print_endline (Js_print.expr (String s))) strings;
  let sym op =
    let s = Js_print.expr (Binary (op, v "_", v "_")) in
    String.sub s 2 (String.length s - 4)
  in
  let a = v "a" and b = v "b" and c = v "c" in
  List.iter
    (fun outer ->
      List.iter
        (fun inner ->
          Printf.printf "%s %s | %s | %s | %s\n" (sym outer) (sym inner)
            (Js_print.expr (Binary (outer, Binary (inner, a, b), c)))
            (Js_print.expr (Binary (outer, a, Binary (inner, b, c))))
            (Js_print.expr ~parens:`All
               (Binary (outer, Binary (inner, a, b), c))))
        binops)
    binops;
  List.iter
    (fun e -> print_endline (Js_print.expr e))
    [
      Unary (Neg, Unary (Neg, a));
      Unary (Plus, Unary (Plus, a));
      Binary (Sub, a, Unary (Neg, b));
      Unary (Neg, Number (-1.));
      Member (Number (-1.), id "x");
      Member (Bigint 1L, id "toString");
      Index (Binary (Add, a, b), c);
    ];
  print_string
    (Js_print.script
       {
         Program.prelude = [ f16_to_float ];
         entry =
           {
             Func.name = id "kernel";
             params = [ id "b0" ];
             body =
               [
                 Stmt.For
                   {
                     var = id "i0";
                     init = Number 0.;
                     test = Binary (Lt, v "i0", Number 4.);
                     body =
                       [
                         Stmt.Assign
                           ( Lindex (v "b0", v "i0"),
                             Eq,
                             Call (v "f16_to_float", [ Index (v "b0", v "i0") ])
                           );
                       ];
                   };
                 Stmt.Return (Some Null);
               ];
           };
       })
