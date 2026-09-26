open Js_ast

let v s = Var (Ident.v s)

let a = v "a"
and b = v "b"
and c = v "c"

let show e = Fmt.pr "%s@." (Js_print.expr e)

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

let sym op =
  let s = Js_print.expr (Binary (op, Var (Ident.v "_"), Var (Ident.v "_"))) in
  String.sub s 2 (String.length s - 4)

(* Every ordered pair of operators, the inner one as the left operand and as the
   right one: the parentheses appear exactly where the table says. *)
let%expect_test "every binop pair, inner as left child then as right child" =
  List.iter
    (fun outer ->
      List.iter
        (fun inner ->
          Fmt.pr "%-3s %-3s | %s | %s@." (sym outer) (sym inner)
            (Js_print.expr (Binary (outer, Binary (inner, a, b), c)))
            (Js_print.expr (Binary (outer, a, Binary (inner, b, c)))))
        binops)
    binops;
  [%expect
    {|
    +   +   | a + b + c | a + (b + c)
    +   &&  | (a && b) + c | a + (b && c)
    +   &   | (a & b) + c | a + (b & c)
    +   |   | (a | b) + c | a + (b | c)
    +   /   | a / b + c | a + b / c
    +   === | (a === b) + c | a + (b === c)
    +   >=  | (a >= b) + c | a + (b >= c)
    +   >   | (a > b) + c | a + (b > c)
    +   <=  | (a <= b) + c | a + (b <= c)
    +   <   | (a < b) + c | a + (b < c)
    +   *   | a * b + c | a + b * c
    +   !== | (a !== b) + c | a + (b !== c)
    +   ||  | (a || b) + c | a + (b || c)
    +   <<  | (a << b) + c | a + (b << c)
    +   >>  | (a >> b) + c | a + (b >> c)
    +   -   | a - b + c | a + (b - c)
    &&  +   | a + b && c | a && b + c
    &&  &&  | a && b && c | a && (b && c)
    &&  &   | a & b && c | a && b & c
    &&  |   | a | b && c | a && b | c
    &&  /   | a / b && c | a && b / c
    &&  === | a === b && c | a && b === c
    &&  >=  | a >= b && c | a && b >= c
    &&  >   | a > b && c | a && b > c
    &&  <=  | a <= b && c | a && b <= c
    &&  <   | a < b && c | a && b < c
    &&  *   | a * b && c | a && b * c
    &&  !== | a !== b && c | a && b !== c
    &&  ||  | (a || b) && c | a && (b || c)
    &&  <<  | a << b && c | a && b << c
    &&  >>  | a >> b && c | a && b >> c
    &&  -   | a - b && c | a && b - c
    &   +   | a + b & c | a & b + c
    &   &&  | (a && b) & c | a & (b && c)
    &   &   | a & b & c | a & (b & c)
    &   |   | (a | b) & c | a & (b | c)
    &   /   | a / b & c | a & b / c
    &   === | a === b & c | a & b === c
    &   >=  | a >= b & c | a & b >= c
    &   >   | a > b & c | a & b > c
    &   <=  | a <= b & c | a & b <= c
    &   <   | a < b & c | a & b < c
    &   *   | a * b & c | a & b * c
    &   !== | a !== b & c | a & b !== c
    &   ||  | (a || b) & c | a & (b || c)
    &   <<  | a << b & c | a & b << c
    &   >>  | a >> b & c | a & b >> c
    &   -   | a - b & c | a & b - c
    |   +   | a + b | c | a | b + c
    |   &&  | (a && b) | c | a | (b && c)
    |   &   | a & b | c | a | b & c
    |   |   | a | b | c | a | (b | c)
    |   /   | a / b | c | a | b / c
    |   === | a === b | c | a | b === c
    |   >=  | a >= b | c | a | b >= c
    |   >   | a > b | c | a | b > c
    |   <=  | a <= b | c | a | b <= c
    |   <   | a < b | c | a | b < c
    |   *   | a * b | c | a | b * c
    |   !== | a !== b | c | a | b !== c
    |   ||  | (a || b) | c | a | (b || c)
    |   <<  | a << b | c | a | b << c
    |   >>  | a >> b | c | a | b >> c
    |   -   | a - b | c | a | b - c
    /   +   | (a + b) / c | a / (b + c)
    /   &&  | (a && b) / c | a / (b && c)
    /   &   | (a & b) / c | a / (b & c)
    /   |   | (a | b) / c | a / (b | c)
    /   /   | a / b / c | a / (b / c)
    /   === | (a === b) / c | a / (b === c)
    /   >=  | (a >= b) / c | a / (b >= c)
    /   >   | (a > b) / c | a / (b > c)
    /   <=  | (a <= b) / c | a / (b <= c)
    /   <   | (a < b) / c | a / (b < c)
    /   *   | a * b / c | a / (b * c)
    /   !== | (a !== b) / c | a / (b !== c)
    /   ||  | (a || b) / c | a / (b || c)
    /   <<  | (a << b) / c | a / (b << c)
    /   >>  | (a >> b) / c | a / (b >> c)
    /   -   | (a - b) / c | a / (b - c)
    === +   | a + b === c | a === b + c
    === &&  | (a && b) === c | a === (b && c)
    === &   | (a & b) === c | a === (b & c)
    === |   | (a | b) === c | a === (b | c)
    === /   | a / b === c | a === b / c
    === === | a === b === c | a === (b === c)
    === >=  | a >= b === c | a === b >= c
    === >   | a > b === c | a === b > c
    === <=  | a <= b === c | a === b <= c
    === <   | a < b === c | a === b < c
    === *   | a * b === c | a === b * c
    === !== | a !== b === c | a === (b !== c)
    === ||  | (a || b) === c | a === (b || c)
    === <<  | a << b === c | a === b << c
    === >>  | a >> b === c | a === b >> c
    === -   | a - b === c | a === b - c
    >=  +   | a + b >= c | a >= b + c
    >=  &&  | (a && b) >= c | a >= (b && c)
    >=  &   | (a & b) >= c | a >= (b & c)
    >=  |   | (a | b) >= c | a >= (b | c)
    >=  /   | a / b >= c | a >= b / c
    >=  === | (a === b) >= c | a >= (b === c)
    >=  >=  | a >= b >= c | a >= (b >= c)
    >=  >   | a > b >= c | a >= (b > c)
    >=  <=  | a <= b >= c | a >= (b <= c)
    >=  <   | a < b >= c | a >= (b < c)
    >=  *   | a * b >= c | a >= b * c
    >=  !== | (a !== b) >= c | a >= (b !== c)
    >=  ||  | (a || b) >= c | a >= (b || c)
    >=  <<  | a << b >= c | a >= b << c
    >=  >>  | a >> b >= c | a >= b >> c
    >=  -   | a - b >= c | a >= b - c
    >   +   | a + b > c | a > b + c
    >   &&  | (a && b) > c | a > (b && c)
    >   &   | (a & b) > c | a > (b & c)
    >   |   | (a | b) > c | a > (b | c)
    >   /   | a / b > c | a > b / c
    >   === | (a === b) > c | a > (b === c)
    >   >=  | a >= b > c | a > (b >= c)
    >   >   | a > b > c | a > (b > c)
    >   <=  | a <= b > c | a > (b <= c)
    >   <   | a < b > c | a > (b < c)
    >   *   | a * b > c | a > b * c
    >   !== | (a !== b) > c | a > (b !== c)
    >   ||  | (a || b) > c | a > (b || c)
    >   <<  | a << b > c | a > b << c
    >   >>  | a >> b > c | a > b >> c
    >   -   | a - b > c | a > b - c
    <=  +   | a + b <= c | a <= b + c
    <=  &&  | (a && b) <= c | a <= (b && c)
    <=  &   | (a & b) <= c | a <= (b & c)
    <=  |   | (a | b) <= c | a <= (b | c)
    <=  /   | a / b <= c | a <= b / c
    <=  === | (a === b) <= c | a <= (b === c)
    <=  >=  | a >= b <= c | a <= (b >= c)
    <=  >   | a > b <= c | a <= (b > c)
    <=  <=  | a <= b <= c | a <= (b <= c)
    <=  <   | a < b <= c | a <= (b < c)
    <=  *   | a * b <= c | a <= b * c
    <=  !== | (a !== b) <= c | a <= (b !== c)
    <=  ||  | (a || b) <= c | a <= (b || c)
    <=  <<  | a << b <= c | a <= b << c
    <=  >>  | a >> b <= c | a <= b >> c
    <=  -   | a - b <= c | a <= b - c
    <   +   | a + b < c | a < b + c
    <   &&  | (a && b) < c | a < (b && c)
    <   &   | (a & b) < c | a < (b & c)
    <   |   | (a | b) < c | a < (b | c)
    <   /   | a / b < c | a < b / c
    <   === | (a === b) < c | a < (b === c)
    <   >=  | a >= b < c | a < (b >= c)
    <   >   | a > b < c | a < (b > c)
    <   <=  | a <= b < c | a < (b <= c)
    <   <   | a < b < c | a < (b < c)
    <   *   | a * b < c | a < b * c
    <   !== | (a !== b) < c | a < (b !== c)
    <   ||  | (a || b) < c | a < (b || c)
    <   <<  | a << b < c | a < b << c
    <   >>  | a >> b < c | a < b >> c
    <   -   | a - b < c | a < b - c
    *   +   | (a + b) * c | a * (b + c)
    *   &&  | (a && b) * c | a * (b && c)
    *   &   | (a & b) * c | a * (b & c)
    *   |   | (a | b) * c | a * (b | c)
    *   /   | a / b * c | a * (b / c)
    *   === | (a === b) * c | a * (b === c)
    *   >=  | (a >= b) * c | a * (b >= c)
    *   >   | (a > b) * c | a * (b > c)
    *   <=  | (a <= b) * c | a * (b <= c)
    *   <   | (a < b) * c | a * (b < c)
    *   *   | a * b * c | a * (b * c)
    *   !== | (a !== b) * c | a * (b !== c)
    *   ||  | (a || b) * c | a * (b || c)
    *   <<  | (a << b) * c | a * (b << c)
    *   >>  | (a >> b) * c | a * (b >> c)
    *   -   | (a - b) * c | a * (b - c)
    !== +   | a + b !== c | a !== b + c
    !== &&  | (a && b) !== c | a !== (b && c)
    !== &   | (a & b) !== c | a !== (b & c)
    !== |   | (a | b) !== c | a !== (b | c)
    !== /   | a / b !== c | a !== b / c
    !== === | a === b !== c | a !== (b === c)
    !== >=  | a >= b !== c | a !== b >= c
    !== >   | a > b !== c | a !== b > c
    !== <=  | a <= b !== c | a !== b <= c
    !== <   | a < b !== c | a !== b < c
    !== *   | a * b !== c | a !== b * c
    !== !== | a !== b !== c | a !== (b !== c)
    !== ||  | (a || b) !== c | a !== (b || c)
    !== <<  | a << b !== c | a !== b << c
    !== >>  | a >> b !== c | a !== b >> c
    !== -   | a - b !== c | a !== b - c
    ||  +   | a + b || c | a || b + c
    ||  &&  | a && b || c | a || b && c
    ||  &   | a & b || c | a || b & c
    ||  |   | a | b || c | a || b | c
    ||  /   | a / b || c | a || b / c
    ||  === | a === b || c | a || b === c
    ||  >=  | a >= b || c | a || b >= c
    ||  >   | a > b || c | a || b > c
    ||  <=  | a <= b || c | a || b <= c
    ||  <   | a < b || c | a || b < c
    ||  *   | a * b || c | a || b * c
    ||  !== | a !== b || c | a || b !== c
    ||  ||  | a || b || c | a || (b || c)
    ||  <<  | a << b || c | a || b << c
    ||  >>  | a >> b || c | a || b >> c
    ||  -   | a - b || c | a || b - c
    <<  +   | a + b << c | a << b + c
    <<  &&  | (a && b) << c | a << (b && c)
    <<  &   | (a & b) << c | a << (b & c)
    <<  |   | (a | b) << c | a << (b | c)
    <<  /   | a / b << c | a << b / c
    <<  === | (a === b) << c | a << (b === c)
    <<  >=  | (a >= b) << c | a << (b >= c)
    <<  >   | (a > b) << c | a << (b > c)
    <<  <=  | (a <= b) << c | a << (b <= c)
    <<  <   | (a < b) << c | a << (b < c)
    <<  *   | a * b << c | a << b * c
    <<  !== | (a !== b) << c | a << (b !== c)
    <<  ||  | (a || b) << c | a << (b || c)
    <<  <<  | a << b << c | a << (b << c)
    <<  >>  | a >> b << c | a << (b >> c)
    <<  -   | a - b << c | a << b - c
    >>  +   | a + b >> c | a >> b + c
    >>  &&  | (a && b) >> c | a >> (b && c)
    >>  &   | (a & b) >> c | a >> (b & c)
    >>  |   | (a | b) >> c | a >> (b | c)
    >>  /   | a / b >> c | a >> b / c
    >>  === | (a === b) >> c | a >> (b === c)
    >>  >=  | (a >= b) >> c | a >> (b >= c)
    >>  >   | (a > b) >> c | a >> (b > c)
    >>  <=  | (a <= b) >> c | a >> (b <= c)
    >>  <   | (a < b) >> c | a >> (b < c)
    >>  *   | a * b >> c | a >> b * c
    >>  !== | (a !== b) >> c | a >> (b !== c)
    >>  ||  | (a || b) >> c | a >> (b || c)
    >>  <<  | a << b >> c | a >> (b << c)
    >>  >>  | a >> b >> c | a >> (b >> c)
    >>  -   | a - b >> c | a >> b - c
    -   +   | a + b - c | a - (b + c)
    -   &&  | (a && b) - c | a - (b && c)
    -   &   | (a & b) - c | a - (b & c)
    -   |   | (a | b) - c | a - (b | c)
    -   /   | a / b - c | a - b / c
    -   === | (a === b) - c | a - (b === c)
    -   >=  | (a >= b) - c | a - (b >= c)
    -   >   | (a > b) - c | a - (b > c)
    -   <=  | (a <= b) - c | a - (b <= c)
    -   <   | (a < b) - c | a - (b < c)
    -   *   | a * b - c | a - b * c
    -   !== | (a !== b) - c | a - (b !== c)
    -   ||  | (a || b) - c | a - (b || c)
    -   <<  | (a << b) - c | a - (b << c)
    -   >>  | (a >> b) - c | a - (b >> c)
    -   -   | a - b - c | a - (b - c)
    |}]

let%expect_test "unary operators and their spacing" =
  List.iter show
    [
      Unary (Neg, a);
      Unary (Not, a);
      Unary (Plus, a);
      Unary (Neg, Unary (Neg, a));
      Unary (Plus, Unary (Plus, a));
      Unary (Neg, Unary (Plus, a));
      Unary (Not, Unary (Not, a));
      Unary (Neg, Binary (Add, a, b));
      Binary (Add, a, Unary (Plus, b));
      Binary (Sub, a, Unary (Neg, b));
      Binary (Mul, Unary (Neg, a), b);
      Unary (Neg, Number (-1.));
      Unary (Not, Cond (a, b, c));
      Unary (Neg, Call (v "f", [ a ]));
    ];
  [%expect
    {|
    -a
    !a
    +a
    - -a
    + +a
    -+a
    !!a
    -(a + b)
    a + +b
    a - -b
    -a * b
    - -1
    !(a ? b : c)
    -f(a)
    |}]

let%expect_test "number literals" =
  List.iter
    (fun x -> show (Number x))
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
      Float.nan;
      Float.infinity;
      Float.neg_infinity;
      float_of_int 2147483647;
      float_of_int (-2147483648);
    ];
  [%expect
    {|
    0
    -0
    1
    -1
    0.5
    0.10000000000000001
    -0.10000000000000001
    16777217
    16777216
    4.9406564584124654e-324
    1.7976931348623157e+308
    2.2250738585072014e-308
    1e+21
    9.9999999999999995e-08
    1.2345678901234568e+17
    NaN
    Infinity
    -Infinity
    2147483647
    -2147483648
    |}]

let%expect_test "number literals read back to the same bits" =
  let bad = ref 0 in
  List.iter
    (fun x ->
      let s = Js_print.expr (Number x) in
      let back = if s = "NaN" then Float.nan else float_of_string s in
      if
        Int64.bits_of_float back <> Int64.bits_of_float x
        && not (Float.is_nan x)
      then (
        incr bad;
        Fmt.pr "%h printed %s@." x s))
    [
      0.;
      -0.;
      0.1;
      -0.1;
      16777217.;
      4.9e-324;
      Float.max_float;
      Float.min_float;
      1e21;
      1e-7;
      1. /. 3.;
      Float.pi;
      -.Float.pi;
      Float.infinity;
      Float.neg_infinity;
    ];
  Fmt.pr "%d mismatches@." !bad;
  [%expect {| 0 mismatches |}]

let%expect_test "bigint literals" =
  List.iter
    (fun n -> show (Bigint n))
    [
      0L;
      1L;
      -1L;
      Int64.min_int;
      Int64.max_int;
      9007199254740993L;
      -9007199254740993L;
    ];
  [%expect
    {|
    0n
    1n
    -1n
    -9223372036854775808n
    9223372036854775807n
    9007199254740993n
    -9007199254740993n
    |}]

let%expect_test "negative literals in every position" =
  let n = Number (-1.) and big = Bigint (-2L) in
  List.iter show
    [
      Binary (Sub, a, n);
      Binary (Sub, n, a);
      Binary (Mul, n, big);
      Member (n, Ident.v "x");
      Index (n, a);
      Index (a, n);
      Call (v "f", [ n; big ]);
      Cond (n, n, n);
      Array [ n; big ];
      Object [ (Ident.v "k", n) ];
    ];
  [%expect
    {|
    a - -1
    -1 - a
    -1 * -2n
    (-1).x
    (-1)[a]
    a[-1]
    f(-1, -2n)
    -1 ? -1 : -1
    [-1, -2n]
    { k: -1 }
    |}]

let%expect_test "members, indexing, calls, constructors" =
  let m = Global Global.Math and id = Ident.v in
  List.iter show
    [
      Call (Member (m, id "floor"), [ Binary (Div, a, Number 4.) ]);
      Member (Number 1., id "toString");
      Member (Bigint 1L, id "toString");
      Index (Member (a, id "buf"), Binary (Add, b, c));
      Index (Binary (Add, a, b), c);
      Member (Cond (a, b, c), id "x");
      Member (Call (v "f", []), id "x");
      Call (Call (v "f", [ a ]), [ b ]);
      New (Global Global.Float32_array, [ Number 8. ]);
      Member (New (Global Global.Float32_array, [ Number 8. ]), id "length");
      New (Call (v "f", []), [ a ]);
      New (Member (a, id "C"), []);
      Call (Global Global.Big_int, [ Binary (Sub, a, b) ]);
    ];
  [%expect
    {|
    Math.floor(a / 4)
    (1).toString
    (1n).toString
    a.buf[b + c]
    (a + b)[c]
    (a ? b : c).x
    f().x
    f(a)(b)
    new Float32Array(8)
    new Float32Array(8).length
    new (f())(a)
    new a.C()
    BigInt(a - b)
    |}]

let%expect_test "conditionals nest to the right and parenthesise the test" =
  List.iter show
    [
      Cond (a, b, Cond (b, c, a));
      Cond (Cond (a, b, c), b, c);
      Cond (a, Cond (a, b, c), c);
      Cond (Binary (Lt, a, b), Binary (Add, a, b), Binary (Or, a, b));
      Binary (Add, Cond (a, b, c), a);
      Binary (Add, a, Cond (a, b, c));
    ];
  [%expect
    {|
    a ? b : b ? c : a
    (a ? b : c) ? b : c
    a ? a ? b : c : c
    a < b ? a + b : a || b
    (a ? b : c) + a
    a + (a ? b : c)
    |}]

let%expect_test "strings" =
  List.iter
    (fun s -> show (String s))
    [
      "kind";
      "a\"b";
      "a\\b";
      "a\nb";
      "a\rb";
      "a\tb";
      "\001";
      "\xe2\x80\xa8";
      "\xe2\x80\xa9";
      "\xe2\x82\xac";
      "";
    ];
  [%expect
    {|
    "kind"
    "a\"b"
    "a\\b"
    "a\nb"
    "a\rb"
    "a\tb"
    "\u0001"
    "\u2028"
    "\u2029"
    "€"
    ""
    |}]

let%expect_test "objects, arrays, and an object at statement start" =
  let id = Ident.v in
  show (Object []);
  show (Object [ (id "kind", String "x"); (id "n", Number 1.) ]);
  show (Array []);
  show (Array [ a; Binary (Add, a, b) ]);
  Fmt.pr "%s"
    (Js_print.stmts
       [
         Stmt.Expr (Object [ (id "k", a) ]);
         Stmt.Expr (Member (Object [ (id "k", a) ], id "k"));
         Stmt.Return (Some (Object [ (id "k", a) ]));
       ]);
  [%expect
    {|
    {}
    { kind: "x", n: 1 }
    []
    [a, a + b]
    ({ k: a });
    ({ k: a }.k);
    return { k: a };
    |}]

let%expect_test "fully parenthesised mode" =
  let e = Binary (Add, a, Binary (Mul, Unary (Neg, b), Call (v "f", [ c ]))) in
  Fmt.pr "%s@.%s@." (Js_print.expr e) (Js_print.expr ~parens:`All e);
  [%expect {|
    a + -b * f(c)
    a + ((-b) * f(c))
    |}]

let%expect_test "statements and a program" =
  let id = Ident.v in
  let body =
    [
      Stmt.Let (id "x", Number 0.);
      Stmt.For
        {
          var = id "i";
          init = Number 0.;
          test = Binary (Lt, v "i", v "n");
          body =
            [
              Stmt.Assign (Lvar (id "x"), Plus_eq, Index (a, v "i"));
              Stmt.If
                ( Binary (Gt, v "x", Number 9.),
                  [ Stmt.Return (Some (Object [ (id "kind", String "big") ])) ],
                  [] );
              Stmt.If
                ( Unary (Not, v "ok"),
                  [ Stmt.Assign (Lindex (a, v "i"), Eq, Number 1.) ],
                  [ Stmt.Assign (Lvar (id "x"), Minus_eq, Number 1.) ] );
            ];
        };
      Stmt.Return None;
    ]
  in
  let entry =
    { Func.name = id "kernel"; params = [ id "a"; id "n"; id "ok" ]; body }
  in
  let program =
    { Program.prelude = [ Stmt.Const (id "zero", Number 0.) ]; entry }
  in
  Fmt.pr "%s---@.%s" (Js_print.script program) (Js_print.factory_body program);
  [%expect
    {|
    "use strict";
    const zero = 0;
    function kernel(a, n, ok) {
      let x = 0;
      for (let i = 0; i < n; i++) {
        x += a[i];
        if (x > 9) {
          return { kind: "big" };
        }
        if (!ok) {
          a[i] = 1;
        } else {
          x -= 1;
        }
      }
      return;
    }
    ---
    "use strict";
    const zero = 0;
    function kernel(a, n, ok) {
      let x = 0;
      for (let i = 0; i < n; i++) {
        x += a[i];
        if (x > 9) {
          return { kind: "big" };
        }
        if (!ok) {
          a[i] = 1;
        } else {
          x -= 1;
        }
      }
      return;
    }
    return kernel;
    |}]

let%expect_test "printing twice is byte-identical, and so is rebuilding" =
  let build () =
    let id = Ident.v in
    {
      Program.prelude = [];
      entry =
        {
          Func.name = id "k";
          params = [ id "a" ];
          body =
            [
              Stmt.For
                {
                  var = id "i";
                  init = Number 0.;
                  test = Binary (Lt, v "i", Number 3.);
                  body = [ Stmt.Assign (Lindex (a, v "i"), Eq, Number 0.1) ];
                };
            ];
        };
    }
  in
  let p = build () in
  Fmt.pr "%b %b@."
    (Js_print.script p = Js_print.script p)
    (Js_print.script p = Js_print.script (build ()));
  [%expect {| true true |}]
