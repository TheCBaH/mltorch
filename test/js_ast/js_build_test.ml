open Js_build

let id = Js_ident.v
let show e = Fmt.pr "%s@." (Js_print.expr (expr e))
let stmts l = Fmt.pr "%s" (Js_print.stmts l)

let x = Num.var (id "x")
and y = Num.var (id "y")

let i = Idx.var (id "i")
and j = Idx.var (id "j")

let b = Big.var (id "b")
and c = Big.var (id "c")

let%expect_test "float builders, none of which folds" =
  List.iter
    (fun e -> Fmt.pr "%s@." (Js_print.expr e))
    [
      expr (Num.add x (Num.const 0.));
      expr (Num.mul x (Num.const 1.));
      expr (Num.sub x (Num.const (-0.)));
      expr (Num.div x y);
      expr (Num.neg (Num.neg x));
      expr (Num.abs x);
      expr (Num.cos x);
      expr (Num.exp x);
      expr (Num.log x);
      expr (Num.sin x);
      expr (Num.sqrt x);
      expr (Num.trunc x);
      expr (Num.fround x);
      expr (Num.max x y);
      expr (Num.pow x y);
      expr (Num.of_idx i);
      expr (Num.of_big b);
      expr (Num.of_bits (Bits.var (id "h")));
      expr (select (Num.lt x y) x y);
    ];
  [%expect
    {|
    x + 0
    x * 1
    x - -0
    x / y
    - -x
    Math.abs(x)
    Math.cos(x)
    Math.exp(x)
    Math.log(x)
    Math.sin(x)
    Math.sqrt(x)
    Math.trunc(x)
    Math.fround(x)
    Math.max(x, y)
    Math.pow(x, y)
    i + 0
    Number(b)
    h
    x < y ? x : y
    |}]

let%expect_test "float predicates" =
  List.iter
    (fun e -> Fmt.pr "%s@." (Js_print.expr e))
    [
      expr (Num.eq x y);
      expr (Num.ne x y);
      expr (Num.lt x y);
      expr (Num.gt x y);
      expr (Num.is_nan x);
      expr (Num.is_finite x);
      expr (Pred.not_ (Num.is_nan x));
      expr (Pred.or_ (Num.gt x y) (Num.ne x x));
      expr Pred.false_;
    ];
  [%expect
    {|
    x === y
    x !== y
    x < y
    x > y
    Number.isNaN(x)
    Number.isFinite(x)
    !Number.isNaN(x)
    x > y || x !== x
    false
    |}]

let%expect_test "index builders fold only what is exact" =
  List.iter
    (fun e -> Fmt.pr "%s@." (Js_print.expr e))
    [
      expr (Idx.add i (Idx.const 0));
      expr (Idx.add (Idx.const 0) i);
      expr (Idx.add i j);
      expr (Idx.add i (Idx.const (-3)));
      expr (Idx.scale 1 i);
      expr (Idx.scale 4 i);
      expr (Idx.scale (-1) i);
      expr (Idx.scale 0 i);
      expr (Idx.scale 7 (Idx.const 0));
      expr (Idx.floor_div_pos i 4);
      expr (Idx.ceil_div_pos (Idx.add i (Idx.const (-3))) 4);
      expr (Idx.clamp_low i);
      expr (Idx.max i j);
      expr (Idx.min i j);
      expr (Idx.const (-5));
      expr (Idx.eq i j);
      expr (Idx.lt i j);
      expr (Idx.out_of_range i 8);
      expr (Idx.outside_int32 (Idx.add i j));
      expr (Idx.of_big_bounded b);
    ];
  [%expect
    {|
    i
    i
    i + j
    i + -3
    i
    4 * i
    -1 * i
    0 * i
    0
    Math.floor(i / 4)
    Math.ceil((i + -3) / 4)
    Math.max(0, i)
    Math.max(i, j)
    Math.min(i, j)
    -5
    i === j
    i < j
    i < 0 || i >= 8
    i + j < -2147483648 || i + j >= 2147483648
    Number(b)
    |}]

let%expect_test "a row-major offset of a W-only coordinate is the coordinate" =
  let extents = [ 1; 1; 1; 1; 5; 1 ] in
  let coords =
    [ Idx.const 0; Idx.const 0; Idx.const 0; Idx.const 0; i; Idx.const 0 ]
  in
  let offset =
    List.fold_left2
      (fun acc e c -> Idx.add (Idx.scale e acc) c)
      (List.hd coords) (List.tl extents) (List.tl coords)
  in
  show offset;
  [%expect {| i |}]

let%expect_test "int64 arithmetic always wraps" =
  List.iter
    (fun e -> Fmt.pr "%s@." (Js_print.expr e))
    [
      expr (Big.add_wrap b c);
      expr (Big.sub_wrap b c);
      expr (Big.mul_wrap b c);
      expr (Big.div_unchecked b c);
      expr (Big.const 0L);
      expr (Big.const (-1L));
      expr (Big.const Int64.min_int);
      expr (Big.of_idx i);
      expr (Big.of_num_trunc x);
      expr (Big.eq b c);
      expr (Big.lt b c);
    ];
  [%expect
    {|
    BigInt.asIntN(64, b + c)
    BigInt.asIntN(64, b - c)
    BigInt.asIntN(64, b * c)
    b / c
    0n
    -1n
    -9223372036854775808n
    BigInt(i)
    BigInt(Math.trunc(x))
    b === c
    b < c
    |}]

let%expect_test "bit patterns" =
  let h = Bits.var (id "h") in
  List.iter
    (fun e -> Fmt.pr "%s@." (Js_print.expr e))
    [
      expr (Bits.and_ (Bits.shr h 15) (Bits.const 1));
      expr (Bits.or_ (Bits.const 1024) h);
      expr (Bits.eq h (Bits.const 0));
      expr (Bits.shl16 h);
    ];
  [%expect {|
    h >> 15 & 1
    1024 | h
    h === 0
    h << 16
    |}]

let%expect_test "arrays, loads and stores" =
  let a = Arr.num (id "a") and w = Arr.big (id "w") and u = Arr.bits (id "u") in
  show (load a i);
  show (load w (Idx.add i j));
  show (load u i);
  show (Arr.new_float64 4);
  let scratch = Arr.new_uint32 1 in
  show scratch;
  show (Arr.view_float32 (Arr.new_uint32 1));
  show (Arr.literal [ Num.const 0.5; Num.const 2. ]);
  stmts
    [
      store a i (Num.add (load a i) x);
      store w i (Big.add_wrap (load w i) b);
      Stmt.const_arr (id "tab") scratch;
    ];
  [%expect
    {|
    a[i]
    w[i + j]
    u[i]
    new Float64Array(4)
    new Uint32Array(1)
    new Float32Array(new Uint32Array(1).buffer)
    [0.5, 2]
    a[i] = a[i] + x;
    w[i] = BigInt.asIntN(64, w[i] + b);
    const tab = new Uint32Array(1);
    |}]

let%expect_test "statements" =
  stmts
    [
      Stmt.let_num (id "s") (Num.const 0.);
      Stmt.let_big (id "n") (Big.const 0L);
      Stmt.for_ (id "i") ~lo:(Idx.const 0) ~hi:(Idx.const 4)
        [
          Stmt.assign_num (id "s") (Num.add x y);
          Stmt.assign_idx (id "k") (Idx.add i j);
          Stmt.assign_big (id "n") b;
          Stmt.incr_num (id "s") x;
          Stmt.decr_num (id "s") y;
          Stmt.if_ (Num.lt x y)
            [ Stmt.return_ (record [ ("kind", string "f") ]) ]
            [];
        ];
      Stmt.const_num (id "p") (Num.const 0.3275911);
      Stmt.const_bits (id "m") (Bits.and_ (Bits.var (id "h")) (Bits.const 1023));
      Stmt.return_null;
    ];
  [%expect
    {|
    let s = 0;
    let n = 0n;
    for (let i = 0; i < 4; i++) {
      s = x + y;
      k = i + j;
      n = b;
      s += x;
      s -= y;
      if (x < y) {
        return { kind: "f" };
      }
    }
    const p = 0.32759110000000002;
    const m = h & 1023;
    return null;
    |}]

let%expect_test "records, strings and calls" =
  Fmt.pr "%s@."
    (Js_print.expr (record [ ("kind", string "x"); ("ok", bool true) ]));
  Fmt.pr "%s@." (Js_print.expr (to_string b));
  show (Unsafe.call (id "erf") [ expr x ]);
  [%expect {|
    { kind: "x", ok: true }
    b.toString()
    erf(x)
    |}]
