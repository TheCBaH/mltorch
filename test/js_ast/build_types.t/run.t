The kinds of [Js_build]: what the type checker must reject. Each negative case
is paired with a control that must still compile; without the controls a broken
harness would reject everything and look like a pass.

  $ R=../../..
  $ check() { sh $R/test/js_ast/build_types.sh $R/test/js_ast/build_probe.exe "$1" "$2"; }

A Number is not a BigInt: [1n + 1] throws a TypeError at run time, so the
builder for one must not accept the other.

  $ check "float add on a bigint" "ignore (Num.add x b)"
  float add on a bigint: rejected
  $ check "float add on floats" "ignore (Num.add x x)"
  float add on floats: COMPILES
  $ check "wrapping add on a float" "ignore (Big.add_wrap b x)"
  wrapping add on a float: rejected
  $ check "wrapping add on bigints" "ignore (Big.add_wrap b b)"
  wrapping add on bigints: COMPILES

Only an index goes in a subscript, so a computed float cannot address an array.

  $ check "float as a subscript" "ignore (load a x)"
  float as a subscript: rejected
  $ check "index as a subscript" "ignore (load a i)"
  index as a subscript: COMPILES

Bit patterns are a kind of their own: no bitwise operator accepts an index, so
[i | 0] cannot be written.

  $ check "bitwise or on an index" "ignore (Bits.or_ i i)"
  bitwise or on an index: rejected
  $ check "bitwise or on patterns" "ignore (Bits.or_ h h)"
  bitwise or on patterns: COMPILES

The crossings are named: an index becomes a float or a bigint, and a bigint an
index only through the one bounded crossing.

  $ check "bigint as an index" "ignore (load a b)"
  bigint as an index: rejected
  $ check "bigint bounded to an index" "ignore (load a (Idx.of_big_bounded b))"
  bigint bounded to an index: COMPILES
