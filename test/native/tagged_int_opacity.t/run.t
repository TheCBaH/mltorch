Domain separation: what the type checker must reject.

Each [Core.Tagged_int.Make ()] application is its own type, so the ids of one
space cannot be passed for another's. Every negative case is paired with a
control that must still compile -- without the controls a broken harness would
reject everything and look like a pass.

  $ R=../../..
  $ check() { sh $R/test/native/tagged_int_opacity.sh $R/test/native/core_probe.exe "$1" "$2"; }

Two applications are incompatible, whatever their prefix.

  $ check "compare across applications" "ignore (A.equal a b)"
  compare across applications: rejected
  $ check "compare within one application" "ignore (A.equal a a)"
  compare within one application: COMPILES
  $ check "key one map by another's type" "ignore (A.Map.add b 0 A.Map.empty)"
  key one map by another's type: rejected
  $ check "key a map by its own type" "ignore (A.Map.add a 0 A.Map.empty)"
  key a map by its own type: COMPILES

A bare [int] is not a domain value. Entry is [of_int]; exit is a coercion.

  $ check "pass an int where a domain is wanted" "ignore (A.equal a n)"
  pass an int where a domain is wanted: rejected
  $ check "enter through of_int" "ignore (A.equal a (A.of_int n))"
  enter through of_int: COMPILES
  $ check "coerce a domain out to int" "ignore ((a :> int) + n)"
  coerce a domain out to int: COMPILES
  $ check "coerce an int into a domain" "ignore ((n :> A.t))"
  coerce an int into a domain: rejected

The counter that hands out the next id is typed too, so one space's watermark
cannot be passed where another's is wanted, nor can an id of one space raise
another's.

  $ check "alloc from another space's counter" "ignore (A.Next.alloc B.Next.first)"
  alloc from another space's counter: rejected
  $ check "alloc from its own counter" "ignore (A.Next.alloc A.Next.first)"
  alloc from its own counter: COMPILES
  $ check "raise a counter past another space's id" "ignore (A.Next.after b A.Next.first)"
  raise a counter past another space's id: rejected
  $ check "raise a counter past its own id" "ignore (A.Next.after a A.Next.first)"
  raise a counter past its own id: COMPILES
  $ check "compare counters across spaces" "ignore (A.Next.equal A.Next.first B.Next.first)"
  compare counters across spaces: rejected
  $ check "an int is not a counter" "ignore (A.Next.alloc n)"
  an int is not a counter: rejected
