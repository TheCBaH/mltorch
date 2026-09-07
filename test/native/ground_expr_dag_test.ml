(* Exercises the hash-consed [Ground_expr] representation itself: sharing,
   cross-arena structural equality/hashing, lazy [Select] evaluation, and that
   [project]/[normalise] preserve (or increase) sharing rather than unfolding
   it. *)

let cell n =
  {
    Ground_expr.Cell.origin =
      Ground_expr.Origin.Src (Graph_ir.Tensor_id.of_int n);
    coord = Vec6.origin;
  }

let%expect_test "a diamond shares its common subterm: size counts it once" =
  let a = Ground_expr.Arena.create () in
  let x = Ground_expr.cell a (cell 0) and y = Ground_expr.cell a (cell 1) in
  let shared = Ground_expr.binary a Expr.Value.Add x y in
  let top = Ground_expr.binary a Expr.Value.Mul shared shared in
  (* Unshared: x, y, add (left) + x, y, add (right) + mul = 7. Shared: x, y,
     add, mul = 4. *)
  Format.printf "size=%d@." (Ground_expr.size top);
  [%expect {| size=4 |}]

let%expect_test "repeated squaring: DAG size is linear, not exponential" =
  let a = Ground_expr.Arena.create () in
  let x = Ground_expr.cell a (cell 0) in
  let rec squares n acc =
    if n = 0 then acc
    else squares (n - 1) (Ground_expr.binary a Expr.Value.Mul acc acc)
  in
  let top = squares 30 x in
  (* An unshared tree of 30 squarings would have 2^30 leaves; the DAG has one
     node per squaring plus the leaf. *)
  Format.printf "size=%d@." (Ground_expr.size top);
  [%expect {| size=31 |}]

let%expect_test
    "hash agrees for structurally-equal terms built in different arenas" =
  let build () =
    let a = Ground_expr.Arena.create () in
    let x = Ground_expr.cell a (cell 0) and y = Ground_expr.cell a (cell 1) in
    Ground_expr.binary a Expr.Value.Add x y
  in
  let e1 = build () and e2 = build () in
  Format.printf "hash_eq=%b equal=%b compare=%d@."
    (Ground_expr.hash e1 = Ground_expr.hash e2)
    (Ground_expr.equal e1 e2)
    (Ground_expr.compare e1 e2);
  [%expect {| hash_eq=true equal=true compare=0 |}]

let%expect_test "compare is antisymmetric across arenas, and exact on floats" =
  let a1 = Ground_expr.Arena.create () and a2 = Ground_expr.Arena.create () in
  let e1 =
    Ground_expr.binary a1 Expr.Value.Add
      (Ground_expr.cell a1 (cell 0))
      (Ground_expr.const a1 1.)
  in
  let e2 =
    Ground_expr.binary a2 Expr.Value.Add
      (Ground_expr.cell a2 (cell 0))
      (Ground_expr.const a2 2.)
  in
  Format.printf "cmp12=%d cmp21=%d@."
    (compare (Ground_expr.compare e1 e2) 0)
    (compare (Ground_expr.compare e2 e1) 0);
  (* Signed zero and NaN are compared by exact bits, never by [Float.equal]. *)
  let zero_pos = Ground_expr.const a1 0.0
  and zero_neg = Ground_expr.const a1 (-0.0) in
  let nan1 = Ground_expr.const a1 Float.nan
  and nan2 = Ground_expr.const a2 Float.nan in
  Format.printf "zero_equal=%b nan_equal_cross_arena=%b@."
    (Ground_expr.equal zero_pos zero_neg)
    (Ground_expr.equal nan1 nan2);
  [%expect
    {|
    cmp12=-1 cmp21=1
    zero_equal=false nan_equal_cross_arena=true |}]

let%expect_test "eval only evaluates the branch its guard selects" =
  let a = Ground_expr.Arena.create () in
  let bound = cell 0 and unbound = cell 1 in
  let guard =
    Ground_expr.lt a (Ground_expr.const a 0.) (Ground_expr.const a 1.)
  in
  let e =
    Ground_expr.select a guard (Ground_expr.cell a bound)
      (Ground_expr.cell a unbound)
  in
  let v = Ground_expr.Valuation.of_list [ (bound, 42.) ] in
  (* [unbound] has no binding; a naive eager evaluator would raise [Not_found]. *)
  Format.printf "v=%h@." (Ground_expr.eval e v);
  [%expect {| v=0x1.5p+5 |}]

let%expect_test "project preserves sharing rather than unfolding it" =
  let a = Ground_expr.Arena.create ()
  and scratch = Ground_expr.Arena.create () in
  let x = Ground_expr.cell a (cell 0) and y = Ground_expr.cell a (cell 1) in
  let shared = Ground_expr.binary a Expr.Value.Add x y in
  let top = Ground_expr.binary a Expr.Value.Mul shared shared in
  let projected =
    Ground_expr.project ~into:scratch ~boundary:(fun _ -> None) top
  in
  Format.printf "size=%d@." (Ground_expr.size projected);
  [%expect {| size=4 |}]

let%expect_test "pp renders a shared node once, as a let binding" =
  let a = Ground_expr.Arena.create () in
  let x = Ground_expr.cell a (cell 0) and y = Ground_expr.cell a (cell 1) in
  let shared = Ground_expr.binary a Expr.Value.Add x y in
  let top = Ground_expr.binary a Expr.Value.Mul shared shared in
  Format.printf "%a@." Ground_expr.pp top;
  [%expect {|
    let l0 = (src.t0(0) + src.t1(0)) in
    (l0 * l0) |}]

let%expect_test "pp of an unshared term is byte-identical to the old syntax" =
  let a = Ground_expr.Arena.create () in
  let e =
    Ground_expr.binary a Expr.Value.Add
      (Ground_expr.cell a (cell 0))
      (Ground_expr.cell a (cell 1))
  in
  Format.printf "%a@." Ground_expr.pp e;
  [%expect {| (src.t0(0) + src.t1(0)) |}]

let%expect_test "a wide max-pool-shaped accumulator does not blow up compare" =
  (* Two independently built copies of a 2000-deep left-associated Add chain,
     each accumulator reused as BOTH operands of the next step's sibling
     comparison walk (i.e. comparing the two chains pairwise at every prefix)
     must not cost exponential time -- memoised pair comparison is what makes
     this finish at all. *)
  let a = Ground_expr.Arena.create () in
  let leaf = Ground_expr.cell a (cell 0) in
  let rec chain n acc =
    if n = 0 then acc
    else chain (n - 1) (Ground_expr.binary a Expr.Value.Add acc leaf)
  in
  let e1 = chain 2000 leaf and e2 = chain 2000 leaf in
  Format.printf "equal=%b size=%d@." (Ground_expr.equal e1 e2)
    (Ground_expr.size e1);
  [%expect {| equal=true size=2001 |}]
