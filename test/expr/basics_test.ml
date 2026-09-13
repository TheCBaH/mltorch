(* Leaf-module behaviour. The public-surface properties -- namespace isolation
   and the privacy of the AST -- are type-level and live in namespace_safety.t;
   they cannot be tested from here, because a bad reference in an ordinary
   module just fails this library to build. *)

open Expr

let%expect_test "Axis: order and rendering" =
  Fmt.pr "@[%a@]@." (Fmt.list ~sep:(Fmt.any ",") Axis.pp) Axis.all;
  [%expect {| N,T,D,H,W,C |}];
  Fmt.pr "%a@."
    (Fmt.list ~sep:(Fmt.any ",") Fmt.int)
    (List.map Axis.to_int Axis.all);
  [%expect {| 0,1,2,3,4,5 |}];
  (* [compare] follows the frame order, which is what keeps a Coord printed in
     N/T/D/H/W/C order rather than alphabetically. *)
  Fmt.pr "%b %b %b@."
    (Axis.compare Axis.N Axis.C < 0)
    (Axis.equal Axis.H Axis.H) (Axis.equal Axis.H Axis.W);
  [%expect {| true true false |}]

let%expect_test "Coord: access, update, traversal" =
  let c = Coord.of_fn Axis.to_int in
  Fmt.pr "%a@." (Coord.pp Fmt.int) c;
  [%expect {| 0,1,2,3,4,5 |}];
  (* [get] must agree with [of_fn] on every axis, and [set] must touch only the
     named one -- a swapped field here is the classic silent D/H mix-up. *)
  Fmt.pr "%b@." (List.for_all (fun a -> Coord.get c a = Axis.to_int a) Axis.all);
  [%expect {| true |}];
  Fmt.pr "%a@." (Coord.pp Fmt.int) (Coord.set c Axis.H 99);
  [%expect {| 0,1,2,99,4,5 |}];
  Fmt.pr "%a@." (Coord.pp Fmt.int) (Coord.map (fun x -> x * 10) c);
  [%expect {| 0,10,20,30,40,50 |}];
  Fmt.pr "%a@." (Coord.pp Fmt.int) (Coord.mapi (fun a x -> Axis.to_int a + x) c);
  [%expect {| 0,2,4,6,8,10 |}];
  Fmt.pr "%d %b@." (Coord.fold ( + ) 0 c) (Coord.for_all (fun x -> x < 6) c);
  [%expect {| 15 true |}]

let%expect_test "Coord: separator does not break under a narrow margin" =
  (* [Fmt.comma] is a breakable ",@ ": using it would let a long symbolic
     coordinate wrap mid-Load and move every golden that prints one. *)
  let buf = Buffer.create 64 in
  let fmt = Format.formatter_of_buffer buf in
  Format.pp_set_margin fmt 10;
  Coord.pp Fmt.int fmt (Coord.of_fn (fun _ -> 123456));
  Format.pp_print_flush fmt ();
  Fmt.pr "%S@." (Buffer.contents buf);
  [%expect {| "123456,123456,123456,123456,123456,123456" |}]

let%expect_test "Scalar: carrier names and Leibniz equality" =
  Fmt.pr "%a %a %a@." Scalar.pp Scalar.Bool Scalar.pp Scalar.Float Scalar.pp
    Scalar.I64;
  [%expect {| bool float i64 |}];
  (* Every same-constructor pair proves equal; every distinct pair does not --
     the exhaustive [None] arm in [Scalar.equal] is what a wildcard branch
     there would silently stop covering if a carrier were ever added. *)
  let same : type a. a Scalar.t -> bool =
   fun s -> Option.is_some (Scalar.equal s s)
  in
  Fmt.pr "%b %b %b@." (same Scalar.Bool) (same Scalar.Float) (same Scalar.I64);
  [%expect {| true true true |}];
  Fmt.pr "%b %b %b@."
    (Option.is_some (Scalar.equal Scalar.Bool Scalar.Float))
    (Option.is_some (Scalar.equal Scalar.Float Scalar.I64))
    (Option.is_some (Scalar.equal Scalar.I64 Scalar.Bool));
  [%expect {| false false false |}];
  (* [Refl] genuinely refines the type, not just witnesses a bool: this cast
     only type-checks because matching [Some Refl] unifies ['a] and ['b]. *)
  let cast : type a b. a Scalar.t -> b Scalar.t -> a -> b option =
   fun expected actual x ->
    match Scalar.equal expected actual with Some Refl -> Some x | None -> None
  in
  Fmt.pr "%b %b@."
    (cast Scalar.I64 Scalar.I64 9_007_199_254_740_993L
    = Some 9_007_199_254_740_993L)
    (cast Scalar.I64 Scalar.Float 1L = None);
  [%expect {| true true |}]

let%expect_test "Scalar: packed existential round trip" =
  let pi64 = Scalar.Pack (Scalar.I64, 9_007_199_254_740_993L) in
  let pbool = Scalar.Pack (Scalar.Bool, true) in
  Fmt.pr "%b %b %b %b@."
    (Scalar.unpack Scalar.I64 pi64 = Some 9_007_199_254_740_993L)
    (Scalar.unpack Scalar.Float pi64 = None)
    (Scalar.unpack Scalar.Bool pbool = Some true)
    (Scalar.unpack Scalar.I64 pbool = None);
  [%expect {| true true true true |}]

let%expect_test "Value: exact I64 arithmetic, no float intermediary" =
  let open Value in
  (* No [Float_to_i64] node in any tree here, so [eval_float] is provably
     never called -- these trees are closed over [I64_const]/[I64_binary]. *)
  let eval_i64 e =
    match eval_i64 ~eval_float:(fun _ -> assert false) e with
    | Ok v -> v
    | Error _ -> assert false
  in
  (* The design's canonical example: exact above float's 2^53 mantissa. A
     float round trip would have silently changed this value. *)
  let big = i64_const 9_007_199_254_740_993L in
  Fmt.pr "%Ld@." (eval_i64 (i64_add big (i64_const 1L)));
  [%expect {| 9007199254740994 |}];
  (* Same-width modular wraparound is the documented policy, not an error. *)
  Fmt.pr "%Ld %Ld@."
    (eval_i64 (i64_add (i64_const Int64.max_int) (i64_const 1L)))
    (eval_i64 (i64_sub (i64_const Int64.min_int) (i64_const 1L)));
  [%expect {| -9223372036854775808 9223372036854775807 |}];
  Fmt.pr "%Ld@." (eval_i64 (i64_mul (i64_const 6L) (i64_const 7L)));
  [%expect {| 42 |}]

let%expect_test
    "Value: I64_to_float through the real (environment-carrying) evaluator" =
  let open Value in
  let e =
    i64_to_float (i64_add (i64_const 9_007_199_254_740_993L) (i64_const 1L))
  in
  Fmt.pr "%a@." Pp.value e;
  [%expect {| i64_to_float((9007199254740993 + 1)) |}];
  Fmt.pr "size=%d depth=%d@." (Fold.size e) (Fold.depth e);
  [%expect {| size=4 depth=3 |}];
  let env =
    {
      Eval.Env.load = (fun _ _ -> assert false);
      load_index = (fun _ _ -> assert false);
    }
  in
  let output = Coord.of_fn (fun _ -> 0) in
  (* [Fmt.float]'s default %g-style printer rounds to a handful of significant
     digits, which would hide the very exactness this test exists to check
     (9007199254740994. is representable exactly as a double: it is 2^53 + 2,
     even and so on the post-2^53 grid of representable integers) -- an exact
     equality against the expected [float] is the real assertion, not the
     printed form. *)
  (match Eval.value env ~output e with
  | Ok v -> Fmt.pr "eval ok, exact: %b@." (Float.equal v 9_007_199_254_740_994.)
  | Error _ as r ->
      Fmt.pr "%a@."
        (Core.Pretty.err_result ~ok:Fmt.float ~error:Eval.pp_error)
        r);
  [%expect {| eval ok, exact: true |}]

let%expect_test "Value: i64_of_float boundaries, standalone" =
  let open Value in
  let pp =
    Core.Pretty.err_result ~ok:Fmt.int64 ~error:pp_i64_from_float_error
  in
  (* Finite, in-range: truncates toward zero, both signs. *)
  Fmt.pr "%a %a@." pp (i64_of_float 3.7) pp (i64_of_float (-3.7));
  [%expect {| 3 -3 |}];
  (* The exact power-of-two boundaries: [-2^63] is in range (representable and
     admitted); [2^63] itself is the first REJECTED value, not
     [Int64.max_int]'s float approximation (which would already have rounded
     up to this same boundary and be wrongly admitted by a max_int-based
     check). *)
  Fmt.pr "%a@." pp (i64_of_float (-9_223_372_036_854_775_808.));
  [%expect {| -9223372036854775808 |}];
  Fmt.pr "%a@." pp (i64_of_float 9_223_372_036_854_775_808.);
  [%expect {| Float-to-I64 cast of 0x1p+63, outside [-2^63, 2^63) |}];
  Fmt.pr "%a@." pp (i64_of_float Float.nan);
  [%expect {| Float-to-I64 cast of NaN |}];
  Fmt.pr "%a %a@." pp
    (i64_of_float Float.infinity)
    pp
    (i64_of_float Float.neg_infinity);
  [%expect
    {| Float-to-I64 cast of an infinite value Float-to-I64 cast of an infinite value |}]

let%expect_test
    "Value: Float_to_i64 round trip through the real evaluator, including its \
     structured errors" =
  let open Value in
  let env =
    {
      Eval.Env.load = (fun _ _ -> assert false);
      load_index = (fun _ _ -> assert false);
    }
  in
  let output = Coord.of_fn (fun _ -> 0) in
  let pp = Core.Pretty.err_result ~ok:Fmt.float ~error:Eval.pp_error in
  (* i64_to_float(float_to_i64(x)): a valid in-range float round-trips exactly
     through the exact-int64 domain and back, going through [go]'s real
     [I64_to_float] arm, which in turn calls [Value.eval_i64]'s [Float_to_i64]
     arm, which evaluates its [float t] operand through [go] itself -- the
     mutual dependency this slice introduced, exercised end to end. *)
  let round_trip x = i64_to_float (float_to_i64 (const x)) in
  Fmt.pr "%a@." pp (Eval.value env ~output (round_trip 42.));
  [%expect {| 42 |}];
  (* A NaN operand, reached only by actually evaluating the embedded [float t]
     child (not a static property of the tree), surfaces [Eval]'s own
     structured error -- not a wrapped/silent value. *)
  Fmt.pr "%a@." pp (Eval.value env ~output (round_trip Float.nan));
  [%expect {| Float-to-I64 cast of NaN |}];
  Fmt.pr "%a@." pp (Eval.value env ~output (round_trip Float.infinity));
  [%expect {| Float-to-I64 cast of an infinite value |}];
  Fmt.pr "%a@." pp
    (Eval.value env ~output (round_trip 9_223_372_036_854_775_808.));
  [%expect {| Float-to-I64 cast of 0x1p+63, outside [-2^63, 2^63) |}]

let%expect_test "Fold: Float_to_i64's operand is not a closed leaf" =
  (* [int64 Value.t] stopped being closed the moment [Float_to_i64] existed:
     its operand is the unbounded float language, so every scope-aware [Fold]
     query must still see inside it through an enclosing [I64_to_float]. This
     reduction's [var] is bound only inside the [Reduce] node the builder
     produces; taking [body] out on its own (as this test does deliberately,
     never through the public API otherwise) is exactly what makes the
     reference free -- checking that against the WRAPPED tree is what proves
     [Fold]/[Check]/[Rewrite]'s int64-side companions actually recurse into
     [Float_to_i64] rather than stopping at [I64_to_float]. *)
  let open Builder.Syntax in
  let e =
    Builder.run
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
         (fun r ->
           let+ () = Builder.return () in
           Value.value_of_index (Index.of_position r)))
  in
  match e with
  | Value.Reduce red ->
      let escaped =
        Value.i64_to_float (Value.float_to_i64 red.Reduction.body)
      in
      Fmt.pr "free reducers reached through the cast: %d@."
        (Reduce_var.Set.cardinal (Fold.free_reducers escaped));
      [%expect {| free reducers reached through the cast: 1 |}]
  | _ -> assert false

let%expect_test "Source: stateless bijection and rendering" =
  let s = Source.create 7 in
  (* [pp] must match lib/native's [Tensor_id.pp] so a printed Load stays
     byte-identical across the migration. *)
  Fmt.pr "%a %d@." Source.pp s (Source.to_int s);
  [%expect {| t7 7 |}];
  (* The round trip is what lets the native adapter map Tensor_id <-> Source
     with no side table, so nothing depends on allocation order. *)
  Fmt.pr "%b@."
    (List.for_all
       (fun n -> Source.to_int (Source.create n) = n)
       [ 0; 1; 42; 1000 ]);
  [%expect {| true |}];
  Fmt.pr "%b %b@."
    (Source.equal s (Source.create 7))
    (Source.equal s (Source.create 8));
  [%expect {| true false |}]

let%expect_test "Builder: one supply, resumed rather than restarted" =
  let open Builder.Syntax in
  let two =
    let* a = Builder.fresh_reduce in
    let+ b = Builder.fresh_reduce in
    (a, b)
  in
  let (a, b), st = Builder.run_from Builder.initial two in
  Fmt.pr "distinct within one run: %b@." (not (Reduce_var.equal a b));
  [%expect {| distinct within one run: true |}];
  (* The point of [run_from]: a caller that keeps the state can carry on minting
     fresh identities in the same namespace. Restarting from [initial] collides,
     which is not a defect -- identity is local to one expression -- but it is
     exactly why a fragment must be freshened before it is composed under
     another's binder. *)
  let (c, _), _ = Builder.run_from st two in
  let d, _ = Builder.run_from Builder.initial Builder.fresh_reduce in
  Fmt.pr "resumed is fresh: %b   restarted collides: %b@."
    (not (Reduce_var.equal a c))
    (Reduce_var.equal a d);
  [%expect {| resumed is fresh: true   restarted collides: true |}]
