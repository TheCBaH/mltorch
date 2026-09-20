(* Exercises [Semantics.TYPED_SEMANTICS] (semantics.ml), Direct's and
   Symbolic's new instances of it (direct.ml/symbolic.ml) -- typed
   Direct/Symbolic semantics and checked operation dispatch. Follows
   symbolic_test.ml's own [eval_expr]/[build] idiom for the Direct/Symbolic
   agreement checks, deliberately WITHOUT [Schedule.ground]: Stage
   grounding/fusion still explicitly rejects [I64_to_float] (see
   ground_eval_i64_ground_test.ml) -- grounding an I64 expression is out of
   scope here. *)

let i64_fmt = Payload.Fmt Payload.I64
let one_cell = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let zero6 (z : 'r) : 'r Vec6.t = Vec6.of_fn (fun _ -> z)
let build = Expr.Builder.run

let eval_expr ~binding e c =
  Err.or_raise ~pp_error:Expr.Eval.pp_error
    (Expr.Eval.value (Expr_bridge.env ~binding)
       ~output:(Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int c))
       e)

let%expect_test
    "Typed_semantics: Direct I64 arithmetic is exact, no float intermediary" =
  let x = Tensor.materialize_i64 one_cell (fun _ -> 9_007_199_254_740_993L) in
  let loaded = Direct.i64_load x (zero6 Direct.index_zero) in
  let sum =
    Direct.i64_binary Expr.Value.I64_add loaded
      (Direct.typed_const Expr.Scalar.I64 1L)
  in
  Format.printf "%Ld@." sum;
  [%expect {| 9007199254740994 |}];
  (* Proof this is genuinely exact, not silently lossy through float: the
     same "+1" collapses to a no-op once the operand has gone through a
     float, since [9_007_199_254_740_993.] and [+1.] round to the identical
     float -- exactly what "no float intermediary" is guarding against. *)
  Format.printf "float-lossy would give %b@."
    (Float.equal (9_007_199_254_740_993. +. 1.) 9_007_199_254_740_993.);
  [%expect {| float-lossy would give true |}]

let%expect_test
    "Typed_semantics: Direct typed_select picks the I64-comparison-selected \
     branch" =
  let a = Direct.typed_const Expr.Scalar.I64 3L in
  let b = Direct.typed_const Expr.Scalar.I64 5L in
  let lt = Direct.i64_lt a b in
  Format.printf "lt: %Ld@." (Direct.typed_select lt a b);
  [%expect {| lt: 3 |}];
  Format.printf "not lt: %Ld@." (Direct.typed_select (not lt) a b);
  [%expect {| not lt: 5 |}];
  Format.printf "eq: %b@." (Direct.i64_eq a a);
  [%expect {| eq: true |}];
  Format.printf "float_to_i64(i64_to_float 5L) = %Ld@."
    (Direct.float_to_i64 (Direct.i64_to_float b));
  [%expect {| float_to_i64(i64_to_float 5L) = 5 |}]

let%expect_test
    "Typed_semantics: Direct and Symbolic agree on load+binary+select, through \
     the real Eval/Expr_bridge path" =
  let module S = Symbolic in
  let cases = [ (10L, 3L); (-5L, 5L); (0L, 0L) ] in
  List.iter
    (fun (av, bv) ->
      let a = Tensor.materialize_i64 one_cell (fun _ -> av) in
      let b = Tensor.materialize_i64 one_cell (fun _ -> bv) in
      let a_sig =
        Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"a" ~shape:one_cell
          ~fmt:i64_fmt ()
      in
      let b_sig =
        Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"b" ~shape:one_cell
          ~fmt:i64_fmt ()
      in
      let direct =
        let la = Direct.i64_load a (zero6 Direct.index_zero) in
        let lb = Direct.i64_load b (zero6 Direct.index_zero) in
        let smaller = Direct.typed_select (Direct.i64_lt la lb) la lb in
        Direct.i64_to_float
          (Direct.i64_binary Expr.Value.I64_add smaller
             (Direct.typed_const Expr.Scalar.I64 100L))
      in
      (* Each combinator is itself a computation-to-computation function
         (threading the builder supply internally via [map2]/[map3], as
         [add]/[select] etc. already do above) -- so, like every op's own
         [pixel] composition, this chains plain [let]s of COMPUTATIONS, never
         unwrapping one with [Expr.Builder.Syntax]'s [let*] (that unwraps to
         the built VALUE, which none of [i64_lt]/[typed_select]/[i64_binary]
         accept). *)
      let e =
        build
          (let la = S.i64_load a_sig (zero6 S.index_zero) in
           let lb = S.i64_load b_sig (zero6 S.index_zero) in
           let lt = S.i64_lt la lb in
           let smaller = S.typed_select lt la lb in
           let hundred = S.typed_const Expr.Scalar.I64 100L in
           S.i64_to_float (S.i64_binary Expr.Value.I64_add smaller hundred))
      in
      let binding id = if id = a_sig.id then Some a else Some b in
      let symbolic = eval_expr ~binding e Vec6.origin in
      Format.printf "direct=%g symbolic=%g agree=%b@." direct symbolic
        (Core.Float_bits.equal_exact direct symbolic))
    cases;
  [%expect
    {|
    direct=103 symbolic=103 agree=true
    direct=95 symbolic=95 agree=true
    direct=100 symbolic=100 agree=true
    |}]

let%expect_test
    "Typed_semantics: I64 division truncates and fails alike on Direct and \
     Symbolic" =
  let module S = Symbolic in
  let cases = [ (7L, 2L); (-7L, 2L); (5L, 0L); (Int64.min_int, -1L) ] in
  List.iter
    (fun (x, y) ->
      let direct =
        match
          Direct.i64_binary Expr.Value.I64_div
            (Direct.typed_const Expr.Scalar.I64 x)
            (Direct.typed_const Expr.Scalar.I64 y)
        with
        | q -> Int64.to_string q
        | exception Err.Exn.E _ -> "error"
      in
      let symbolic =
        let e =
          build
            (S.i64_to_float
               (S.i64_binary Expr.Value.I64_div
                  (S.typed_const Expr.Scalar.I64 x)
                  (S.typed_const Expr.Scalar.I64 y)))
        in
        match
          Expr.Eval.value
            (Expr_bridge.env ~binding:(fun _ -> None))
            ~output:
              (Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int Vec6.origin))
            e
        with
        | Ok v -> Printf.sprintf "%.0f" v
        | Error _ -> "error"
      in
      Format.printf "%Ld / %Ld: direct=%s symbolic=%s@." x y direct symbolic)
    cases;
  [%expect
    {|
    7 / 2: direct=3 symbolic=3
    -7 / 2: direct=-3 symbolic=-3
    5 / 0: direct=error symbolic=error
    -9223372036854775808 / -1: direct=error symbolic=error |}]
