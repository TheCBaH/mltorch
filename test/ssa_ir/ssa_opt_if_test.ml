open Ssa_ir
open Ssa_fixtures
module B = Ssa_builder

(* If-conversion: a branch whose arms are safe to evaluate becomes a select.
   Each program runs before and after on inputs with NaN, signed zeros and the
   values either side of the thresholds; the outputs must be bit-identical, and
   a branch that could skip a failure or a side effect must stay. *)

let bufs =
  [
    buffer 0 ~h:1L ~w:4L Ssa_format.F32 Ssa_buffer.Input;
    buffer 1 ~h:1L ~w:8L Ssa_format.F32 Ssa_buffer.Output;
  ]

let idx bld n = B.index bld (Int64.of_int n)

let load_at bld i =
  B.load_f64 bld (buf 0) ~decode:Ssa_op.Decode.F32_to_f64
    (at bld ~h:(idx bld 0) ~w:i)

let store_at bld i x =
  B.store_f64 bld (buf 1) ~encode:Ssa_op.Encode.F32_round
    (at bld ~h:(idx bld 0) ~w:i)
    x

let lt bld a b = B.float_compare bld Ssa_op.Compare.Lt a b

let rec branches (r : Ssa_region.t) =
  List.fold_left
    (fun n -> function
      | Ssa_stmt.If { then_; else_; _ } ->
          n + 1 + branches then_ + branches else_
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          n + branches body
      | Ssa_stmt.Instr _ -> n)
    0 r.Ssa_region.body

let convert p =
  fst
    (Ssa_opt.run ~alias:Ssa_effects.Distinct_buffers
       ~passes:Ssa_opt.[ simplify; guards; simplify; convert_ifs; simplify ]
       p)

let inputs =
  [
    [| -1.; 0.; 3.; 9. |];
    [| nan; -0.; 6.; 6.0001 |];
    [| infinity; neg_infinity; 5.99; -3. |];
  ]

let run_on p input =
  let out = Array.make 8 0. in
  let memory = memory [ (0, floats input); (1, floats out) ] in
  let result = run_result p ~memory in
  (Result.map_error (fun e -> Fmt.str "%a" Ssa_interp.pp_error e) result, out)

let report name p =
  let q = convert p in
  let same =
    List.for_all
      (fun input ->
        let r, out = run_on p input and r', out' = run_on q input in
        r = r' && Array.for_all2 Core.Float_bits.equal_portable out out')
      inputs
  in
  Fmt.pr "%s: branches %d -> %d; identical: %b@." name
    (branches p.Ssa_program.entry)
    (branches q.Ssa_program.entry)
    same

let loop bld body =
  let B.Nil =
    B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 4) ~init:B.Nil (fun bld i B.Nil ->
        body bld i;
        B.Nil)
  in
  ()

let%expect_test "a clamp made of branches over values becomes selects" =
  report "relu6"
    (build ~buffers:bufs (fun bld ->
         loop bld (fun bld i ->
             let x = load_at bld i in
             let zero = B.f64 bld 0. and six = B.f64 bld 6. in
             let (B.Cons (y, B.Nil)) =
               B.if_ bld (lt bld x zero)
                 ~then_:(fun _ -> B.Cons (zero, B.Nil))
                 ~else_:(fun _ -> B.Cons (x, B.Nil))
             in
             let (B.Cons (z, B.Nil)) =
               B.if_ bld (lt bld six y)
                 ~then_:(fun _ -> B.Cons (six, B.Nil))
                 ~else_:(fun _ -> B.Cons (y, B.Nil))
             in
             store_at bld i z)));
  [%expect {| relu6: branches 2 -> 0; identical: true |}]

let%expect_test "a branch around a load proved in bounds is speculated" =
  report "guarded load"
    (build ~buffers:bufs (fun bld ->
         loop bld (fun bld i ->
             let x = load_at bld i in
             let zero = B.f64 bld 0. in
             let (B.Cons (v, B.Nil)) =
               B.if_ bld (lt bld x zero)
                 ~then_:(fun _ -> B.Cons (zero, B.Nil))
                 ~else_:(fun bld -> B.Cons (load_at bld i, B.Nil))
             in
             store_at bld i v)));
  [%expect {| guarded load: branches 1 -> 0; identical: true |}]

let%expect_test "loads in both arms stay one chain" =
  report "loads in both arms"
    (build ~buffers:bufs (fun bld ->
         loop bld (fun bld i ->
             let x = load_at bld i in
             let zero = B.f64 bld 0. in
             let (B.Cons (v, B.Nil)) =
               B.if_ bld (lt bld x zero)
                 ~then_:(fun bld ->
                   let y = load_at bld i in
                   B.Cons (B.f64_binary bld Expr.Value.Add y y, B.Nil))
                 ~else_:(fun bld -> B.Cons (load_at bld i, B.Nil))
             in
             store_at bld i v)));
  [%expect {| loads in both arms: branches 1 -> 0; identical: true |}]

let%expect_test "a branch that guards a failing load keeps it" =
  (* the load of cell 100 would fail, but only when the branch is taken *)
  report "failing load"
    (build ~buffers:bufs (fun bld ->
         loop bld (fun bld i ->
             let x = load_at bld i in
             let zero = B.f64 bld 0. in
             let (B.Cons (v, B.Nil)) =
               B.if_ bld (lt bld x zero)
                 ~then_:(fun _ -> B.Cons (zero, B.Nil))
                 ~else_:(fun bld -> B.Cons (load_at bld (idx bld 100), B.Nil))
             in
             store_at bld i v)));
  [%expect {| failing load: branches 1 -> 1; identical: true |}]

let%expect_test "a branch that stores or marks keeps it" =
  report "store in an arm"
    (build ~buffers:bufs (fun bld ->
         loop bld (fun bld i ->
             let x = load_at bld i in
             let zero = B.f64 bld 0. in
             let (B.Cons (v, B.Nil)) =
               B.if_ bld (lt bld x zero)
                 ~then_:(fun bld ->
                   store_at bld (idx bld 4) x;
                   B.Cons (zero, B.Nil))
                 ~else_:(fun _ -> B.Cons (x, B.Nil))
             in
             store_at bld i v)));
  report "mark in an arm"
    (build ~buffers:bufs (fun bld ->
         loop bld (fun bld i ->
             let x = load_at bld i in
             let zero = B.f64 bld 0. in
             let (B.Cons (v, B.Nil)) =
               B.if_ bld (lt bld x zero)
                 ~then_:(fun bld ->
                   B.mark bld Ssa_mark.Reduction;
                   B.Cons (zero, B.Nil))
                 ~else_:(fun _ -> B.Cons (x, B.Nil))
             in
             store_at bld i v)));
  [%expect
    {|
    store in an arm: branches 1 -> 1; identical: true
    mark in an arm: branches 1 -> 1; identical: true |}]
