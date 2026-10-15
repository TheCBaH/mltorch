(* The Fp32_ordered policy as x86-64 processes: every kernel binary32 and
   scalar (no vector target), selected as SSE scalar single-precision forms,
   run under qemu-user (or on an x86-64 CPU) and compared with the physical
   interpreter over the same bound bytes. The transcendentals are the project's
   own binary32 sequences, so no helper, no call and no double appears: the
   instruction bytes and control flow are what qemu checks. *)

open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
module H = Native_harness

let show kernel ~bind =
  Fmt.pr "%s@."
    (H.planned ~target:Ssa_ir.Ssa_target.scalar
       ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered kernel ~bind)

let spread n =
  Array.init n (fun i ->
      match i with
      | 0 -> Float.nan
      | 1 -> -0.
      | 2 -> 3.4e38
      | 3 -> Float.infinity
      | 4 -> 1e-40
      | _ -> (float_of_int (i * 7919 mod 997) /. 31.) -. 16.)

let x = Loop_fixtures.load_t0

let%expect_test "binary32 arithmetic and the owned transcendentals" =
  List.iter
    (fun (name, body) ->
      Fmt.pr "%s: " name;
      let n = 37 in
      let shape = Loop_fixtures.shape_w n in
      show
        (Loop_fixtures.pixel_kernel ~shape body)
        ~bind:(bind_data ~shape (spread n)))
    [
      ("x * x + x", Expr.Value.(add (mul x x) x));
      ("x / 3 - 1.5", Expr.Value.(sub (div x (const 3.)) (const 1.5)));
      ("sqrt x", Expr.Value.sqrt x);
      ("exp x", Expr.Value.exp x);
      ("log x", Expr.Value.log x);
      ("sin x", Expr.Value.sin x);
      ("cos x", Expr.Value.cos x);
      ("erf x", Expr.Value.erf x);
    ];
  [%expect
    {|
    x * x + x: ok [process: agree]
    x / 3 - 1.5: ok [process: agree]
    sqrt x: ok [process: agree]
    exp x: ok [process: agree]
    log x: ok [process: agree]
    sin x: ok [process: agree]
    cos x: ok [process: agree]
    erf x: ok [process: agree] |}]
