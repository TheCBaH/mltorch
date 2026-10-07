module M = Machine_source_test.Mir_math_test

(* The math programs through AArch64 selected and allocated, and x86-64
   selected and realized: each libm function is a call through the target's
   convention to its helper's model, and erf's expansion keeps its order. *)

let%expect_test "math helpers on both targets" =
  let xs = Array.append M.inputs (Array.sub M.samples 0 12) in
  List.iter
    (fun (u, f32) ->
      let p = M.program u xs in
      let p = if f32 then Ssa_ir.Ssa_precision.to_f32 p else p in
      let precision =
        if f32 then Machine_ir.Mir_planning.Precision.F32
        else Machine_ir.Mir_planning.Precision.F64
      in
      let inputs = [ (0, Ssa_ir.Ssa_memory.Floats xs) ] in
      Fmt.pr "%s%s: aarch64 %s, allocated %s | x86_64 %s, realized %s@."
        (Ssa_ir.Ssa_op.unary_name u)
        (if f32 then " (binary32)" else "")
        (Machine_aarch64_test.A64_harness.program ~precision p ~inputs)
        (Machine_alloc_test.Alloc_harness.program ~precision p ~inputs)
        (X64_harness.program ~precision p ~inputs)
        (X64_harness.program ~precision ~stage:X64_harness.Realized p ~inputs))
    Expr.Value.
      [
        (Cos, false);
        (Erf, false);
        (Erf, true);
        (Exp, false);
        (Exp, true);
        (Log, false);
        (Sin, false);
      ];
  [%expect
    {|
    cos: aarch64 ok, allocated ok | x86_64 ok, realized ok
    erf: aarch64 ok, allocated ok | x86_64 ok, realized ok
    erf (binary32): aarch64 ok, allocated ok | x86_64 ok, realized ok
    exp: aarch64 ok, allocated ok | x86_64 ok, realized ok
    exp (binary32): aarch64 ok, allocated ok | x86_64 ok, realized ok
    log: aarch64 ok, allocated ok | x86_64 ok, realized ok
    sin: aarch64 ok, allocated ok | x86_64 ok, realized ok |}]
