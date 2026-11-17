(* The band driver's chains as RV32 processes under qemu: the banded program
   is lowered through the contraction helper route, published, and run with its
   contractions as generated RVV functions; its output bits are compared with the
   whole-tensor chain on the SSA interpreter. *)

module Src = Machine_source_test.Mir_source
module C = Band_chains

let qemu ?mutation (p : Ssa_ir.Ssa_program.t) ~inputs =
  match
    Src.case_of_program p
      ~inputs:(List.map (fun (i, c) -> (i, Ssa_ir.Ssa_memory.Floats c)) inputs)
      ~precision:Machine_ir.Mir_planning.Precision.F32 ()
  with
  | Error e -> Error e
  | Ok case -> Result.bind (Kernel_test.qemu ?mutation case) C.observed

(* The pointwise chains are equal. A window chain adds into its output (seed
   Tensor), which the RVV contraction does not realize: it is refused, not run. *)
let%expect_test "banded chains under qemu equal the whole chain" =
  List.iter
    (fun i ->
      Fmt.pr "%s@." (C.verdict ~run:(qemu ?mutation:None) (List.nth C.chains i)))
    [ 0; 1; 4; 5 ];
  [%expect
    {|
    pointwise 3 stages, 8 rows, unrolled: equal
    pointwise 3 stages, 16 rows, looped: equal
    pointwise then 3x3, 10 rows, looped: contraction contract.f32.b1.m3.n2.k9.a0_3_1.b0_1_2.c0_2_1.tensor.ordered: a contraction seeded from its output
    3x3, pointwise, 3x3, 20 rows, looped: contraction contract.f32.b1.m4.n3.k6.a0_2_1.b0_1_3.c0_3_1.tensor.ordered: a contraction seeded from its output |}]

let%expect_test "a driver mutation is red under qemu" =
  Fmt.pr "%s@."
    (C.verdict ~driver:Band_driver.Step_dropped ~run:(qemu ?mutation:None)
       (List.nth C.chains 1));
  [%expect {| pointwise 3 stages, 16 rows, looped: DIFFERS |}]
