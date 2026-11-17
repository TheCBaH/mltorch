(* The band driver's chains as RV32 processes under qemu: the banded program
   is lowered through the contraction helper route, published, and run with its
   contractions as generated RVV functions; its output bits are compared with the
   whole-tensor chain on the SSA interpreter. *)

module Src = Machine_source_test.Mir_source
module C = Band_chains

let qemu ?kernel_mutation (p : Ssa_ir.Ssa_program.t) ~inputs =
  match
    Src.case_of_program p
      ~inputs:(List.map (fun (i, c) -> (i, Ssa_ir.Ssa_memory.Floats c)) inputs)
      ~precision:Machine_ir.Mir_planning.Precision.F32 ()
  with
  | Error e -> Error e
  | Ok case -> Result.bind (Kernel_test.qemu ?kernel_mutation case) C.observed

(* Every chain is equal. A window chain adds into its output (seed Tensor): the
   RVV contraction starts each row from C's cells. *)
let%expect_test "banded chains under qemu equal the whole chain" =
  List.iter
    (fun i ->
      Fmt.pr "%s@."
        (C.verdict ~run:(qemu ?kernel_mutation:None) (List.nth C.chains i)))
    [ 0; 1; 4; 5 ];
  [%expect
    {|
    pointwise 3 stages, 8 rows, unrolled: equal
    pointwise 3 stages, 16 rows, looped: equal
    pointwise then 3x3, 10 rows, looped: equal
    3x3, pointwise, 3x3, 20 rows, looped: equal |}]

let%expect_test "a driver mutation is red under qemu" =
  Fmt.pr "%s@."
    (C.verdict ~driver:Band_driver.Step_dropped
       ~run:(qemu ?kernel_mutation:None)
       (List.nth C.chains 1));
  [%expect {| pointwise 3 stages, 16 rows, looped: DIFFERS |}]

let%expect_test "a seed mutation is red under qemu" =
  Fmt.pr "%s@."
    (C.verdict
       ~run:
         (qemu
            ~kernel_mutation:
              Coralnpu_kernels.Rvv_contraction.Mutation.Seed_zeroed)
       (List.nth C.chains 4));
  [%expect {| pointwise then 3x3, 10 rows, looped: DIFFERS |}]
