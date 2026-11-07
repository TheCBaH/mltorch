module T = Machine_model_test.Model_test
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral

let subsystem =
  {
    C.Config.dir = Sys.getenv "CORALNPU_SUBSYSTEM_DATA";
    configuration = "subsystem-vector-normal";
    deadline = 1200.;
    flags = [];
  }

let run ?dma name g =
  let cycles = ref 0L in
  let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
  T.check
    ~route:(C.route ?dma ~on_run subsystem)
    ~contractions:true
    ~pipeline:
      (Ssa_backends.Pipeline.Planned
         {
           numerics = Ssa_ir.Ssa_numerics.Fp32_contracted;
           target = Ssa_ir.Ssa_target.zve32f128;
         })
    ~oracle:T.Fp32_contracted ~constant:T.positive ~calls:1 name g;
  !cycles

let%expect_test "operands in DTCM, in place and with DMA allowed" =
  List.iter
    (fun (name, g) ->
      let direct = run (name ^ ", DTCM") g in
      let allowed = run ~dma:C.Dma.Allowed (name ^ ", DTCM, allowed") g in
      Fmt.pr "  cycles %Ld -> %Ld (%.2fx)@." direct allowed
        (Int64.to_float direct /. Int64.to_float allowed))
    (List.filter
       (fun (n, _) ->
         List.mem n
           [ "linear 9x32 k64"; "linear 7x29 k63 bias"; "bmm 1x9x32 k64" ])
       T.contraction_fixtures);
  [%expect {||}]
