(* The matrix products on the VME core (vme-core-mini-axi), as generated
   helper calls: under Fp32_ordered the matrix engine's function (Zvt
   binary32 tiles) against the vector unit's on the same core, bitwise
   against the binary32 oracle, with the cycles each takes; under
   Fp32_contracted the vector unit's, since the engine does not fuse. Then two
   engine-kernel defects the runs must catch. *)

module T = Machine_model_test.Model_test
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral
module Z = Coralnpu_kernels_zvt.Zvt_contraction

let config =
  {
    C.Config.dir = Sys.getenv "CORALNPU_DATA";
    configuration = "vme-core-mini-axi";
    deadline = 1200.;
    flags = [ "--cycles=500000000" ];
  }

let run ?engine ?engine_mutation ?(contractions = true) numerics oracle name g =
  let cycles = ref 0L in
  let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
  T.check
    ~route:(C.route ?engine ?engine_mutation ~on_run config)
    ~contractions
    ~pipeline:
      (Ssa_backends.Pipeline.Planned
         { numerics; target = Ssa_ir.Ssa_target.zve32f128 })
    ~oracle ~same:T.same_portable ~input:T.adversarial_input
    ~constant:(fun sg -> T.adversarial ~salt:3 sg)
    ~calls:1 name g;
  !cycles

let%expect_test "matrix products on the matrix engine and the vector unit" =
  let ordered = Ssa_ir.Ssa_numerics.Fp32_ordered in
  List.iter
    (fun (name, g) ->
      let v = run ~engine:false ordered T.Fp32 (name ^ ", vector unit") g in
      let z = run ordered T.Fp32 (name ^ ", matrix engine") g in
      let rate c =
        2. *. float_of_int (T.contraction_macs g) /. Int64.to_float c
      in
      Fmt.pr "  cycles %Ld -> %Ld (%.2fx); %.3f -> %.3f ops/cycle@." v z
        (Int64.to_float v /. Int64.to_float z)
        (rate v) (rate z);
      ignore
        (run Ssa_ir.Ssa_numerics.Fp32_contracted T.Fp32_contracted
           (name ^ ", contracted (vector unit)")
           g))
    T.contraction_fixtures;
  let tails = List.assoc "linear 7x29 k63 bias" T.contraction_fixtures in
  ignore
    (run ~engine_mutation:Z.Mutation.Full_tn ordered T.Fp32 "full tn" tails);
  ignore
    (run ~engine_mutation:Z.Mutation.Swapped_operands ordered T.Fp32
       "swapped operands" tails);
  [%expect
    {|
    linear 9x32 k64, vector unit (1 invocations): bitwise
    linear 9x32 k64, matrix engine (1 invocations): bitwise
      cycles 13694 -> 4870 (2.81x); 2.692 -> 7.570 ops/cycle
    linear 9x32 k64, contracted (vector unit) (1 invocations): bitwise
    linear 9x32 k64 bias, vector unit (1 invocations): bitwise
    linear 9x32 k64 bias, matrix engine (1 invocations): bitwise
      cycles 14020 -> 5198 (2.70x); 2.629 -> 7.092 ops/cycle
    linear 9x32 k64 bias, contracted (vector unit) (1 invocations): bitwise
    linear 7x29 k63 bias, vector unit (1 invocations): bitwise
    linear 7x29 k63 bias, matrix engine (1 invocations): bitwise
      cycles 12036 -> 6081 (1.98x); 2.125 -> 4.206 ops/cycle
    linear 7x29 k63 bias, contracted (vector unit) (1 invocations): bitwise
    bmm 1x9x32 k64, vector unit (1 invocations): bitwise
    bmm 1x9x32 k64, matrix engine (1 invocations): bitwise
      cycles 8288 -> 3497 (2.37x); 4.448 -> 10.542 ops/cycle
    bmm 1x9x32 k64, contracted (vector unit) (1 invocations): bitwise
    bmm 3x7x29 k63, vector unit (1 invocations): bitwise
    bmm 3x7x29 k63, matrix engine (1 invocations): bitwise
      cycles 35500 -> 16091 (2.21x); 2.162 -> 4.769 ops/cycle
    bmm 3x7x29 k63, contracted (vector unit) (1 invocations): bitwise
    full tn (1 invocations): DIFFERS
    swapped operands (1 invocations): DIFFERS |}]

(* 1x1 convolutions, the products the CNNs hold, under the contracted
   policy: their window sums round every product, so they are ordered
   products and go to the matrix engine; against the vector unit's function
   and the loops, bitwise all three. *)
let%expect_test "1x1 convolutions on the matrix engine" =
  let contracted = Ssa_ir.Ssa_numerics.Fp32_contracted in
  List.iter
    (fun (name, g) ->
      let l =
        run ~contractions:false contracted T.Fp32_contracted (name ^ ", loops")
          g
      in
      let v =
        run ~engine:false contracted T.Fp32_contracted (name ^ ", vector unit")
          g
      in
      let z = run contracted T.Fp32_contracted (name ^ ", matrix engine") g in
      let rate c =
        2. *. float_of_int (T.contraction_macs g) /. Int64.to_float c
      in
      Fmt.pr "  cycles %Ld / %Ld / %Ld; %.3f / %.3f / %.3f ops/cycle@." l v z
        (rate l) (rate v) (rate z))
    [
      ( "conv1x1 3x5 c19 to 29",
        T.pointwise_fixture ~pad:0 ~stride:1 ~h:3 ~w:5 ~cin:19 ~cout:29 );
      ( "conv1x1 7x7 c32 to 48",
        T.pointwise_fixture ~pad:0 ~stride:1 ~h:7 ~w:7 ~cin:32 ~cout:48 );
    ];
  [%expect
    {|
    conv1x1 3x5 c19 to 29, loops (1 invocations): bitwise
    conv1x1 3x5 c19 to 29, vector unit (1 invocations): bitwise
    conv1x1 3x5 c19 to 29, matrix engine (1 invocations): bitwise
      cycles 238307 / 11663 / 7262; 0.069 / 1.417 / 2.276 ops/cycle
    conv1x1 7x7 c32 to 48, loops (1 invocations): bitwise
    conv1x1 7x7 c32 to 48, vector unit (1 invocations): bitwise
    conv1x1 7x7 c32 to 48, matrix engine (1 invocations): bitwise
      cycles 426616 / 65550 / 26195; 0.353 / 2.296 / 5.746 ops/cycle |}]

(* bfloat16 products on the same core: the selection sends them to the
   vector unit (the engine's build multiplies no bfloat16), against the
   loops, bitwise both, under each binary32 policy. *)
let%expect_test "bfloat16 products on the vector unit" =
  let run ?(contractions = true) numerics oracle name g =
    let cycles = ref 0L in
    let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
    T.check ~route:(C.route ~on_run config) ~contractions
      ~pipeline:
        (Ssa_backends.Pipeline.Planned
           { numerics; target = Ssa_ir.Ssa_target.zve32f128 })
      ~oracle ~same:T.same_portable
      ~input:(fun ~salt id sg ->
        T.in_format sg (T.adversarial_input ~salt id sg))
      ~constant:(fun sg -> T.in_format sg (T.adversarial ~salt:3 sg))
      ~calls:1 name g;
    !cycles
  in
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (pname, numerics, oracle) ->
          let l =
            run ~contractions:false numerics oracle
              (Fmt.str "%s, %s, loops" name pname)
              g
          in
          let v =
            run numerics oracle (Fmt.str "%s, %s, vector unit" name pname) g
          in
          Fmt.pr "  cycles %Ld -> %Ld (%.2fx)@." l v
            (Int64.to_float l /. Int64.to_float v))
        [
          ("ordered", Ssa_ir.Ssa_numerics.Fp32_ordered, T.Fp32);
          ("contracted", Ssa_ir.Ssa_numerics.Fp32_contracted, T.Fp32_contracted);
        ])
    (List.filter
       (fun (n, _) -> n <> "linear bf16 7x29 k63 bias")
       T.bf16_fixtures);
  [%expect
    {|
    bmm bf16 3x7x29 k63, ordered, loops (1 invocations): bitwise
    bmm bf16 3x7x29 k63, ordered, vector unit (1 invocations): bitwise
      cycles 321662 -> 25971 (12.39x)
    bmm bf16 3x7x29 k63, contracted, loops (1 invocations): bitwise
    bmm bf16 3x7x29 k63, contracted, vector unit (1 invocations): bitwise
      cycles 272711 -> 18354 (14.86x)
    linear bf16 9x32 k64, ordered, loops (1 invocations): bitwise
    linear bf16 9x32 k64, ordered, vector unit (1 invocations): bitwise
      cycles 188190 -> 15466 (12.17x)
    linear bf16 9x32 k64, contracted, loops (1 invocations): bitwise
    linear bf16 9x32 k64, contracted, vector unit (1 invocations): bitwise
      cycles 184078 -> 11130 (16.54x) |}]

(* Prepacked weights on the vector unit (the contracted policy, so the RVV
   function): B's rows contiguous instead of strided, against the unpacked
   function, bitwise both. *)
let%expect_test "prepacked weights on the vector unit" =
  let run ~prepack name g =
    let cycles = ref 0L in
    let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
    T.check ~route:(C.route ~on_run config) ~contractions:true ~prepack
      ~pipeline:
        (Ssa_backends.Pipeline.Planned
           {
             numerics = Ssa_ir.Ssa_numerics.Fp32_contracted;
             target = Ssa_ir.Ssa_target.zve32f128;
           })
      ~oracle:T.Fp32_contracted ~same:T.same_portable
      ~input:(fun ~salt id sg ->
        T.in_format sg (T.adversarial_input ~salt id sg))
      ~constant:(fun sg -> T.in_format sg (T.adversarial ~salt:3 sg))
      ~calls:1 name g;
    !cycles
  in
  List.iter
    (fun (name, g) ->
      let s = run ~prepack:false (name ^ ", strided") g in
      let p = run ~prepack:true (name ^ ", packed") g in
      Fmt.pr "  cycles %Ld -> %Ld (%.2fx)@." s p
        (Int64.to_float s /. Int64.to_float p))
    (List.filter
       (fun (n, _) ->
         List.mem n
           [ "linear 9x32 k64"; "linear 7x29 k63 bias"; "linear bf16 9x32 k64" ])
       (T.contraction_fixtures @ T.bf16_fixtures));
  [%expect
    {|
    linear 9x32 k64, strided (1 invocations): bitwise
    linear 9x32 k64, packed (1 invocations, 1 packed): bitwise
      cycles 9280 -> 4687 (1.98x)
    linear 7x29 k63 bias, strided (1 invocations): bitwise
    linear 7x29 k63 bias, packed (1 invocations, 1 packed): bitwise
      cycles 10034 -> 6214 (1.61x)
    linear bf16 9x32 k64, strided (1 invocations): bitwise
    linear bf16 9x32 k64, packed (1 invocations, 1 packed): bitwise
      cycles 11130 -> 6351 (1.75x) |}]
