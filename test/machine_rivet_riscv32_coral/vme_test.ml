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

(* bfloat16 products granted the engine's pair rounding ([bf16_pairs]): the
   matrix engine's function against the descriptor's [Paired] reference over
   the model's own operands, which the Loop interpreter has no accumulation
   for. *)
let%expect_test "bfloat16 pairs from a model on the matrix engine" =
  let module K = Contraction in
  let flat (Tensor.Tensor t as p) =
    let a = Array.make (Vec6.numel t.Tensor.shape :> int) 0. in
    Vec6.iter t.Tensor.shape (fun c ->
        a.((Vec6.offset t.Tensor.shape c :> int)) <- Tensor.read p c);
    a
  in
  let paired ~batch ~m ~n ~k ~b_outer ~b_k ~b_batch ~a_of ~b_of =
    T.Custom
      (fun (g : Graph_ir.graph) constants inputs ->
        let d =
          Result.get_ok
            (Err.payload
               (K.make ~batch ~m ~n ~k
                  ~a:{ K.Operand.batch = Int64.mul m k; outer = k; k = 1L }
                  ~b:{ K.Operand.batch = b_batch; outer = b_outer; k = b_k }
                  ~c:{ K.Output.batch = Int64.mul m n; row = n; col = 1L }
                  ~format:K.Element.Bf16 ~seed:K.Seed.Zero
                  ~numerics:K.Numerics.Paired))
        in
        let a = flat (a_of constants inputs)
        and b = flat (b_of constants inputs) in
        let out =
          Array.make (Int64.to_int (Int64.mul batch (Int64.mul m n))) 0.
        in
        K.eval d
          ~a:(fun o -> a.(Int64.to_int o))
          ~b:(fun o -> b.(Int64.to_int o))
          ~c:(fun _ -> 0.)
          ~write:(fun o x -> out.(Int64.to_int o) <- x);
        let id = List.hd g.Graph_ir.Graph.outputs in
        let shape = (T.sig_of g id).Tensor_sig.shape in
        Tensor_id.Map.singleton id
          (Tensor.materialize shape (fun c ->
               out.((Vec6.offset shape c :> int)))))
  in
  let nth l i = snd (List.nth l i) in
  let run ?(pairs = true) name g oracle =
    let cycles = ref 0L in
    let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
    T.check ~route:(C.route ~on_run config) ~contractions:true ~bf16_pairs:pairs
      ~pipeline:
        (Ssa_backends.Pipeline.Planned
           {
             numerics = Ssa_ir.Ssa_numerics.Fp32_contracted;
             target = Ssa_ir.Ssa_target.zve32f128;
           })
      ~oracle ~same:T.same_portable
      ~input:(fun ~salt id sg ->
        T.in_format sg (T.adversarial_input ~salt id sg))
      ~constant:(fun sg -> T.in_format sg (T.adversarial ~salt:3 sg))
      ~calls:1 name g;
    Fmt.pr "  cycles %Ld@." !cycles
  in
  run "linear bf16 9x32 k64, paired"
    (T.linear_bf16_fixture ~m:9 ~n:32 ~k:64 ~bias:false)
    (paired ~batch:1L ~m:9L ~n:32L ~k:64L ~b_outer:64L ~b_k:1L ~b_batch:0L
       ~a_of:(fun _ i -> nth i 0)
       ~b_of:(fun c _ -> nth c 0));
  run "linear bf16 64x64 k64, paired"
    (T.linear_bf16_fixture ~m:64 ~n:64 ~k:64 ~bias:false)
    (paired ~batch:1L ~m:64L ~n:64L ~k:64L ~b_outer:64L ~b_k:1L ~b_batch:0L
       ~a_of:(fun _ i -> nth i 0)
       ~b_of:(fun c _ -> nth c 0));
  run "bmm bf16 1x64x32 k64, paired"
    (T.bmm_bf16_fixture ~batch:1 ~m:64 ~n:32 ~k:64)
    (paired ~batch:1L ~m:64L ~n:32L ~k:64L ~b_outer:1L ~b_k:32L
       ~b_batch:(Int64.of_int (64 * 32))
       ~a_of:(fun _ i -> nth i 0)
       ~b_of:(fun _ i -> nth i 1));
  run "bmm bf16 3x7x29 k63, paired"
    (T.bmm_bf16_fixture ~batch:3 ~m:7 ~n:29 ~k:63)
    (paired ~batch:3L ~m:7L ~n:29L ~k:63L ~b_outer:1L ~b_k:29L
       ~b_batch:(Int64.of_int (63 * 29))
       ~a_of:(fun _ i -> nth i 0)
       ~b_of:(fun _ i -> nth i 1));
  (* the pair rounding is the policy's grant: without it the same product runs
     rounded step by step on the vector unit, which the reference refuses *)
  run ~pairs:false "linear bf16 9x32 k64, not granted"
    (T.linear_bf16_fixture ~m:9 ~n:32 ~k:64 ~bias:false)
    (paired ~batch:1L ~m:9L ~n:32L ~k:64L ~b_outer:64L ~b_k:1L ~b_batch:0L
       ~a_of:(fun _ i -> nth i 0)
       ~b_of:(fun c _ -> nth c 0));
  [%expect
    {|
    linear bf16 9x32 k64, paired (1 invocations): bitwise
      cycles 4241
    linear bf16 64x64 k64, paired (1 invocations): bitwise
      cycles 37789
    bmm bf16 1x64x32 k64, paired (1 invocations): bitwise
      cycles 16141
    bmm bf16 3x7x29 k63, paired (1 invocations): bitwise
      cycles 10418
    linear bf16 9x32 k64, not granted (1 invocations): DIFFERS
      cycles 11130 |}]
