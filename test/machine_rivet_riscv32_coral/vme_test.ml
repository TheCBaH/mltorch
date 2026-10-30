(* The matrix products on the VME core (vme-core-mini-axi), as generated
   helper calls: under Fp32_ordered the matrix engine's function (Zvt
   binary32 tiles) against the vector unit's on the same core, bitwise
   against the binary32 oracle, with the cycles each takes; under
   Fp32_contracted the vector unit's, since the engine does not fuse. Then two
   engine-kernel defects the runs must catch. *)

module T = Machine_model_test.Model_test
module B = Machine_model_test.Attention_binding_test
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

(* The generated [Sdpa] does not fit the core: its one large kernel alone
   overflows the 8 KiB instruction memory, so the bfloat16 fixtures are
   refused at placement. *)
let%expect_test "bfloat16 Sdpa is refused at the instruction memory" =
  List.iter
    (fun (name, g) ->
      T.check ~route:(C.route config) ~contractions:true
        ~pipeline:
          (Ssa_backends.Pipeline.Planned
             {
               numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
               target = Ssa_ir.Ssa_target.zve32f128;
             })
        ~oracle:T.Fp32 ~same:T.same_portable
        ~input:(fun ~salt id sg ->
          T.in_format sg (T.adversarial_input ~salt id sg))
        ~calls:1 name g)
    T.sdpa_bf16_fixtures;
  [%expect
    {|
    sdpa bf16 32x32 d32: refused: invocation 0 (n0): itcm: 8300 bytes needed, 8192 available
    sdpa bf16 64x64 d64: refused: invocation 0 (n0): itcm: 8300 bytes needed, 8192 available
    sdpa bf16 9x29 d64 masked: refused: invocation 0 (n0): itcm: 8448 bytes needed, 8192 available |}]

(* The per-op composition of an [Sdpa] on the core: the scores product
   granted the engine's pair rounding, then the scale, a softmax and a
   binary32 context product on the vector unit, bitwise against the
   descriptor's [Paired] scores followed by the binary32 oracle of the rest.
   The cycles are the bar the fused kernel must beat. *)
let%expect_test "the per-op composition of Sdpa" =
  let module K = Contraction in
  let flat (Tensor.Tensor t as p) =
    let a = Array.make (Vec6.numel t.Tensor.shape :> int) 0. in
    Vec6.iter t.Tensor.shape (fun c ->
        a.((Vec6.offset t.Tensor.shape c :> int)) <- Tensor.read p c);
    a
  in
  let run ~tq ~tk ~d =
    let name = Fmt.str "sdpa composition %dx%d d%d" tq tk d in
    let tail = T.sdpa_composition_tail_fixture ~tq ~tk ~d in
    let oracle =
      T.Custom
        (fun (g : Graph_ir.graph) _constants inputs ->
          let q, kt, v =
            match inputs with
            | [ (_, q); (_, kt); (_, v) ] -> (flat q, flat kt, v)
            | _ -> assert false
          in
          let desc =
            Result.get_ok
              (Err.payload
                 (K.make ~batch:1L ~m:(Int64.of_int tq) ~n:(Int64.of_int tk)
                    ~k:(Int64.of_int d)
                    ~a:{ K.Operand.batch = 0L; outer = Int64.of_int d; k = 1L }
                    ~b:{ K.Operand.batch = 0L; outer = 1L; k = Int64.of_int tk }
                    ~c:{ K.Output.batch = 0L; row = Int64.of_int tk; col = 1L }
                    ~format:K.Element.Bf16 ~seed:K.Seed.Zero
                    ~numerics:K.Numerics.Paired))
          in
          let s = Array.make (tq * tk) 0. in
          K.eval desc
            ~a:(fun o -> q.(Int64.to_int o))
            ~b:(fun o -> kt.(Int64.to_int o))
            ~c:(fun _ -> 0.)
            ~write:(fun o x -> s.(Int64.to_int o) <- x);
          let tg = tail () in
          let tb = T.bundle tg in
          let s_id, v_id =
            match tb.Loop_ir.Loop_bundle.inputs with
            | [ a; b ] -> (a, b)
            | _ -> assert false
          in
          let s_shape = (T.sig_of tg s_id).Tensor_sig.shape in
          let s_t =
            Tensor.materialize s_shape (fun c ->
                s.((Vec6.offset s_shape c :> int)))
          in
          let out =
            T.expected T.Fp32 tg tb ~constants:[]
              ~inputs:[ (s_id, s_t); (v_id, v) ]
          in
          let tid = List.hd tg.Graph_ir.Graph.outputs
          and gid = List.hd g.Graph_ir.Graph.outputs in
          Tensor_id.Map.singleton gid (Tensor_id.Map.find tid out))
    in
    let cycles = ref [] in
    let on_run (o : C.Run.t) = cycles := o.C.Run.mcycle :: !cycles in
    T.check ~route:(C.route ~on_run config) ~contractions:true ~bf16_pairs:true
      ~pipeline:
        (Ssa_backends.Pipeline.Planned
           {
             numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
             target = Ssa_ir.Ssa_target.zve32f128;
           })
      ~oracle ~same:T.same_portable
      ~input:(fun ~salt id sg -> T.in_format sg (T.values ~salt:(salt + id) sg))
      ~calls:1 name
      (T.sdpa_composition_bf16_fixture ~tq ~tk ~d);
    Fmt.pr "  cycles %Ld = %a@."
      (List.fold_left Int64.add 0L !cycles)
      Fmt.(list ~sep:(any " + ") int64)
      (List.rev !cycles)
  in
  run ~tq:16 ~tk:16 ~d:16;
  run ~tq:32 ~tk:32 ~d:32;
  run ~tq:64 ~tk:64 ~d:64;
  [%expect
    {|
    sdpa composition 16x16 d16 (5 invocations): bitwise
      cycles 266450 = 1414 + 915 + 261626 + 1130 + 1365
    sdpa composition 32x32 d32 (5 invocations): bitwise
      cycles 1055074 = 4202 + 2740 + 1038604 + 3498 + 6030
    sdpa composition 64x64 d64 (5 invocations): bitwise
      cycles 4211209 = 30617 + 13333 + 4092795 + 16266 + 58198 |}]

(* What the generated binary32 [exp] and softmax cost on the core, planned for
   the scalar unit or forced onto the vector one: cycles for the whole image
   and per element. The softmax kernel of the fused attention is built from
   the same [exp]. *)
let%expect_test "generated exp and softmax cycles per element" =
  let fixture name f ~h ~w =
    ( name,
      fun () ->
        Native_test.Graph_fixtures.build name
          Graph_builder.(
            let* x =
              input ~shape:(Native_test.Graph_fixtures.s 1 1 1 1 h w) ()
            in
            f x) )
  in
  let fixtures ~h ~w =
    [
      fixture "exp" ~h ~w (fun x -> Graph_builder.exp x);
      fixture "softmax over C" ~h ~w (fun x ->
          Graph_builder.softmax { Reduce.Softmax.axis = Axis.C } x);
      fixture "softmax over W" ~h ~w (fun x ->
          Graph_builder.softmax { Reduce.Softmax.axis = Axis.W } x);
    ]
  in
  List.iter
    (fun (target_name, target) ->
      List.iter
        (fun (h, w) ->
          List.iter
            (fun (name, g) ->
              let cycles = ref [] in
              let on_run (o : C.Run.t) = cycles := o.C.Run.mcycle :: !cycles in
              let label = Fmt.str "%s %dx%d, %s" name h w target_name in
              T.check ~route:(C.route ~on_run config) ~contractions:true
                ~pipeline:
                  (Ssa_backends.Pipeline.Planned
                     { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target })
                ~oracle:T.Fp32 ~same:T.same_portable ~calls:1 label g;
              let total = List.fold_left Int64.add 0L !cycles in
              Fmt.pr "  cycles %Ld, %.1f per element@." total
                (Int64.to_float total /. float_of_int (h * w)))
            (fixtures ~h ~w))
        [ (16, 16); (64, 64) ])
    [
      ("planned", Ssa_ir.Ssa_target.zve32f128);
      ("forced vector", Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.zve32f128);
    ];
  [%expect
    {|
    exp 16x16, planned (1 invocations): bitwise
      cycles 38028, 148.5 per element
    softmax over C 16x16, planned (1 invocations): bitwise
      cycles 262426, 1025.1 per element
    softmax over W 16x16, planned (1 invocations): bitwise
      cycles 262426, 1025.1 per element
    exp 64x64, planned (1 invocations): bitwise
      cycles 607246, 148.3 per element
    softmax over C 64x64, planned (1 invocations): bitwise
      cycles 4190117, 1023.0 per element
    softmax over W 64x64, planned (1 invocations): bitwise
      cycles 4190117, 1023.0 per element
    exp 16x16, forced vector (1 invocations): bitwise
      cycles 38028, 148.5 per element
    softmax over C 16x16, forced vector (1 invocations): bitwise
      cycles 262426, 1025.1 per element
    softmax over W 16x16, forced vector (1 invocations): bitwise
      cycles 262426, 1025.1 per element
    exp 64x64, forced vector (1 invocations): bitwise
      cycles 607246, 148.3 per element
    softmax over C 64x64, forced vector (1 invocations): bitwise
      cycles 4190117, 1023.0 per element
    softmax over W 64x64, forced vector (1 invocations): bitwise
      cycles 4190117, 1023.0 per element |}]

(* A granted [Sdpa] on the VME core: the node is one call of the fused
   function, bitwise against the granted streaming reference, with the cycles
   of the whole run (the generated path is refused at the instruction memory);
   an f32 node and a node whose operands are not bfloat16 stay on the generated
   path, with the reason. *)
let%expect_test "a granted attention runs as the fused function" =
  let module S = Machine_model.Sdpa_select in
  let grant = Some S.Grant.default in
  let cycles = ref 0L in
  let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
  List.iter
    (fun (name, g) ->
      let graph = g () in
      let node = B.sdpa_node graph in
      match S.choose ~grant B.engine graph node with
      | Error r -> Fmt.pr "%s: not selected: %a@." name S.Reason.pp r
      | Ok site ->
          cycles := 0L;
          T.check ~route:(C.route ~on_run config)
            ~attention:(fun id ->
              if Graph_ir.Node_id.equal id node then Some site else None)
            ~pipeline:
              (Ssa_backends.Pipeline.Planned
                 {
                   numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
                   target = Ssa_ir.Ssa_target.zve32f128;
                 })
            ~oracle:(B.attention_oracle site) ~same:T.same_portable ~calls:1
            ~input:(fun ~salt i sg ->
              T.in_format sg
                (if i = 3 then
                   Tensor.materialize sg.Tensor_sig.shape (fun c ->
                       let n = (Vec6.offset sg.Tensor_sig.shape c :> int) in
                       if (n + salt) mod 7 = 0 then neg_infinity
                       else float_of_int (n mod 5) /. 4.)
                 else T.values ~salt:(salt + (3 * i)) sg))
            name g;
          Fmt.pr "  cycles %Ld@." !cycles)
    T.sdpa_bf16_fixtures;
  (* an f32 attention is not selected and stays refused at the instruction
     memory; a granted flag with a non-bfloat16 operand falls back *)
  let f32 () =
    T.F.build "sdpa_f32"
      Graph_builder.(
        let* q = input ~shape:(T.F.s 1 1 1 1 32 32) () in
        let* k = input ~shape:(T.F.s 1 1 1 1 32 32) () in
        let* v = input ~shape:(T.F.s 1 1 1 1 32 32) () in
        sdpa
          { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
          ~query:q ~key:k ~value:v ())
  in
  let graph = f32 () in
  (match S.choose ~grant B.engine graph (B.sdpa_node graph) with
  | Error r -> Fmt.pr "f32 32x32: not selected: %a@." S.Reason.pp r
  | Ok _ -> Fmt.pr "f32 32x32: selected@.");
  T.check ~route:(C.route config)
    ~pipeline:
      (Ssa_backends.Pipeline.Planned
         {
           numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
           target = Ssa_ir.Ssa_target.zve32f128;
         })
    ~oracle:T.Fp32 ~same:T.same_portable ~calls:1 "f32 32x32, generated" f32;
  [%expect
    {|
    sdpa bf16 32x32 d32 (1 invocations): bitwise
      cycles 87669
    sdpa bf16 64x64 d64 (1 invocations): bitwise
      cycles 386501
    sdpa bf16 9x29 d64 masked (1 invocations): bitwise
      cycles 45247
    f32 32x32: not selected: the query is not bfloat16
    f32 32x32, generated: refused: invocation 0 (n0): itcm: 8244 bytes needed, 8192 available |}]
