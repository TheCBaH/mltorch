(* fastvit_sa12's attention layer shape as one granted bfloat16 node: 49
   tokens of 512 features, 16 heads of 32, no norm inside the layer, no
   query, key or value bias, an output bias and a layer scale. Its 2 MiB of
   bfloat16 weights and every other tensor live in RAM; bitwise against
   [Attention_layer_block.eval] at [Bf16_paired], with the cycles of the run.
   The suites beside this one run the same code at the small shapes in
   minutes. *)

module T = Machine_model_test.Model_test
module AT = Machine_model_test.Attention_layer_test
module LF = Native_test.Attention_layer_fixtures
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral
module S = Machine_model.Sdpa_select
module Layer_select = Machine_model.Layer_select
module A = Attention_block

let engine = { Contraction.Select.Capabilities.matrix = true; vector = true }

let grant =
  Some { S.Grant.block_keys = 16; precision = A.Precision.Bf16_paired }

let fastvit =
  {
    (LF.all_flags LF.small) with
    LF.Config.tokens = 49;
    width = 512;
    heads = 16;
    batch = 1;
    bf16_weights = true;
    q_bias = false;
    k_bias = false;
    v_bias = false;
  }

let home (_ : Machine_rivet_common.Rivet_table.slot) = Coral_layout.Home.Ram

let run ~dir ~configuration name =
  let g = LF.node fastvit in
  let node = AT.layer_node g in
  match Layer_select.choose ~grant engine g node with
  | Error r -> Fmt.pr "%s: not selected: %a@." name Layer_select.Reason.pp r
  | Ok site ->
      let cycles = ref 0L in
      let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
      let values = LF.inputs g in
      T.check
        ~route:
          (C.route ~budget:131072L (* the 49 x 512 binary32 output *)
             ~home ~on_run
             {
               C.Config.dir = Sys.getenv dir;
               configuration;
               deadline = 14400.;
               flags = [ "--cycles=500000000" ];
             })
        ~layer:(fun id ->
          if Graph_ir.Node_id.equal id node then Some site else None)
        ~pipeline:
          (Ssa_backends.Pipeline.Planned
             {
               numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
               target = Ssa_ir.Ssa_target.zve32f128;
             })
        ~oracle:(AT.layer_oracle site) ~same:T.same_portable ~calls:1
        ~input:(fun ~salt:_ _ (sg : Tensor_sig.t) ->
          T.in_format sg (List.assoc sg.Tensor_sig.id values))
        name
        (fun () -> g);
      Fmt.pr "  cycles %Ld@." !cycles

(* resident: the core has no DMA *)
let%expect_test "fastvit's shape on the VME core" =
  run ~dir:"CORALNPU_DATA" ~configuration:"vme-core-mini-axi"
    "fastvit 49 x 512, 16 x 32";
  [%expect
    {|
    fastvit 49 x 512, 16 x 32 (1 invocations): bitwise
      cycles 40109063 |}]

(* the attention function and the kernel's other regions as overlays *)
let%expect_test "fastvit's shape on the subsystem" =
  run ~dir:"CORALNPU_SUBSYSTEM_DATA" ~configuration:"subsystem-matrix-normal"
    "fastvit 49 x 512, 16 x 32";
  [%expect
    {|
    fastvit 49 x 512, 16 x 32 (1 invocations): bitwise
      cycles 47930016 |}]
