(* A granted bfloat16 attention layer on the VME core: selected, lowered to
   RV32 code and run, against [Attention_layer_block.eval] bit for bit. The
   core has no DMA, so nothing can be loaded on demand and the whole image
   must be resident in the 8 KiB ITCM, which every one of these fits since the
   runtime walks its slots from tables. The subsystem suite runs them with
   their code as overlays. *)

module T = Machine_model_test.Model_test
module AT = Machine_model_test.Attention_layer_test
module LF = Native_test.Attention_layer_fixtures
module L = Attention_layer.Attention_layer
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral
module S = Machine_model.Sdpa_select
module Layer_select = Machine_model.Layer_select
module A = Attention_block

let config =
  {
    C.Config.dir = Sys.getenv "CORALNPU_DATA";
    configuration = "vme-core-mini-axi";
    deadline = Deadline.seconds;
    flags = [ "--cycles=500000000" ];
  }

let engine = { Contraction.Select.Capabilities.matrix = true; vector = true }

let grant =
  Some { S.Grant.block_keys = 16; precision = A.Precision.Bf16_paired }

(* L 16, E 64, two heads of 32: the attention function's own shape limits *)
let base =
  {
    (LF.all_flags LF.small) with
    LF.Config.tokens = 16;
    width = 64;
    heads = 2;
    batch = 1;
    bf16_weights = true;
  }

let run name c =
  let g = LF.node c in
  let node = AT.layer_node g in
  match Layer_select.choose ~grant engine g node with
  | Error r -> Fmt.pr "%s: not selected: %a@." name Layer_select.Reason.pp r
  | Ok site ->
      let cycles = ref 0L in
      let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
      let values = LF.inputs g in
      T.check
        ~route:(C.route ~on_run ~label:name config)
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

let%expect_test "a granted bfloat16 layer on the VME core, resident" =
  run "bare: no norm, bias or layer scale"
    {
      base with
      LF.Config.q_bias = false;
      k_bias = false;
      v_bias = false;
      o_bias = false;
      layer_scale = false;
    };
  run "no norm, all flags" base;
  run "pre layer, all flags, rope"
    (LF.with_norm L.Norm.Kind.Layer L.Placement.Pre
       { base with LF.Config.rope = LF.Rope_form.Shared });
  run "post rms" (LF.with_norm L.Norm.Kind.Rms L.Placement.Post base);
  [%expect
    {|
    bare: no norm, bias or layer scale (1 invocations): bitwise
      cycles 260314
    no norm, all flags (1 invocations): bitwise
      cycles 323334
    pre layer, all flags, rope (1 invocations): bitwise
      cycles 436619
    post rms (1 invocations): bitwise
      cycles 373464 |}]

(* The baseline the fused layer is measured against: the same layers
   decomposed, each node generated at binary32 on the core (no grant; the
   contractions on the matrix engine), against the binary32 bundle oracle. *)
let%expect_test
    ("the decomposed layer, node by node, on the VME core" [@tags "disabled"]) =
  List.iter
    (fun (name, c) ->
      let c = { c with LF.Config.bf16_weights = false } in
      let cycles = ref [] in
      let on_run (o : C.Run.t) = cycles := o.C.Run.mcycle :: !cycles in
      T.check
        ~route:(C.route ~on_run ~label:name config)
        ~contractions:true
        ~pipeline:
          (Ssa_backends.Pipeline.Planned
             {
               numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
               target = Ssa_ir.Ssa_target.zve32f128;
             })
        ~oracle:T.Fp32 ~same:T.same_portable ~calls:1 name
        (fun () -> LF.decomposed c);
      Fmt.pr "  cycles %Ld = %a@."
        (List.fold_left Int64.add 0L !cycles)
        Fmt.(list ~sep:(any " + ") int64)
        (List.rev !cycles))
    [
      ( "bare",
        {
          base with
          LF.Config.q_bias = false;
          k_bias = false;
          v_bias = false;
          o_bias = false;
          layer_scale = false;
        } );
      ("no norm, all flags", base);
    ];
  [%expect
    {|
    bare (14 invocations): bitwise
      cycles 2812666 = 14824 + 173202 + 2598 + 14824 + 173202 + 2598 + 14824 + 173202 + 2598 + 1982630 + 2911 + 237302 + 14824 + 3127
    no norm, all flags (15 invocations): bitwise
      cycles 2821597 = 16159 + 173202 + 2598 + 16159 + 173202 + 2598 + 16159 + 173202 + 2598 + 1982524 + 2911 + 237302 + 16159 + 3697 + 3127 |}]
