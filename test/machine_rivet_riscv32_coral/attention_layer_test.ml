(* A granted bfloat16 attention layer on the VME core: selected, lowered to
   RV32 code, and refused at the instruction memory: resident, its image is
   past the 8 KiB ITCM (the fused attention function is most of it). The
   refusal pins the sizes until the helper can be loaded as an overlay; a run
   would then be bitwise against [Attention_layer_block.eval] at
   [Bf16_paired], with its cycles. *)

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
    deadline = 1200.;
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
      T.check ~route:(C.route ~on_run config)
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

let%expect_test
    "a granted bfloat16 layer is past the VME core's instruction memory" =
  run "no norm, all flags" base;
  run "pre layer, all flags, rope"
    (LF.with_norm L.Norm.Kind.Layer L.Placement.Pre
       { base with LF.Config.rope = LF.Rope_form.Shared });
  run "post rms" (LF.with_norm L.Norm.Kind.Rms L.Placement.Post base);
  [%expect
    {|
    no norm, all flags: refused: invocation 0 (n0): itcm: 9964 bytes needed, 8192 available
      cycles 0
    pre layer, all flags, rope: refused: invocation 0 (n0): itcm: 11240 bytes needed, 8192 available
      cycles 0
    post rms: refused: invocation 0 (n0): itcm: 10384 bytes needed, 8192 available
      cycles 0 |}]
