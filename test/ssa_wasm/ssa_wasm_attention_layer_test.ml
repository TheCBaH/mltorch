open Ssa_bridge
module Lf = Loop_ir_test.Loop_fixtures
module T = Ssa_ir_test.Ssa_attention_layer_test

(* A granted attention layer's sequence, its attention expanded and narrowed,
   through generated WebAssembly: the output is [Attention_layer_block.eval]'s, bit for
   bit. *)

let bind (pr : T.Prepared.t) id =
  List.find_map
    (fun (n, (h, w, cs)) ->
      if Tensor_id.equal id (Lf.tid n) then
        let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c:1 in
        Some (Lf.f32_tensor shape (fun c -> cs.((Vec6.offset shape c :> int))))
      else None)
    pr.T.Prepared.inputs

let read (pr : T.Prepared.t) outputs =
  match Tensor_id.Map.find_opt (Lf.tid T.Id.y) outputs with
  | Some t ->
      let e = pr.T.Prepared.width in
      Array.init pr.T.Prepared.cells (fun i ->
          Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:(i / e) ~w:(i mod e) ~c:0))
  | None -> [||]

let run (pr : T.Prepared.t) =
  let p = pr.T.Prepared.program in
  let lowered =
    match Ssa_wasm.lower p with
    | Ok k -> k
    | Error e -> Fmt.failwith "%a" Ssa_wasm.pp_error e
  in
  let loop =
    Lf.program
      ~buffers:(List.map Loop_of_ssa.loop_buffer (Ssa_wasm.arguments p))
      []
  in
  match
    Err.payload (Loop_wasm_exec.exec_module ~lowered loop ~bind:(bind pr))
  with
  | Error e -> Fmt.failwith "%a" Loop_wasm_exec.pp_error e
  | Ok outputs -> read pr outputs

let%expect_test "an attention layer through Wasm equals the reference" =
  T.sweep ~backend:"Wasm" run;
  [%expect
    {|
    f32_ordered, bare: Wasm equals the reference (80 of 80 nonzero finite)
    f32_ordered, pre layer, all flags, masked: Wasm equals the reference (80 of 80 nonzero finite)
    f32_ordered, post layer, all flags: Wasm equals the reference (80 of 80 nonzero finite)
    f32_ordered, pre rms, qkv biases, shared rope, masked: Wasm equals the reference (80 of 80 nonzero finite)
    f32_ordered, pre layer w, all flags, batched rope: Wasm equals the reference (80 of 80 nonzero finite)
    f32_ordered, no norm, all flags, shared rope, masked, 7 tokens: Wasm equals the reference (112 of 112 nonzero finite)
    f32_ordered, post rms, one head of 12, 9 tokens: Wasm equals the reference (216 of 216 nonzero finite)
    f32_fused, bare: Wasm equals the reference (80 of 80 nonzero finite)
    f32_fused, pre layer, all flags, masked: Wasm equals the reference (80 of 80 nonzero finite)
    f32_fused, post layer, all flags: Wasm equals the reference (80 of 80 nonzero finite)
    f32_fused, pre rms, qkv biases, shared rope, masked: Wasm equals the reference (80 of 80 nonzero finite)
    f32_fused, pre layer w, all flags, batched rope: Wasm equals the reference (80 of 80 nonzero finite)
    f32_fused, no norm, all flags, shared rope, masked, 7 tokens: Wasm equals the reference (112 of 112 nonzero finite)
    f32_fused, post rms, one head of 12, 9 tokens: Wasm equals the reference (216 of 216 nonzero finite) |}]
