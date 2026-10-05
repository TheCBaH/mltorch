open Ssa_bridge

(* One plan through generated WebAssembly: lowered and prepared as the structured
   program is, emitted by [Ssa_wasm], validated and run under node over the
   bound tensors, and judged against [Kernel_eval] exactly as the structured
   interpreter is (or, for a numerical plan, against the interpreter on the same
   program). The Loop program of the same plan supplies the buffer layout and the
   output decoding only: none of its code is emitted. *)

type features = { binary32 : bool; relaxed : bool; vectors : bool }
type outcome = Refused of string | Verdict of Ssa_check.verdict * features

let features (m : Wasm.Module.t) =
  let fs = Wasm_features.of_module m in
  {
    binary32 = false;
    relaxed = List.mem Wasm_features.Relaxed_simd fs;
    vectors = List.mem Wasm_features.Simd128 fs;
  }

let kernel_error = function
  | `Wasm_host m -> Fmt.failwith "the wasm host failed: %s" m
  | `Wasm_invalid _ as e ->
      Fmt.failwith "the module is invalid: %a" Wasm_check.pp_error e
  | `Wasm_unsupported _ -> failwith "the Loop emitter was not asked for"
  | #Loop_ir.Loop_interp.error as e -> (e :> Kernel_eval.error)

let loop_of plan p =
  match Err.payload (Loop_ir.Loop_lower.lower plan) with
  | Error _ -> None
  | Ok loop ->
      Some
        {
          loop with
          Loop_ir.Loop_program.buffers =
            List.map Loop_of_ssa.loop_buffer (Ssa_wasm.arguments p);
        }

let run_program ~relaxed ~prepare ~against (plan : Fusion_plan.t) ~bind =
  match Err.payload (Ssa_lower.Ssa_lower_plan.lower plan) with
  | Error (`Unsupported _) -> Refused "unsupported by lowering"
  | Ok program -> (
      let p = prepare program in
      match (Ssa_wasm.lower ~relaxed_madd:relaxed p, loop_of plan p) with
      | Error e, _ -> Refused (Fmt.str "%a" Ssa_wasm.pp_error e)
      | _, None -> Refused "unsupported by lowering"
      | Ok lowered, Some loop ->
          let reference =
            match against with
            | `Reference -> Kernel_eval.run_plan plan ~bind
            | `Interpreter ->
                Err.map_error
                  (function
                    | `Invalid_program _ | `Cfg_lowering _ | `Invalid_cfg _ ->
                        failwith "the planned program does not run"
                    | ( #Ssa_ir.Ssa_interp.failure
                      | `Binding_mismatch _ | `Unbound_input _ ) as e ->
                        (e :> Kernel_eval.error))
                  (Ssa_lower.Ssa_exec.run plan p ~bind)
          in
          let actual =
            Err.map_error kernel_error
              (Loop_wasm_exec.exec_module ~lowered loop ~bind)
          in
          Verdict
            ( Ssa_check.compare_kernel ~reference ~actual,
              features lowered.Loop_ir.Loop_wasm.module_ ))

let run ?(prepare = Fun.id) ?(relaxed = false) plan ~bind =
  run_program ~relaxed ~prepare ~against:`Reference plan ~bind

let run_against_interpreter ?(relaxed = false) ~prepare plan ~bind =
  run_program ~relaxed ~prepare ~against:`Interpreter plan ~bind
