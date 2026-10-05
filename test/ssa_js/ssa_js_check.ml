open Ssa_bridge

(* One plan through generated JavaScript: lowered and prepared as the structured
   program is, emitted by [Ssa_js], compiled by the engine and run in this
   process over the bound tensors' own storage, and judged against
   [Kernel_eval] exactly as the structured interpreter is. The Loop program of
   the same plan supplies the argument arrays and their validation only: none
   of its code is emitted. *)

type outcome = Refused of string | Verdict of Ssa_check.verdict

let kernel_error = function
  | `Js_compile m -> Fmt.failwith "the engine refused the emitted source: %s" m
  | `Js_exception m -> Fmt.failwith "a host exception escaped the kernel: %s" m
  | #Loop_ir.Loop_interp.error as e -> (e :> Kernel_eval.error)

let run ?(prepare = Fun.id) (plan : Fusion_plan.t) ~bind =
  match
    ( Err.payload (Ssa_lower.Ssa_lower_plan.lower plan),
      Err.payload (Loop_ir.Loop_lower.lower plan) )
  with
  | Error (`Unsupported _), _ | _, Error (`Unsupported _) ->
      Refused "unsupported by lowering"
  | Ok program, Ok loop -> (
      let p = prepare program in
      match
        try Ssa_js.program p
        with Invalid_argument m -> Fmt.failwith "%s@.%a" m Ssa_ir.Ssa_pp.pp p
      with
      | Error e -> Refused (Fmt.str "%a" Ssa_js.pp_error e)
      | Ok (ast, sites) ->
          let source = Js_print.factory_body ast in
          let loop =
            {
              loop with
              Loop_ir.Loop_program.buffers =
                List.map Loop_of_ssa.loop_buffer (Ssa_js.arguments p);
            }
          in
          let reference = Kernel_eval.run_plan plan ~bind in
          let actual =
            match
              Err.payload (Loop_js_exec.compile_kernel ~sites loop source)
            with
            | Error (`Js_compile m) ->
                Fmt.failwith "the engine refused the emitted source: %s\n%s" m
                  source
            | Ok compiled ->
                Err.map_error kernel_error (Loop_js_exec.run compiled ~bind)
          in
          Verdict (Ssa_check.compare_kernel ~reference ~actual))
