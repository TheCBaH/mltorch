open Ssa_bridge

(* One plan through generated C: lowered and prepared as the structured program
   is, emitted by [Ssa_c], compiled and run over the bound tensors, and judged
   against [Kernel_eval] exactly as the structured interpreter is. The Loop
   program of the same plan supplies the buffer layout and the output decoding
   only: none of its code is emitted. *)

(* What the emitted text holds, so a sweep can say it was not vacuous. *)
type features = { binary32 : bool; fused : bool; vectors : bool }
type outcome = Refused of string | Verdict of Ssa_check.verdict * features

let contains text sub =
  let n = String.length sub and m = String.length text in
  let rec go i = i + n <= m && (String.sub text i n = sub || go (i + 1)) in
  go 0

let features (k : Loop_ir.Loop_c.t) =
  let t = k.Loop_ir.Loop_c.source in
  {
    binary32 = contains t "float v";
    fused = contains t "fma(" || contains t "fmaf(";
    vectors = contains t "ssa_v";
  }

let kernel_error = function
  | `C_compile m ->
      Fmt.failwith "the C compiler rejected the emitted source: %s" m
  | `C_host m -> Fmt.failwith "the C host failed: %s" m
  | `C_unsupported _ -> failwith "the Loop emitter was not asked for"
  | #Loop_ir.Loop_interp.error as e -> (e :> Kernel_eval.error)

let run_program ?(prepare = Fun.id) (plan : Fusion_plan.t) ~bind =
  match
    ( Err.payload (Ssa_lower.Ssa_lower_plan.lower plan),
      Err.payload (Loop_ir.Loop_lower.lower plan) )
  with
  | Error (`Unsupported _), _ | _, Error (`Unsupported _) ->
      Refused "unsupported by lowering"
  | Ok program, Ok loop -> (
      let p = prepare program in
      match Ssa_c.kernel ~name:Loop_c_exec.kernel_name p with
      | Error e -> Refused (Fmt.str "%a" Ssa_c.pp_error e)
      | Ok (kernel, sites) ->
          let reference = Kernel_eval.run_plan plan ~bind in
          let actual =
            Err.map_error kernel_error
              (Loop_c_exec.exec_kernel ~kernel ~sites
                 {
                   loop with
                   Loop_ir.Loop_program.buffers =
                     List.map Loop_of_ssa.loop_buffer (Ssa_c.arguments p);
                 }
                 ~bind)
          in
          Verdict (Ssa_check.compare_kernel ~reference ~actual, features kernel)
      )

let run ?prepare plan ~bind = run_program ?prepare plan ~bind

(* The same plan, as a program the structured interpreter also runs: generated C
   must agree with it bit for bit. For the programs a numerical plan makes, whose
   own checks against the reference are the planner suites', this is the question
   a C consumer owes an answer to. *)
let run_against_interpreter ~prepare (plan : Fusion_plan.t) ~bind =
  match
    ( Err.payload (Ssa_lower.Ssa_lower_plan.lower plan),
      Err.payload (Loop_ir.Loop_lower.lower plan) )
  with
  | Error (`Unsupported _), _ | _, Error (`Unsupported _) ->
      Refused "unsupported by lowering"
  | Ok program, Ok loop -> (
      let p = prepare program in
      match Ssa_c.kernel ~name:Loop_c_exec.kernel_name p with
      | Error e -> Refused (Fmt.str "%a" Ssa_c.pp_error e)
      | Ok (kernel, sites) ->
          let reference =
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
              (Loop_c_exec.exec_kernel ~kernel ~sites
                 {
                   loop with
                   Loop_ir.Loop_program.buffers =
                     List.map Loop_of_ssa.loop_buffer (Ssa_c.arguments p);
                 }
                 ~bind)
          in
          Verdict (Ssa_check.compare_kernel ~reference ~actual, features kernel)
      )
