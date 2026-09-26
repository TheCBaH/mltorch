(* jsoo route of the Loop IR benchmark: [Loop_interp.run] and the generated
   JavaScript executor, both under node, over the same program the native
   route (test/loop_ir/loop_bench_native.ml) measures. Run by
   `make loop_js.bench`; not part of jsoo.runtest/jsoo.inline-runtest -- a
   timing report, not a gate. *)

let generated_case =
  let label = "generated (jsoo/node)" in
  match Err.payload (Loop_js_exec.compile Loop_bench_program.program) with
  | Error e ->
      {
        Loop_bench_run.label;
        run_once = (fun () -> Error (Fmt.str "%a" Loop_js_exec.pp_error e));
      }
  | Ok compiled ->
      {
        Loop_bench_run.label;
        run_once =
          (fun () ->
            Result.map_error
              (fun e -> Fmt.str "%a" Loop_js_exec.pp_error e)
              (Err.payload
                 (Loop_js_exec.run compiled ~bind:Loop_bench_program.bind)));
      }

let () =
  Loop_bench_run.main
    [ Loop_bench_run.interp_case ~label:"interp (jsoo/node)"; generated_case ]
