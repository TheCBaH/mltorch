(* Native route of the Loop IR benchmark: [Loop_interp.run] under ocamlopt.
   The jsoo route (js/loop_js_exec/bench/loop_bench_jsoo.ml) measures the
   same interpreter under node plus the generated-code executor, over the
   identical program (loop_bench_program.ml), so the three numbers are
   comparable. Run by `make loop_js.bench`; not part of runtest. *)

open Loop_ir_test

let () =
  Loop_bench_run.main
    [ Loop_bench_run.interp_case ~label:"interp (native, ocamlopt)" ]
