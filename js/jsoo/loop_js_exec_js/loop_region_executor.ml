(* Builds a real [Region_executor.t] value from [Loop_region_program] +
   [Loop_js_exec]: lower the region_result-built [Region_program.t] (via
   [Region_execution.program], T1's accessor), compile it to JavaScript,
   run it in-process. Any failure at any of those three steps -- a
   [Loop_unsupported] refusal from [Loop_lower], a [Js_compile] the engine
   refuses, a [Js_exception] escaping the kernel -- falls back to
   [Region_executor.default] rather than propagating: this executor's job
   is "run through generated JS when possible", not "require it possible". *)

module Coverage = struct
  type t = { mutable generated_js : int; mutable fallback : int }

  let create () = { generated_js = 0; fallback = 0 }

  (* The coverage assertion: a whole-model run that silently took only the
     fallback path is a worse defect than one that errors loudly, since
     nothing else distinguishes it from a real, passing generated-JS run. A standalone
     function (not folded into the pt2 entry point) so its own mutation --
     "threshold set above what the model actually reaches" -- is a fast unit
     test, not a run of the whole (expensive, real-archive) pipeline. *)
  let check ~min_generated_js t =
    if t.generated_js >= min_generated_js then Ok ()
    else
      Error
        (Fmt.str
           "coverage: generated_js=%d fallback=%d, expected generated_js >= %d"
           t.generated_js t.fallback min_generated_js)
end

(* [Loop_js_exec.compile] already memoises by printed source (its own doc:
   "identical programs share one function"), so no memoisation is added
   here -- a second cache keyed the same way would just be a slower path to
   the same hit. *)
let make ?(limits = Kernel.Limits.default)
    ?(on_fallback =
      fun reason ->
        Printf.eprintf
          "loop_region_executor: falling back to materialize: %s\n%!" reason)
    (coverage : Coverage.t) : Region_executor.t =
 fun ?counters lowered ~env ~bindings ->
  let fallback reason =
    on_fallback reason;
    coverage.Coverage.fallback <- coverage.Coverage.fallback + 1;
    Region_executor.default ?counters lowered ~env ~bindings
  in
  let program = Region_execution.program lowered in
  let out_shape = Region_execution.output_shape lowered in
  match Loop_region_program.lower ~limits ~out_shape ~bindings program with
  | Error e ->
      fallback (Fmt.str "%a" Loop_region_program.pp_error (Err.Error.kind e))
  | Ok loop_program -> (
      let bind id = Tensor_id.Map.find_opt id bindings in
      match Loop_js_exec.exec loop_program ~bind with
      | Error e ->
          fallback (Fmt.str "%a" Loop_js_exec.pp_error (Err.Error.kind e))
      | Ok result -> (
          match Tensor_id.Map.bindings result with
          | [ (_, tensor) ] ->
              coverage.Coverage.generated_js <-
                coverage.Coverage.generated_js + 1;
              Ok tensor
          | _ ->
              fallback
                (Fmt.str "expected exactly one output buffer, got %d"
                   (Tensor_id.Map.cardinal result))))
