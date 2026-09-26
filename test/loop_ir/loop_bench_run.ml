(* Shared timing harness for the Loop IR benchmarks, native and jsoo: warmup,
   then best-of-[trials] repetitions of [iterations] calls to a compiled
   case -- [Sys.time]-based like test/expr_bench's [time_it] (same
   reasoning: best-of-N filters GC pauses and scheduler jitter without an
   average's downward bias, and enough calls per trial clear [Sys.time]'s
   coarse resolution under jsoo/node). [warmup] and the early trials are also
   what gives a JIT engine (V8 under node) a chance to tier up the hot path,
   for both the interpreter and the compiled kernel: both are called through
   the SAME [run_once] closure every time, from the first warmup call
   onward, so neither gets a cold-start advantage over the other.

   Every case is checked once against [Loop_interp.run]'s own output before
   it is timed: a broken generated program would otherwise be benchmarked as
   if it were a correct one. *)

open Loop_ir

let iterations = 200
let warmup = 50
let trials = 7

type case = {
  label : string;
  run_once : unit -> (Tensor.packed Tensor_id.Map.t, string) result;
}

let time_it f =
  for _ = 1 to warmup do
    ignore (f ())
  done;
  let best = ref infinity in
  for _ = 1 to trials do
    let before = Sys.time () in
    for _ = 1 to iterations do
      ignore (f ())
    done;
    let elapsed = Sys.time () -. before in
    if elapsed < !best then best := elapsed
  done;
  !best

let report ~label elapsed =
  Printf.printf
    "%-28s %10.1f ns/elem  (%d elems x %d iterations, best of %d trials)\n%!"
    label
    (elapsed *. 1e9 /. float_of_int (iterations * Loop_bench_program.size))
    Loop_bench_program.size iterations trials

let matches reference outputs =
  Tensor_id.Map.cardinal outputs = Tensor_id.Map.cardinal reference
  && Tensor_id.Map.for_all
       (fun id t ->
         match Tensor_id.Map.find_opt id reference with
         | Some r -> Loop_check.tensors_equal t r
         | None -> false)
       outputs

let interp_case ~label =
  {
    label;
    run_once =
      (fun () ->
        Result.map_error
          (fun e -> Fmt.str "%a" Loop_interp.pp_error e)
          (Err.payload
             (Loop_interp.run Loop_bench_program.program
                ~bind:Loop_bench_program.bind)));
  }

let run_case ~reference (c : case) =
  match c.run_once () with
  | Error e -> Printf.printf "%-28s FAILED: %s\n%!" c.label e
  | Ok outputs when not (matches reference outputs) ->
      Printf.printf "%-28s MISMATCH against the interpreter\n%!" c.label
  | Ok _ -> report ~label:c.label (time_it c.run_once)

let main cases =
  let reference =
    Err.or_raise ~pp_error:Loop_interp.pp_error
      (Loop_interp.run Loop_bench_program.program ~bind:Loop_bench_program.bind)
  in
  Printf.printf
    "loop_js bench: %d-element elementwise kernel, %d iterations x %d trials \
     (best of trials, warmup %d)\n\
     %!"
    Loop_bench_program.size iterations trials warmup;
  List.iter (run_case ~reference) cases
