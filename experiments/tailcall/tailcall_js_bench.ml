open Tailcall_cases

let int_arg position default =
  if position >= 0 && Array.length Sys.argv > position then
    int_of_string Sys.argv.(position)
  else default

let depth = int_arg (Array.length Sys.argv - 2) 64
let rounds = int_arg (Array.length Sys.argv - 1) 200_000
let expression, wanted = binary_chain depth
let sink = ref 0

let run ?(iterations = rounds) name expected f =
  let warmup = min iterations 10_000 in
  for _ = 1 to warmup do
    sink := f ()
  done;
  let before = Sys.time () in
  for _ = 1 to iterations do
    sink := f ()
  done;
  let elapsed = Sys.time () -. before in
  if not (Int.equal !sink expected) then failwith (name ^ ": wrong result");
  Printf.printf "%-28s %8.3fs %10.1f ns/eval\n" name elapsed
    (elapsed *. 1e9 /. float_of_int iterations)

let () =
  let eager = eval_trampoline ~threshold:1 expression in
  let delayed_32 = eval_trampoline ~threshold:32 expression in
  let delayed_128 = eval_trampoline ~threshold:128 expression in
  Printf.printf "binary-depth=%d rounds=%d\n" depth rounds;
  Printf.printf "bounces/eval eager=%d delayed-32=%d delayed-128=%d\n"
    eager.bounces delayed_32.bounces delayed_128.bounces;
  Printf.printf "max-depth    eager=%d delayed-32=%d delayed-128=%d\n"
    eager.max_depth delayed_32.max_depth delayed_128.max_depth;
  let mutual_wanted = mutual_expected depth in
  let control_rounds = rounds * 10 in
  run ~iterations:control_rounds "mutual functions" mutual_wanted (fun () ->
      mutual_tail_a depth 0);
  run ~iterations:control_rounds "mutual payload variant" mutual_wanted
    (fun () -> mutual_variant depth 0);
  run ~iterations:control_rounds "mutual tag + args" mutual_wanted (fun () ->
      mutual_tag depth 0);
  run "direct evaluator" wanted (fun () -> eval_direct expression);
  run "evaluator dispatcher" wanted (fun () -> eval_dispatch expression);
  run "eager trampoline (1)" wanted (fun () ->
      (eval_trampoline ~threshold:1 expression).value);
  run "delayed trampoline (32)" wanted (fun () ->
      (eval_trampoline ~threshold:32 expression).value);
  run "delayed trampoline (128)" wanted (fun () ->
      (eval_trampoline ~threshold:128 expression).value);
  run "explicit list frames" wanted (fun () -> eval_machine expression);
  let reusable_frames = create_reusable_frames 16 in
  run "reused array frames" wanted (fun () ->
      eval_machine_reuse reusable_frames expression);
  run "hybrid cutoff (32)" wanted (fun () ->
      eval_hybrid ~threshold:32 reusable_frames expression);
  run "hybrid cutoff (128)" wanted (fun () ->
      eval_hybrid ~threshold:128 reusable_frames expression)
