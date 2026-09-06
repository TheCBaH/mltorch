open Tailcall_cases

(* Melange exposes node's argv including the node executable, while native and
   js_of_ocaml expose the OCaml convention.  Reading an optional final integer
   keeps the experiment about recursion rather than argv shims. *)
let depth =
  let last = Sys.argv.(Array.length Sys.argv - 1) in
  match int_of_string_opt last with Some n -> n | None -> 200_000

let report name expected f =
  Printf.printf "%-24s %s\n" name (outcome_name (observe expected f))

let report_expr name build eval =
  let expression, expected = build depth in
  report name expected (fun () -> eval expression)

let () =
  Printf.printf "depth %d\n" depth;
  report "self tail" depth (fun () -> self_tail depth 0);
  report "mutual tail" (mutual_expected depth) (fun () -> mutual_tail_a depth 0);
  report "mutual via variant" (mutual_expected depth) (fun () ->
      mutual_variant depth 0);
  report "mutual via tag args" (mutual_expected depth) (fun () ->
      mutual_tag depth 0);
  report "indirect tail" depth (fun () -> indirect_tail depth 0);
  report_expr "direct binary" binary_chain eval_direct;
  report_expr "direct unary" unary_chain eval_direct;
  report_expr "direct value/guard" guard_chain eval_direct;
  report_expr "variant binary" binary_chain eval_dispatch;
  report_expr "variant unary" unary_chain eval_dispatch;
  report_expr "variant value/guard" guard_chain eval_dispatch;
  report_expr "eager trampoline binary" binary_chain eval_trampoline_eager;
  report_expr "eager trampoline unary" unary_chain eval_trampoline_eager;
  report_expr "eager trampoline guard" guard_chain eval_trampoline_eager;
  report_expr "delayed trampoline binary" binary_chain eval_trampoline_delayed;
  report_expr "delayed trampoline unary" unary_chain eval_trampoline_delayed;
  report_expr "delayed trampoline guard" guard_chain eval_trampoline_delayed;
  report_expr "machine binary" binary_chain eval_machine;
  report_expr "machine unary" unary_chain eval_machine;
  report_expr "machine value/guard" guard_chain eval_machine;
  let reusable_frames = create_reusable_frames 16 in
  report_expr "reused frames binary" binary_chain
    (eval_machine_reuse reusable_frames);
  report_expr "reused frames unary" unary_chain
    (eval_machine_reuse reusable_frames);
  report_expr "reused frames guard" guard_chain
    (eval_machine_reuse reusable_frames);
  report_expr "hybrid binary" binary_chain
    (eval_hybrid ~threshold:128 reusable_frames);
  report_expr "hybrid unary" unary_chain
    (eval_hybrid ~threshold:128 reusable_frames);
  report_expr "hybrid guard" guard_chain
    (eval_hybrid ~threshold:128 reusable_frames);
  let expected = mutual_expected depth in
  Printf.printf "cppo-selected %-8s " Backend_driver.name;
  print_endline
    (outcome_name (observe expected (fun () -> Backend_driver.run depth)))
