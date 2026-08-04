(* The cross-backend stack guard behind [Kernel.Limits.Hard.depth] and
   [Hard.eval_depth].

   Those constants are not census maxima — they are empirical stack limits. Both
   backends must survive a validated body at the ceiling, and js_of_ocaml has
   the tighter stack, so node is what picks them. This test is what makes the
   claim falsifiable: it goes red under `make jsoo.inline-runtest` if a compiler,
   jsoo or node upgrade lowers the real threshold below the constant we chose.

   Measured on this tree (see .ai/native_kernel_dsl_design.md for the table):
   natively every traversal survives depth 16384; under node the FIRST failures
   are at 2048 — [Pp.value], [Value.compare], [Value.hash] — while
   [Check.value] still survives there and only fails at 4096. That asymmetry is
   the whole point of probing every traversal rather than the checker alone:
   Kernel's safety argument is "check first, then unmetered [Fold]", and
   clearing the checker's threshold says nothing about [Fold]'s or the
   printer's. Last depth at which everything survives under node: 1024.

   [Eval.value] is the outlier in the other direction — it survives 4096 under
   node and fails at 8192 — which is why the combined [eval_depth] ceiling is
   set higher than the per-body one. It has to be: a whole-program resnet18
   kernel reaches roughly 70 layers x 11 levels of combined depth, and the
   bound must not reject a model the buffer-based evaluator never recurses
   through.

   The verdicts are booleans, so the golden is identical on both backends. *)

(* Stage 1 replaces these with [Kernel.Limits.Hard.depth] / [.eval_depth]; they
   are literals only until that module exists. *)
let hard_depth = 256
let hard_eval_depth = 2048

let survives f =
  try
    ignore (f ());
    true
  with Stack_overflow -> false

(* Built with a loop, not recursion: the builder must not be what overflows. *)
let nest n =
  let leaf =
    Expr.Value.load (Expr.Source.create 0)
      (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
  in
  let e = ref leaf in
  for _ = 1 to n do
    e := Expr.Value.add !e (Expr.Value.const 1.0)
  done;
  !e

let env = Expr.Eval.Env.{ load = (fun _ _ -> Core.return 1.0) }
let origin = Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0

(* Every recursive traversal applied to a body AFTER it has passed [Check]. A
   new one added to [Expr] belongs here, or [Hard.depth] stops covering it. *)
let traversals =
  [
    ("Check.value", fun e -> ignore (Expr.Check.value e));
    ("Fold.sources", fun e -> ignore (Expr.Fold.sources e));
    ("Fold.depth", fun e -> ignore (Expr.Fold.depth e));
    ("Fold.size", fun e -> ignore (Expr.Fold.size e));
    ("Fold.binders", fun e -> ignore (Expr.Fold.binders e));
    ("Fold.free_reducers", fun e -> ignore (Expr.Fold.free_reducers e));
    ("Fold.output_axes", fun e -> ignore (Expr.Fold.output_axes e));
    ("Fold.assume_sites", fun e -> ignore (Expr.Fold.assume_sites e));
    ("Fold.intrinsics", fun e -> ignore (Expr.Fold.intrinsics e));
    ("Pp.value", fun e -> ignore (Format.asprintf "%a" Expr.Pp.value e));
    ( "Rewrite.freshen",
      fun e -> ignore (Expr.Builder.run (Expr.Rewrite.freshen e)) );
    ( "Rewrite.substitute_output",
      fun e ->
        ignore
          (Expr.Rewrite.substitute_output
             (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
             e) );
    ("Rewrite.alpha_normalize", fun e -> ignore (Expr.Rewrite.alpha_normalize e));
    ("Rewrite.map_sources", fun e -> ignore (Expr.Rewrite.map_sources Fun.id e));
    ("Value.compare", fun e -> ignore (Expr.Value.compare e e));
    ("Value.hash", fun e -> ignore (Expr.Value.hash e));
    ("Eval.value", fun e -> ignore (Expr.Eval.value env ~output:origin e));
  ]

let report n names =
  let e = nest n in
  let failed =
    List.filter_map
      (fun (name, f) ->
        if List.mem name names && not (survives (fun () -> f e)) then Some name
        else None)
      traversals
  in
  match failed with
  | [] -> "all survive"
  | fs -> "OVERFLOW in " ^ String.concat "," fs

let%expect_test "Hard.depth: every traversal survives a body at the ceiling" =
  Printf.printf "depth %d: %s\n" hard_depth
    (report hard_depth (List.map fst traversals));
  [%expect {| depth 256: all survive |}]

let%expect_test "Hard.eval_depth: the evaluator survives the combined ceiling" =
  (* Only [Eval.value]: [eval_depth] bounds the recursive [value_at] path, whose
     stack is expression levels summed along a producer chain. The per-body
     traversals are bounded by [Hard.depth] instead. *)
  Printf.printf "eval depth %d: %s\n" hard_eval_depth
    (report hard_eval_depth [ "Eval.value" ]);
  [%expect {| eval depth 2048: all survive |}]
