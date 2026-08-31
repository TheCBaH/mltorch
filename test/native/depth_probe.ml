(* The cross-backend stack guard behind [Kernel.Limits.Hard.depth] and
   [Hard.eval_depth].

   Those constants are not census maxima — they are empirical stack limits. Both
   backends must survive a validated body at the ceiling, and js_of_ocaml has
   the tighter stack, so node is what picks them. This test is what makes the
   claim falsifiable: it goes red under `make jsoo.inline-runtest` if a compiler,
   jsoo or node upgrade lowers the real threshold below the constant we chose.

   Measured on this tree, and under the ORIGINAL raw-body eval_depth formula —
   validation now measures the converted body, so some of the shapes named below
   are rejected outright today (see .ai/native_kernel_dsl_design.md):
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

let hard_depth = Kernel.Limits.Hard.depth
let hard_eval_depth = Kernel.Limits.Hard.eval_depth
let hard_eval_recursion = Kernel.Limits.Hard.eval_recursion

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

let env =
  Expr.Eval.Env.
    {
      load = (fun _ _ -> Err.return 1.0);
      load_index = (fun _ _ -> assert false);
    }

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
    ("Fold.loads", fun e -> ignore (Expr.Fold.loads e));
    ("Fold.intrinsic_sources", fun e -> ignore (Expr.Fold.intrinsic_sources e));
    ( "Rewrite.substitute_loads",
      fun e ->
        ignore
          (Expr.Builder.run (Expr.Rewrite.substitute_loads (fun _ _ -> None) e))
    );
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

(* ---- the recursive path ----------------------------------------------------

   [Hard.eval_depth] above bounds a FLAT expression. It does not bound
   [Kernel_eval.value_at], whose recursion crosses a producer transition per
   value, and a transition costs far more frames than an expression level: a
   1024-value chain whose computed eval_depth is exactly 2048 — accepted by
   [Kernel.create] — overflowed under node. The frontier is also unstable, that
   same chain having passed once before failing three runs in a row, and does
   not fit a tidy cost model: 384 transitions over depth-4 bodies overflows
   while 192 over depth-16 bodies does not.

   So the recursion is bounded at RUNTIME, by [Hard.eval_recursion], where the
   recursion is. The static DAG limits stay generous on purpose: the
   buffer-based [run] never recurses, so a long chain executes perfectly well
   and only the on-demand path is restricted. *)

let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n
let tid = Tensor_id.of_int

let vsig id =
  Tensor_sig.create ~id:(tid id) ~name:"" ~shape:(s1c 1)
    ~fmt:(Payload.Fmt Payload.F32) ()

(* A chain of [n] values, each reading the previous through a body of [d]
   nodes: [n-1] producer transitions for [value_at] on the last one, and an
   eval_depth of roughly [n * d]. *)
let chain ?limits ?(d = 1) n =
  Kernel.create ?limits
    ~inputs:
      [
        {
          Kernel.Input.id = tid 0;
          sg = vsig 0;
          binding = Kernel.Binding.Caller;
        };
      ]
    ~values:
      (List.init n (fun i ->
           {
             Kernel.Value.id = tid (i + 1);
             sg = vsig (i + 1);
             computation =
               Region_program.pixel
                 (let e =
                    ref
                      (Expr.Value.load
                         (Expr_bridge.source_of_id (tid i))
                         (Expr_bridge.coord_of_vec6 Symbolic.out_vec))
                  in
                  for _ = 2 to d do
                    e := Expr.Value.add !e (Expr.Value.const 1.0)
                  done;
                  !e);
             result = Kernel.Result_conversion.Round_f32;
           }))
    ~outputs:[ tid n ]
    ()

let run_chain ?limits ?(d = 1) n =
  match chain ?limits ~d n with
  | Error e ->
      Format.asprintf "rejected: %a" (Core.Pretty.error_kind Kernel.pp_error) e
  | Ok k -> (
      let bind _ = Some (Tensor.materialize (s1c 1) (fun _ -> 1.0)) in
      let origin = Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0 in
      try
        match Kernel_eval.value_at k ~bind (tid n) origin with
        | Ok _ -> "ok"
        | Error e ->
            Format.asprintf "%a" (Core.Pretty.error_kind Kernel_eval.pp_error) e
      with Stack_overflow -> "STACK OVERFLOW")

(* An accepted call stack holds BOTH expression frames and producer-transition
   frames, so a regression could occupy each budget substantially while both
   isolated endpoints below stay green. These sample the accepted frontier along
   it — many/shallow, medium/medium, few/deep — and every one must execute under
   node, not merely be accepted by [Kernel.create]. *)
let%expect_test "the accepted frontier survives in combination" =
  List.iter
    (fun (n, d) -> Printf.printf "n=%3d d=%3d: %s\n" n d (run_chain ~d n))
    [
      (Kernel.Limits.Hard.eval_recursion + 1, 1);
      (Kernel.Limits.Hard.eval_recursion + 1, 13);
      (64, 30);
      (16, 120);
      (8, 125);
    ];
  (* Everything above sits under the DEFAULT max_depth of 128. The public
     custom-limit API accepts far more — up to [Hard.depth] — so default limits
     alone never sample the frontier that actually bounds an accepted kernel.
     A raw depth of 255 plus the result conversion reaches [Hard.depth]
     exactly. *)
  let at_hard_depth =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:4096 ~max_depth:255 ~max_values:4095
         ~max_dep_depth:1024 ~max_inputs:1024 ~max_outputs:1024
         ~max_extent:0x7FFF_FFFFL ~max_numel:0x7FFF_FFFFL)
  in
  Printf.printf "n=  8 d=254 at Hard.depth: %s\n"
    (run_chain ~limits:at_hard_depth ~d:254 8);
  [%expect
    {|
    n= 97 d=  1: ok
    n= 97 d= 13: ok
    n= 64 d= 30: ok
    n= 16 d=120: ok
    n=  8 d=125: ok
    n=  8 d=254 at Hard.depth: ok |}]

let%expect_test "Hard.eval_recursion: the ceiling runs, one past it reports" =
  (* A chain of n values nests n-1 transitions, so the ceiling is reached at
     [eval_recursion + 1] values. Both must hold on BOTH backends: the accepted
     shape survives, and the next one returns a named error rather than dying. *)
  Printf.printf "at the ceiling:   %s\n" (run_chain (hard_eval_recursion + 1));
  Printf.printf "one past it:      %s\n" (run_chain (hard_eval_recursion + 2));
  (* Far past the recursion bound while still WITHIN the static one, so the
     runtime guard is what stops it: 600 depth-3 values give an eval_depth of
     1800, under Hard.eval_depth, but 599 transitions. This is the shape that
     overflowed under node before the guard existed. *)
  Printf.printf "far past it:      %s\n" (run_chain 600);
  [%expect
    {|
    at the ceiling:   ok
    one past it:      recursive evaluation nested more than 96 producers deep
    far past it:      recursive evaluation nested more than 96 producers deep |}]

let%expect_test "the static DAG limits reject before execution" =
  (* [Dependency_too_deep] and [Eval_too_deep] are cheap early guards on the
     stored DAG, distinct from the runtime recursion bound above. They are what
     stop a kernel being built at all. *)
  let tight ~max_dep_depth =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:4096 ~max_depth:128 ~max_values:4096
         ~max_dep_depth ~max_inputs:1024 ~max_outputs:1024
         ~max_extent:0x7FFF_FFFFL ~max_numel:0x7FFF_FFFFL)
  in
  let report name r =
    Printf.printf "%s: %s\n" name
      (match r with
      | Ok _ -> "accepted"
      | Error e ->
          Format.asprintf "%a" (Core.Pretty.error_kind Kernel.pp_error) e)
  in
  let chain_with limits n =
    match chain n with
    | Error _ as e -> e
    | Ok (k : Kernel.t) ->
        Kernel.create ~limits ~inputs:k.Kernel.inputs ~values:k.Kernel.values
          ~outputs:
            (List.map
               (fun (o : Kernel.Output.t) -> o.Kernel.Output.value)
               k.Kernel.outputs)
          ()
  in
  report "dep_depth 4, chain of 4" (chain_with (tight ~max_dep_depth:4) 4);
  report "dep_depth 4, chain of 5" (chain_with (tight ~max_dep_depth:4) 5);
  (* Deep BODIES rather than a long chain, so eval_depth is what runs out first
     — with depth-1 bodies the dependency limit always fires before it, and this
     arm would never be exercised. *)
  report "20 values of depth 100" (chain ~d:100 20);
  report "24 values of depth 100" (chain ~d:100 24);
  [%expect
    {|
    dep_depth 4, chain of 4: accepted
    dep_depth 4, chain of 5: dependency depth exceeds 4
    20 values of depth 100: accepted
    24 values of depth 100: evaluation depth exceeds 2048 |}]
