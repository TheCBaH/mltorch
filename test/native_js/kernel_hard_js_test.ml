(* Independently verifies the three boundaries the tail-call plan's Stage 7
   asks a jsoo test linked to [native_js] for (see .ai/): the raised,
   backend-specific [Eval_too_deep] ceiling this mirror overrides, and the
   two SHARED, unchanged ceilings ([Too_deep]/[Hard.depth],
   [Recursion_too_deep]/[Hard.eval_recursion]) this mirror inherits
   verbatim from lib/native/kernel_hard_shared.ml. Modelled on
   test/native/depth_probe.ml's own [chain]/[run_chain] helpers, against
   [native_js]'s (mirrored, unqualified) [Kernel]/[Kernel_eval]/[Expr]. *)

let hard_eval_recursion = Kernel.Limits.Hard.eval_recursion
let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n
let tid = Tensor_id.of_int

let vsig id =
  Tensor_sig.create ~id:(tid id) ~name:"" ~shape:(s1c 1)
    ~fmt:(Payload.Fmt Payload.F32) ()

(* A chain of [n] values, each reading the previous through a body built by
   [d - 1] rounds of [Value.add] over a [Value.load] leaf: [n-1] producer
   transitions for [value_at] on the last one, a per-value raw expression
   depth ([Expr.Fold.depth]) of [d + 1] (the leaf itself is depth 1, not 0),
   and -- since [Kernel.create]'s combined-depth fold adds 1 more per value
   for the value's own result conversion -- a combined eval depth of
   [n * (d + 2)]. *)
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
               Region_group.Ref.Solo
                 (Region_program.pixel
                    (let e =
                       ref
                         (Expr.Value.load
                            (Expr_bridge.source_of_id (tid i))
                            (Expr_bridge.coord_of_vec6 Symbolic.out_vec))
                     in
                     for _ = 2 to d do
                       e := Expr.Value.add !e (Expr.Value.const 1.0)
                     done;
                     !e));
             result = Kernel.Result_conversion.Round_f32;
           }))
    ~outputs:[ tid n ]
    ()

let create_report ?limits ?d n =
  match chain ?limits ?d n with
  | Ok _ -> "accepted"
  | Error e -> Format.asprintf "%a" (Core.Pretty.error_kind Kernel.pp_error) e

let bind _ = Some (Tensor.materialize (s1c 1) (fun _ -> 1.0))

let run_report ?limits ?d n =
  match chain ?limits ?d n with
  | Error e ->
      Format.asprintf "rejected: %a" (Core.Pretty.error_kind Kernel.pp_error) e
  | Ok k -> (
      match Kernel_eval.run k ~bind with
      | Ok _ -> "ok"
      | Error e ->
          Format.asprintf "%a" (Core.Pretty.error_kind Kernel_eval.pp_error) e)

let value_at_report ?d n =
  match chain ?d n with
  | Error e ->
      Format.asprintf "rejected: %a" (Core.Pretty.error_kind Kernel.pp_error) e
  | Ok k -> (
      let origin = Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0 in
      try
        match Kernel_eval.value_at k ~bind (tid n) origin with
        | Ok _ -> "ok"
        | Error e ->
            Format.asprintf "%a" (Core.Pretty.error_kind Kernel_eval.pp_error) e
      with Stack_overflow -> "STACK OVERFLOW")

(* ---- Eval_too_deep, at native_js's OWN (overridden) ceiling ------------- *)

let%expect_test
    "Eval_too_deep: admits and runs at the mirror's own ceiling, rejects one \
     value past" =
  (* [d = 2], so each value contributes [d + 2 = 4] to the combined depth:
     [3072 * 4 = 12288], exactly native_js's own [Hard.eval_depth] (12288 --
     see js/jsoo/native_js/kernel_hard.ml; native's own mirror-independent
     value is 1536, and this test would need different constants there).
     One more value of the same shape reaches 12292, past the ceiling.
     Needs [max_dep_depth]/[max_values] above the chain length -- both
     still well under their own unchanged Hard ceilings (4096, 65536). *)
  let limits =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:4096 ~max_depth:128 ~max_values:4095
         ~max_dep_depth:4095 ~max_inputs:1024 ~max_outputs:1024
         ~max_extent:0x7FFF_FFFFL ~max_numel:0x7FFF_FFFFL ~max_local_slots:8192
         ~max_scan_state:8192 ~max_scan_updates_per_key:8192L
         ~max_scan_updates_total:16_000_000L)
  in
  Printf.printf "at the ceiling, create: %s\n" (create_report ~limits ~d:2 3072);
  Printf.printf "at the ceiling, run:    %s\n" (run_report ~limits ~d:2 3072);
  Printf.printf "one value past:         %s\n" (create_report ~limits ~d:2 3073);
  [%expect
    {|
    at the ceiling, create: accepted
    at the ceiling, run:    ok
    one value past:         evaluation depth exceeds 12288
    |}]

(* ---- Too_deep, the SHARED per-body ceiling, unaffected by the override -- *)

let%expect_test
    "Too_deep: the largest admissible max_depth (255, since Hard.depth = 256) \
     admits a depth-255 body, rejects depth-256" =
  let limits =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:4096 ~max_depth:255 ~max_values:4095
         ~max_dep_depth:1024 ~max_inputs:1024 ~max_outputs:1024
         ~max_extent:0x7FFF_FFFFL ~max_numel:0x7FFF_FFFFL ~max_local_slots:8192
         ~max_scan_state:8192 ~max_scan_updates_per_key:8192L
         ~max_scan_updates_total:16_000_000L)
  in
  (* Unlike the combined-depth check above, this one -- [Kernel.create]'s
     per-value [Region_group.Ref.project ~max_depth], the "one Expr.Value.t"
     dimension kernel.mli's own module doc distinguishes from the combined
     one -- compares [Expr.Fold.depth] directly, with no +1 for the result
     conversion: [d = 254] builds a body of raw depth 255 (the leaf itself
     counts as depth 1), exactly at the configured ceiling; [d = 255] builds
     depth 256, one past it. *)
  Printf.printf "raw depth 255: %s\n" (create_report ~limits ~d:254 1);
  Printf.printf "raw depth 256: %s\n" (create_report ~limits ~d:255 1);
  [%expect
    {|
    raw depth 255: accepted
    raw depth 256: t1: depth exceeds limit 255
    |}]

(* ---- Recursion_too_deep, the SHARED runtime recursion ceiling ----------- *)

let%expect_test
    "Recursion_too_deep: value_at nests to the ceiling, reports one producer \
     past it" =
  Printf.printf "at the ceiling (n=%d): %s\n" (hard_eval_recursion + 1)
    (value_at_report (hard_eval_recursion + 1));
  Printf.printf "one past (n=%d):       %s\n" (hard_eval_recursion + 2)
    (value_at_report (hard_eval_recursion + 2));
  [%expect
    {|
    at the ceiling (n=97): ok
    one past (n=98):       recursive evaluation nested more than 96 producers deep
    |}]
