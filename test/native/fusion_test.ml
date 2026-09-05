(* Buffer-elimination fusion: Conv2D followed by Add, executed with the conv's
   buffer removed, bit-for-bit identical to materialising both.

   Three comparisons, because bitwise equality alone cannot demonstrate
   PLACEMENT — both paths deliberately compute the same value. The result-map
   membership assertions are what show the buffer actually went away. *)

open Graph_ir

type error =
  [ `Adapt of Kernel_adapt.error
  | `Direct of Eval_direct.error
  | `Eval of Kernel_eval.error ]

let pp_error ppf : [< error ] -> unit = function
  | `Adapt e -> Kernel_adapt.pp_error ppf e
  | `Direct e -> Eval_direct.pp_error ppf e
  | `Eval e -> Kernel_eval.pp_error ppf e

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:pp_error
let lift f r = Err.map_error (fun e -> f e) r
let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n

let kernel_of g =
  Err.or_raise ~pp_error:Kernel_adapt.pp_error
    (Kernel_adapt.of_stage_program (Eval_symbolic.run g))

let ramp shape =
  Tensor.materialize shape (fun c ->
      float_of_int (Dim.to_int (Vec6.get c Axis.C) + 1))

let bind_of g =
 fun id ->
  if List.mem id g.Graph.inputs then
    Some (ramp (Tensor_id.Map.find id g.Graph.tensors).Tensor_sig.shape)
  else None

let pp_ids ppf ids =
  Fmt.string ppf
    (match ids with
    | [] -> "-"
    | ids ->
        String.concat "," (List.map (Format.asprintf "%a" Tensor_id.pp) ids))

let stored out = Tensor_id.Map.bindings out |> List.map fst

let%expect_test "Fusion: the planner's decisions on conv_add" =
  let k = kernel_of (Graph_fixtures.conv_add ()) in
  let p, decisions = Fusion_plan.plan k in
  Format.printf "@[<v>%a@,@,virtual: %s@,stores:  %s@]@."
    (Fmt.list ~sep:Fmt.cut Fusion_plan.Decision.pp)
    decisions
    (String.concat ","
       (List.map
          (Format.asprintf "%a" Kernel.Use.pp)
          (Kernel.Use.Set.elements p.Fusion_plan.virtual_uses)))
    (String.concat ","
       (List.map
          (Format.asprintf "%a" Tensor_id.pp)
          (Tensor_id.Set.elements p.Fusion_plan.stores)));
  [%expect {|
    virtualize t4->t5

    virtual: t4->t5
    stores:  t5 |}]

let%expect_test "Fusion: conv_add is bit-for-bit under both placements" =
  let g = Graph_fixtures.conv_add () in
  let k = kernel_of g in
  let bind = bind_of g in
  let p, _ = Fusion_plan.plan k in
  let result =
    let open Err.Syntax in
    let* materialized =
      lift
        (fun e -> `Eval e)
        (Kernel_eval.run_plan (Fusion_plan.default k) ~bind)
    in
    let* fused = lift (fun e -> `Eval e) (Kernel_eval.run_plan p ~bind) in
    let tensors =
      List.map (fun id -> (id, Option.get (bind id))) g.Graph.inputs
    in
    let* direct =
      lift (fun e -> `Direct e) (Eval_direct.run g ~inputs:tensors)
    in
    let out = List.hd g.Graph.outputs in
    let get m = Tensor_id.Map.find_opt out m in
    let same a b =
      match (a, b) with Some x, Some y -> Tensor.equal_bits x y | _ -> false
    in
    (* (3) the derived elaborated body, materialised through the consumer's own
       conversion — [elaborate] deliberately returns it unrounded, so the test
       applies exactly what the evaluator would. Omitting that would differ at
       the final f32 boundary for a legitimate reason and read as a bug. *)
    let use = Kernel.Use.Set.choose p.Fusion_plan.virtual_uses in
    let consumer = Option.get (Kernel.value k use.Kernel.Use.consumer) in
    let body =
      Err.or_raise ~pp_error:Kernel_elab.pp_error (Kernel_elab.elaborate k use)
    in
    let inputs_only id = if List.mem id g.Graph.inputs then bind id else None in
    let env = Expr_bridge.env ~binding:inputs_only in
    let elaborated =
      Tensor.materialize consumer.Kernel.Value.sg.Tensor_sig.shape (fun c ->
          Err.or_raise ~pp_error:Expr.Eval.pp_error
            (Expr.Eval.value env
               ~output:(Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int c))
               (Kernel.Result_conversion.apply consumer.Kernel.Value.result body)))
    in
    Err.return
      [
        ( "materialized == direct",
          same (get materialized) (Tensor_id.Map.find_opt out direct) );
        ("fused == materialized", same (get fused) (get materialized));
        ("elaborated == fused", same (Some elaborated) (get fused));
      ]
  in
  Format.printf "@[<v>%a@]@."
    (pp_result
       (Fmt.list ~sep:Fmt.cut (fun ppf (n, ok) -> Fmt.pf ppf "%s: %b" n ok)))
    result;
  (* Placement is STRUCTURAL and must be pinned separately: the two runs
     deliberately compute the same value, so bitwise equality says nothing about
     whether the conv buffer went away. *)
  let materialized =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run_plan (Fusion_plan.default k) ~bind)
  in
  let fused =
    Err.or_raise ~pp_error:Kernel_eval.pp_error (Kernel_eval.run_plan p ~bind)
  in
  Format.printf "stored when materialized: %a@.stored when fused:        %a@."
    pp_ids (stored materialized) pp_ids (stored fused);
  [%expect
    {|
    materialized == direct: true
    fused == materialized: true
    elaborated == fused: true
    stored when materialized: t4,t5
    stored when fused:        t5 |}]

(* ---- the inner round -------------------------------------------------------

   The mutation-sensitive case. Integer weights and inputs chosen so the
   convolution accumulates in binary64 to exactly 2^24+1, with a residual of 1:

     stored: round_f32(2^24+1) = 2^24, +1 -> 2^24+1, stored f32 (ties-to-even)
             -> 2^24
     virtual WITHOUT the inner round: 2^24+1 + 1 = 2^24+2, exact in f32

   Dropping the conversion is invisible while the producer is STORED — the f32
   store rounds anyway — and becomes observable exactly here, where there is no
   store to do it. *)

let round_fixture () =
  let conv_axis k st p : Conv.Conv2d.axis_window =
    {
      kernel = Dim.extent k;
      stride = Op_config.Pos.of_int st;
      pad_before = Op_config.Nonneg.of_int p;
      pad_after = Op_config.Nonneg.of_int p;
      dilation = Op_config.Pos.of_int 1;
    }
  in
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name:"round_conv_add" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 1 1 2) () in
      let* w = input ~shape:(s 1 1 1 1 1 2) () in
      let* b = input ~shape:(s1c 1) () in
      let* r = input ~shape:(s1c 1) () in
      let* c =
        conv2d
          {
            h = conv_axis 1 1 0;
            w = conv_axis 1 1 0;
            in_channels = Dim.extent 2;
            groups = Op_config.Pos.of_int 1;
          }
          ~x ~weight:w ~bias:b ()
      in
      add c r)

let%expect_test "Fusion: the inner round survives buffer elimination" =
  let g = round_fixture () in
  let k = kernel_of g in
  let two24 = 16777216.0 in
  (* x . w = 2^24 * 1 + 1 * 1 = 2^24 + 1, plus a zero bias. *)
  let data =
    [
      (0, [| two24; 1.0 |]); (1, [| 1.0; 1.0 |]); (2, [| 0.0 |]); (3, [| 1.0 |]);
    ]
  in
  let bind id =
    let idx = List.mapi (fun i x -> (i, x)) g.Graph.inputs in
    match List.find_opt (fun (_, x) -> Tensor_id.equal x id) idx with
    | None -> None
    | Some (i, _) ->
        let vals = List.assoc i data in
        let sg = Tensor_id.Map.find id g.Graph.tensors in
        Some
          (Tensor.materialize sg.Tensor_sig.shape (fun c ->
               vals.(Dim.to_int (Vec6.get c Axis.C))))
  in
  let p, _ = Fusion_plan.plan k in
  let out = List.hd g.Graph.outputs in
  let read m = Tensor.read_at_raw (Tensor_id.Map.find out m) (fun _ -> 0) in
  let materialized =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run_plan (Fusion_plan.default k) ~bind)
  in
  let fused =
    Err.or_raise ~pp_error:Kernel_eval.pp_error (Kernel_eval.run_plan p ~bind)
  in
  (* The ELABORATED body belongs here too, not only on conv_add: that fixture's
     integer data makes the round a no-op, so dropping it inside [Kernel_elab]
     changes nothing there and the check would be vacuous. This is the only
     place the elaborator's own conversion is observable. *)
  let use = Kernel.Use.Set.choose p.Fusion_plan.virtual_uses in
  let consumer = Option.get (Kernel.value k use.Kernel.Use.consumer) in
  let body =
    Err.or_raise ~pp_error:Kernel_elab.pp_error (Kernel_elab.elaborate k use)
  in
  let elaborated =
    Tensor.materialize consumer.Kernel.Value.sg.Tensor_sig.shape (fun c ->
        Err.or_raise ~pp_error:Expr.Eval.pp_error
          (Expr.Eval.value
             (Expr_bridge.env ~binding:bind)
             ~output:(Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int c))
             (Kernel.Result_conversion.apply consumer.Kernel.Value.result body)))
  in
  Format.printf
    "materialized: %.1f@.fused:        %.1f@.elaborated:   %.1f@.agree: %b@."
    (read materialized) (read fused)
    (Tensor.read_at_raw elaborated (fun _ -> 0))
    (Tensor.equal_bits
       (Tensor_id.Map.find out materialized)
       (Tensor_id.Map.find out fused)
    && Tensor.equal_bits (Tensor_id.Map.find out fused) elaborated);
  Format.printf "without the inner round it would be %.1f@."
    (two24 +. 1.0 +. 1.0);
  [%expect
    {|
    materialized: 16777216.0
    fused:        16777216.0
    elaborated:   16777216.0
    agree: true
    without the inner round it would be 16777218.0 |}]

(* ---- rejections ------------------------------------------------------------ *)

let%expect_test "Fusion: what the planner refuses, and why" =
  let case name g =
    let k = kernel_of g in
    let _, decisions = Fusion_plan.plan k in
    Format.printf "@[<v>%s:@,%a@]@." name
      (Fmt.list ~sep:Fmt.cut Fusion_plan.Decision.pp)
      decisions
  in
  (* Fan-out: the producer has two ordinary uses, so no edge may take it. *)
  case "fan-out" (Graph_fixtures.diamond ());
  (* A max-pool consumer reaches its producer ONLY through the descriptor, which
     [substitute_loads] cannot touch — the edge is in [Kernel.uses] and not in
     [load_uses], so the planner must say so rather than silently leave the body
     unchanged. It needs a VALUE producer: where max-pool reads a graph input
     directly, the pair is not a [Use] at all. *)
  let pool_chain =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"pool_chain" ~outputs:(fun r -> [ r ])
        @@
        let* x = input ~shape:(s 1 1 1 4 4 2) () in
        let* r = relu x in
        let* v, i =
          max_pool2d_with_indices
            {
              ceil_mode = false;
              kernel = { h = Dim.extent 2; w = Dim.extent 2 };
              stride =
                { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
              pad =
                { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
            }
            r
        in
        let* () = discard i in
        return v)
  in
  case "intrinsic consumer" pool_chain;
  (* A chain: both edges are individually legal, only one may be selected, and
     the loser is reported as overlapping rather than silently dropped. *)
  case "pointwise chain" (Graph_fixtures.residual ());
  (* THREE consumers. The fan-out case above has exactly two, so it cannot tell
     an exact count from a saturating one — the planner's counter stops at two
     to stay safe against a 32-bit wrap, and the diagnostic must not claim more
     than it knows. *)
  let three_uses =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"three_uses" ~outputs:(fun (a, b, c) -> [ a; b; c ])
        @@
        let* x = input ~shape:(s1c 4) () in
        let* p = relu x in
        let* a = relu p in
        let* b = relu p in
        let* c = relu p in
        return (a, b, c))
  in
  case "three consumers" three_uses;
  [%expect
    {|
    fan-out:
    virtualize t1->t3
    reject t2: t2->t3 overlaps the already-selected t1->t3
    intrinsic consumer:
    reject t1: t1->t2 is reached through an intrinsic descriptor
    pointwise chain:
    reject t1: t1 has at least 2 ordinary uses
    virtualize t2->t3
    three consumers:
    reject t1: t1 has at least 2 ordinary uses |}]

let%expect_test "Fusion: the elaborator's site class is not the planner's" =
  (* Two edges the ELABORATOR accepts and the planner rejects, pinning that
     site-local support and placement legality are separate concerns. *)
  let g = Graph_fixtures.diamond () in
  let k = kernel_of g in
  let _, decisions = Fusion_plan.plan k in
  let elaborates u =
    match Kernel_elab.elaborate k u with
    | Ok _ -> "ok"
    | Error e ->
        Format.asprintf "%a" (Core.Pretty.error_kind Kernel_elab.pp_error) e
  in
  Format.printf "@[<v>planner:@,%a@,@,elaborator on each load edge:@,%a@]@."
    (Fmt.list ~sep:Fmt.cut Fusion_plan.Decision.pp)
    decisions
    (Fmt.list ~sep:Fmt.cut (fun ppf u ->
         Fmt.pf ppf "%a: %s" Kernel.Use.pp u (elaborates u)))
    (Kernel.Use.Set.elements (Kernel.load_uses k));
  [%expect
    {|
    planner:
    virtualize t1->t3
    reject t2: t2->t3 overlaps the already-selected t1->t3

    elaborator on each load edge:
    t1->t3: ok
    t2->t3: ok |}]

let%expect_test "Fusion: elaboration rejects what it cannot substitute" =
  let g = Graph_fixtures.conv_add () in
  let k = kernel_of g in
  let show name u =
    Format.printf "%s: %a@." name
      (Core.Pretty.err_result
         ~ok:(fun ppf _ -> Fmt.string ppf "ok")
         ~error:Kernel_elab.pp_error)
      (Kernel_elab.elaborate k u)
  in
  let values =
    List.map (fun (v : Kernel.Value.t) -> v.Kernel.Value.id) k.Kernel.values
  in
  let inputs =
    List.map (fun (i : Kernel.Input.t) -> i.Kernel.Input.id) k.Kernel.inputs
  in
  show "real edge" (Kernel.Use.Set.choose (Kernel.load_uses k));
  (* A boundary input as producer: a real ordinary load, but there is no
     producer body or result conversion to substitute. This is the case that
     makes [Use] value-to-value. *)
  show "boundary input as producer"
    { Kernel.Use.producer = List.hd inputs; consumer = List.nth values 1 };
  show "unknown id"
    { Kernel.Use.producer = Tensor_id.of_int 99; consumer = List.hd values };
  show "not a dependency"
    { Kernel.Use.producer = List.nth values 1; consumer = List.hd values };
  [%expect
    {|
    real edge: ok
    boundary input as producer: t0->t5 does not name a value at both ends
    unknown id: t99->t4 does not name a value at both ends
    not a dependency: t5->t4 is not an ordinary-load edge |}]

(* ---- planning is bounded by the IR limits ----------------------------------

   A valid kernel may hold thousands of values with bodies near the body budget.
   Recomputing load counts, intrinsic witnesses, candidate sites and the site
   check itself per candidate made the planner's work multiplicative in AST size
   and edge count, so the accepted IR limits gave no practical time bound at
   all. All of that is now collected in one pass over the bodies, and a
   validated site carries its resolved endpoints into elaboration. The
   freshening, substitution and budget check per attempted elaboration remain —
   that is the work fusion is for.

   This is a COMPLETION CHECK. It does not fail if the precomputation is
   reverted — it only gets slower — so it is not by itself evidence that the
   one-pass table exists. What it does guard is that planning a wide kernel
   works at all, which caught three fixtures that were green while testing
   nothing.

   The complexity claim rests on the source being one pass, and on a measured
   comparison recorded in .ai/native_kernel_impl_plan.md: at 3072 values,
   reverting [load_count] alone takes the suite from 0.8s to 4.4s. Turning that
   into an assertion would need either a wall-clock threshold, which is flaky
   across machines, or traversal counters, which would mean test-only API. *)

let%expect_test "Fusion: planning a wide kernel completes" =
  let n = 512 in
  let sg id =
    Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape:(s1c 4)
      ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let load id =
    Expr.Value.load
      (Expr_bridge.source_of_id (Tensor_id.of_int id))
      (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
  in
  (* n/2 INDEPENDENT producer/consumer pairs. Two earlier shapes tested nothing:
     values all reading the boundary input yield no [Use] at all, and values all
     reading one producer are rejected at the second check, short-circuiting
     before any expensive work. Disjoint pairs make every candidate legal, so
     each reaches elaboration and the planner does its full job n/2 times. *)
  (* The producer is read exactly ONCE; the bulk of the body is padding read
     from the boundary input. Loading [src] repeatedly would make every edge
     multi-site and be rejected before elaboration, testing rather less. *)
  let body ~src =
    let e = ref (load src) in
    for _ = 1 to 20 do
      e := Expr.Value.add !e (Expr.Value.mul (load 0) (Expr.Value.const 2.0))
    done;
    !e
  in
  (* Raised [max_outputs]: every value here is an output, and the default 1024
     would reject the kernel outright — which auto-promote would then record as
     an expected exception, leaving a test that plans nothing. *)
  let limits =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:4096 ~max_depth:128 ~max_values:4095
         ~max_dep_depth:1024 ~max_inputs:1024 ~max_outputs:4095
         ~max_extent:0x7FFF_FFFFL ~max_numel:0x7FFF_FFFFL ~max_local_slots:8192
         ~max_scan_state:8192 ~max_scan_updates_per_key:8192L
         ~max_scan_updates_total:16_000_000L)
  in
  let k =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create ~limits
         ~inputs:
           [
             {
               Kernel.Input.id = Tensor_id.of_int 0;
               sg = sg 0;
               binding = Kernel.Binding.Caller;
             };
           ]
         ~values:
           (List.init n (fun i ->
                {
                  Kernel.Value.id = Tensor_id.of_int (i + 1);
                  sg = sg (i + 1);
                  (* even i: producer, reads the input.
                     odd  i: consumer, reads the producer before it. *)
                  computation =
                    Region_program.pixel
                      (body ~src:(if i mod 2 = 0 then 0 else i));
                  result = Kernel.Result_conversion.Round_f32;
                }))
         ~outputs:(List.init n (fun i -> Tensor_id.of_int (i + 1)))
         ())
  in
  let p, decisions = Fusion_plan.plan k in
  Format.printf "values: %d, decisions: %d, virtual: %d, stores: %d@."
    (List.length k.Kernel.values)
    (List.length decisions)
    (Kernel.Use.Set.cardinal p.Fusion_plan.virtual_uses)
    (Tensor_id.Set.cardinal p.Fusion_plan.stores);
  [%expect {| values: 512, decisions: 256, virtual: 256, stores: 512 |}]

let%expect_test "Fusion: a Site.t cannot be forged from outside" =
  (* [Site.t] is evidence that the named producer occurs exactly once in the
     named consumer at an admissible coordinate. An earlier fast path took the
     extracted load list as an ARGUMENT and treated it as that proof, so one
     fabricated pointwise entry bought a site for a pair that is not a
     dependency at all — and [elaborate_site] then rewrote the consumer's real
     body, substituting nothing and reporting the unchanged body as a successful
     elaboration.

     The evidence is now owned by [Analysis], reachable only from a [Kernel.t],
     so the only way to obtain a site is to have one.

     The two entry points extract INDEPENDENTLY — [site] walks the named
     consumer's body, [site_in] reads a table built over every consumer at once
     — and share only the small [admit] predicate. That independence is what
     makes comparing them worth anything: while [site] was implemented as
     [site_in (Analysis.of_kernel k)] this compared an expression with itself
     and could not have failed. *)
  let g = Graph_fixtures.conv_add () in
  let k = kernel_of g in
  let a = Kernel_elab.Analysis.of_kernel k in
  let ids =
    List.map (fun (v : Kernel.Value.t) -> v.Kernel.Value.id) k.Kernel.values
  in
  let verdict r = match r with Ok _ -> "accepted" | Error _ -> "rejected" in
  let case name u =
    Format.printf "%s: site=%s site_in=%s@." name
      (verdict (Kernel_elab.site k u))
      (verdict (Kernel_elab.site_in a u))
  in
  let conv = List.nth ids 0 and add = List.nth ids 1 in
  case "real edge conv->add" { Kernel.Use.producer = conv; consumer = add };
  (* The reversed pair is not a dependency. This is the exact edge the forged
     load list used to get past. *)
  case "reversed non-edge" { Kernel.Use.producer = add; consumer = conv };
  case "self edge" { Kernel.Use.producer = conv; consumer = conv };
  (* The two nontrivial occurrence summaries, which real/non-edge/self never
     reach: a producer read TWICE, and one read at a non-pointwise coordinate.
     Both extractions must agree on these too. *)
  let vsg id shape =
    Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape
      ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let ld id =
    Expr.Value.load
      (Expr_bridge.source_of_id (Tensor_id.of_int id))
      (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
  in
  let mk ~consumer_shape ~producer_shape body =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create
         ~inputs:
           [
             {
               Kernel.Input.id = Tensor_id.of_int 0;
               sg = vsg 0 producer_shape;
               binding = Kernel.Binding.Caller;
             };
           ]
         ~values:
           [
             {
               Kernel.Value.id = Tensor_id.of_int 1;
               sg = vsg 1 producer_shape;
               computation = Region_program.pixel (ld 0);
               result = Kernel.Result_conversion.Round_f32;
             };
             {
               Kernel.Value.id = Tensor_id.of_int 2;
               sg = vsg 2 consumer_shape;
               computation = Region_program.pixel body;
               result = Kernel.Result_conversion.Round_f32;
             };
           ]
         ~outputs:[ Tensor_id.of_int 2 ]
         ())
  in
  let edge =
    { Kernel.Use.producer = Tensor_id.of_int 1; consumer = Tensor_id.of_int 2 }
  in
  let compare_on name kk =
    Format.printf "%s: site=%s site_in=%s@." name
      (verdict (Kernel_elab.site kk edge))
      (verdict (Kernel_elab.site_in (Kernel_elab.Analysis.of_kernel kk) edge))
  in
  compare_on "producer read twice"
    (mk ~consumer_shape:(s1c 4) ~producer_shape:(s1c 4)
       (Expr.Value.add (ld 1) (ld 1)));
  (* Producer extent 1 under a consumer of extent 4, read at plain Output C:
     single-site, but not shape-compatible. *)
  compare_on "non-pointwise coordinate"
    (mk ~consumer_shape:(s1c 4) ~producer_shape:(s1c 1) (ld 1));
  [%expect
    {|
    real edge conv->add: site=accepted site_in=accepted
    reversed non-edge: site=rejected site_in=rejected
    self edge: site=rejected site_in=rejected
    producer read twice: site=rejected site_in=rejected
    non-pointwise coordinate: site=rejected site_in=rejected |}]
