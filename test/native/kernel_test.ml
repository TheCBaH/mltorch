(* [Kernel] construction and the [Stage_program] adapter.

   Every error arm gets a test that PRODUCES it. A rule whose test cannot go red
   is not evidence, and several of the rules here guard cases that only a
   hand-built kernel can reach — which is exactly why [Kernel.create] is a
   validating public boundary rather than an adapter-internal helper. *)

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n
let f32 = Payload.Fmt Payload.F32
let tid = Tensor_id.of_int

let sg ?(fmt = f32) ?quant id shape =
  Tensor_sig.create ~id:(tid id) ~name:"" ~shape ~fmt ?quant ()

let pp_kernel = Core.Pretty.err_result ~ok:Kernel.pp ~error:Kernel.pp_error
let pp_adapt = Core.Pretty.err_result ~ok:Kernel.pp ~error:Kernel_adapt.pp_error

let pp_ids ppf ids =
  Fmt.string ppf
    (match ids with
    | [] -> "-"
    | ids ->
        String.concat "," (List.map (Format.asprintf "%a" Tensor_id.pp) ids))

(* A body reading one source at the plain output coordinate. *)
let load id =
  Expr.Value.load
    (Expr_bridge.source_of_id (tid id))
    (Expr_bridge.coord_of_vec6 Symbolic.out_vec)

let value ?(result = Kernel.Result_conversion.Round_f32) id shape body =
  {
    Kernel.Value.id = tid id;
    sg = sg id shape;
    computation = Region_program.pixel body;
    result;
  }

let input ?(binding = Kernel.Binding.Caller) ?fmt ?quant id shape =
  { Kernel.Input.id = tid id; sg = sg ?fmt ?quant id shape; binding }

(* ---- the happy path ------------------------------------------------------- *)

let%expect_test "Kernel: a two-value kernel prints its bindings and rounds" =
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~inputs:
         [
           input 0 (s1c 3); input ~binding:(Kernel.Binding.Filled 0.) 1 (s1c 3);
         ]
       ~values:
         [
           value 2 (s1c 3) (Expr.Value.add (load 0) (load 1));
           value 3 (s1c 3) (Expr.Value.mul (load 2) (load 2));
         ]
       ~outputs:[ tid 3 ]
       ());
  [%expect
    {|
    input t0 : caller
    input t1 : filled 0
    t2 = round_f32((t0[N,T,D,H,W,C] + t1[N,T,D,H,W,C]))
    t3 = round_f32((t2[N,T,D,H,W,C] * t2[N,T,D,H,W,C]))
    outputs: t3 |}]

(* ---- identity and uniqueness ---------------------------------------------- *)

let%expect_test "Kernel: a record id must agree with its signature id" =
  let bad =
    { (value 2 (s1c 3) (load 0)) with Kernel.Value.sg = sg 9 (s1c 3) }
  in
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~inputs:[ input 0 (s1c 3) ]
       ~values:[ bad ]
       ~outputs:[ tid 2 ]
       ());
  [%expect {| id t2 disagrees with its signature id t9 |}]

let%expect_test "Kernel: ids are unique across inputs and values" =
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~inputs:[ input 0 (s1c 3) ]
       ~values:[ value 0 (s1c 3) (load 0) ]
       ~outputs:[ tid 0 ]
       ());
  [%expect {| duplicate id t0 |}]

(* ---- source resolution ---------------------------------------------------- *)

let%expect_test "Kernel: a source resolving to nothing is unresolved" =
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~inputs:[ input 0 (s1c 3) ]
       ~values:[ value 2 (s1c 3) (load 7) ]
       ~outputs:[ tid 2 ]
       ());
  [%expect {| t2 loads unresolved source t7 |}]

let%expect_test "Kernel: a source resolving to a later value is forward" =
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~inputs:[ input 0 (s1c 3) ]
       ~values:[ value 2 (s1c 3) (load 3); value 3 (s1c 3) (load 0) ]
       ~outputs:[ tid 2 ]
       ());
  [%expect {| t2 depends on later value t3 |}]

(* A max-pool body carries NO ordinary load — its dependency lives only in the
   intrinsic descriptor. Validation reads every SOURCE, so an unknown one is
   caught here too; a loads-only rule would let it through and the evaluator
   would then request a source construction never validated. *)
let pool_body src =
  Expr.Value.intrinsic
    (Err.or_raise ~pp_error:Expr.Intrinsic.pp_error
       (Expr.Intrinsic.max_pool
          ~source:(Expr_bridge.source_of_id (tid src))
          ~in_h:4 ~in_w:4 ~kernel_h:2 ~kernel_w:2 ~stride_h:2 ~stride_w:2
          ~pad_h:0 ~pad_w:0
          ~out:(Expr_bridge.coord_of_vec6 Symbolic.out_vec)
          ~result:Expr.Intrinsic.Max_pool.Value))

let%expect_test "Kernel: an intrinsic-only source is validated like any other" =
  let unknown =
    Kernel.create
      ~inputs:[ input 0 (s 1 1 1 4 4 2) ]
      ~values:[ value 2 (s 1 1 1 2 2 2) (pool_body 7) ]
      ~outputs:[ tid 2 ]
      ()
  in
  let forward =
    Kernel.create
      ~inputs:[ input 0 (s 1 1 1 4 4 2) ]
      ~values:
        [
          value 2 (s 1 1 1 2 2 2) (pool_body 3);
          value 3 (s 1 1 1 4 4 2) (load 0);
        ]
      ~outputs:[ tid 2 ]
      ()
  in
  Format.printf "unknown: %a@.forward: %a@." pp_kernel unknown pp_kernel forward;
  [%expect
    {|
    unknown: t2 loads unresolved source t7
    forward: t2 depends on later value t3 |}]

(* ---- outputs and reachability --------------------------------------------- *)

let%expect_test "Kernel: an output must name a value" =
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~inputs:[ input 0 (s1c 3) ]
       ~values:[ value 2 (s1c 3) (load 0) ]
       ~outputs:[ tid 0 ]
       ());
  [%expect {| output t0 names no value |}]

let%expect_test "Kernel: every value must be reachable from an output" =
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~inputs:[ input 0 (s1c 3) ]
       ~values:[ value 2 (s1c 3) (load 0); value 3 (s1c 3) (load 0) ]
       ~outputs:[ tid 2 ]
       ());
  [%expect {| value t3 is not reachable from any output |}]

(* ---- signature contracts --------------------------------------------------- *)

let%expect_test "Kernel: a stored value must be f32 and unquantized" =
  (* [Tensor.materialize] only ever produces f32/unquantized, so a value
     declaring otherwise would be handed a tensor that does not match what all
     analysis says it is. Boundary inputs carry real data and stay free. *)
  let stored_f16 =
    Kernel.create
      ~inputs:[ input 0 (s1c 3) ]
      ~values:
        [
          {
            (value 2 (s1c 3) (load 0)) with
            Kernel.Value.sg = sg ~fmt:(Payload.Fmt Payload.F16) 2 (s1c 3);
          };
        ]
      ~outputs:[ tid 2 ]
      ()
  in
  let filled_f16 =
    Kernel.create
      ~inputs:
        [
          input ~binding:(Kernel.Binding.Filled 0.)
            ~fmt:(Payload.Fmt Payload.F16) 0 (s1c 3);
        ]
      ~values:[ value 2 (s1c 3) (load 0) ]
      ~outputs:[ tid 2 ]
      ()
  in
  let caller_f16 =
    Kernel.create
      ~inputs:[ input ~fmt:(Payload.Fmt Payload.F16) 0 (s1c 3) ]
      ~values:[ value 2 (s1c 3) (load 0) ]
      ~outputs:[ tid 2 ]
      ()
  in
  Format.printf "stored f16: %a@.filled f16: %a@.caller f16: %a@." pp_kernel
    stored_f16 pp_kernel filled_f16 pp_kernel caller_f16;
  [%expect
    {|
    stored f16: t2: a stored value must be f32 and unquantized, got f16
    filled f16: t0: a filled input must be f32 and unquantized, got f16
    caller f16: input t0 : caller
                t2 = round_f32(t0[N,T,D,H,W,C])
                outputs: t2 |}]

let%expect_test "Kernel: quantization must match the format and the C extent" =
  let q n =
    Err.or_raise ~pp_error:Quant.pp_error
      (Quant.per_channel ~scale:(Array.make n 0.5) ~zero_point:(Array.make n 0))
  in
  let i8 = Payload.Fmt Payload.I8 in
  let case name inp =
    Format.printf "%s: %a@." name pp_kernel
      (Kernel.create ~inputs:[ inp ]
         ~values:[ value 2 (s1c 3) (load 0) ]
         ~outputs:[ tid 2 ]
         ())
  in
  case "quantized fmt, no quant" (input ~fmt:i8 0 (s1c 3));
  case "f32 fmt, with quant" (input ~quant:(q 3) 0 (s1c 3));
  (* The mutation-sensitive one for [Quant.channel_count]: the arrays agree with
     each other — [per_channel] would reject them otherwise — but not with C. *)
  case "per-channel length C-1" (input ~fmt:i8 ~quant:(q 2) 0 (s1c 3));
  case "per-channel length C" (input ~fmt:i8 ~quant:(q 3) 0 (s1c 3));
  [%expect
    {|
    quantized fmt, no quant: t0: quantization disagrees with its format or channel extent
    f32 fmt, with quant: t0: quantization disagrees with its format or channel extent
    per-channel length C-1: t0: quantization disagrees with its format or channel extent
    per-channel length C: input t0 : caller
                          t2 = round_f32(t0[N,T,D,H,W,C])
                          outputs: t2 |}]

(* ---- resource limits ------------------------------------------------------- *)

let small ~max_values ~max_inputs ~max_outputs =
  Err.or_raise ~pp_error:Kernel.Limits.pp_error
    (Kernel.Limits.create ~max_size:4096 ~max_depth:128 ~max_values
       ~max_dep_depth:1024 ~max_inputs ~max_outputs ~max_extent:0x7FFF_FFFFL
       ~max_numel:0x7FFF_FFFFL)

let%expect_test "Kernel.Limits: custom limits may tighten, never widen" =
  let case name r =
    Format.printf "%s: %a@." name
      (Core.Pretty.err_result
         ~ok:(fun ppf _ -> Fmt.string ppf "ok")
         ~error:Kernel.Limits.pp_error)
      r
  in
  let mk ?(max_depth = 128) ?(max_numel = 0x7FFF_FFFFL) () =
    Kernel.Limits.create ~max_size:4096 ~max_depth ~max_values:16
      ~max_dep_depth:16 ~max_inputs:16 ~max_outputs:16 ~max_extent:0x7FFF_FFFFL
      ~max_numel
  in
  case "in range" (mk ());
  (* [Expr.Check] is itself recursive and bounded by the CONFIGURED depth, so an
     over-wide one lets validation overflow before reaching the limit it was
     asked to enforce. *)
  case "depth above Hard" (mk ~max_depth:100000 ());
  (* A numel limit at Int64.max_int would let a shape whose product is 2^32 pass
     the checked fold, after which [Tensor.materialize] recomputes numel in host
     [int] and wraps under js_of_ocaml. *)
  case "numel near Int64.max_int" (mk ~max_numel:Int64.max_int ());
  case "non-positive" (mk ~max_depth:0 ());
  [%expect
    {|
    in range: ok
    depth above Hard: invalid limit max_depth = 100000
    numel near Int64.max_int: invalid limit max_numel = 9223372036854775807
    non-positive: invalid limit max_depth = 0 |}]

let%expect_test "Kernel: interface arity is bounded" =
  let many_inputs =
    Kernel.create
      ~limits:(small ~max_values:16 ~max_inputs:2 ~max_outputs:16)
      ~inputs:[ input 0 (s1c 1); input 1 (s1c 1); input 3 (s1c 1) ]
      ~values:[ value 2 (s1c 1) (load 0) ]
      ~outputs:[ tid 2 ]
      ()
  in
  (* Repeated occurrences of ONE value: a set-based count would pass this. *)
  let many_outputs =
    Kernel.create
      ~limits:(small ~max_values:16 ~max_inputs:16 ~max_outputs:2)
      ~inputs:[ input 0 (s1c 1) ]
      ~values:[ value 2 (s1c 1) (load 0) ]
      ~outputs:[ tid 2; tid 2; tid 2 ]
      ()
  in
  let many_values =
    Kernel.create
      ~limits:(small ~max_values:1 ~max_inputs:16 ~max_outputs:16)
      ~inputs:[ input 0 (s1c 1) ]
      ~values:[ value 2 (s1c 1) (load 0); value 3 (s1c 1) (load 2) ]
      ~outputs:[ tid 3 ]
      ()
  in
  Format.printf "inputs: %a@.outputs: %a@.values: %a@." pp_kernel many_inputs
    pp_kernel many_outputs pp_kernel many_values;
  [%expect
    {|
    inputs: more than 2 inputs
    outputs: more than 2 outputs
    values: more than 1 values |}]

let%expect_test "Kernel: the numel guard runs before the product is formed" =
  let case ?limits name shape =
    Format.printf "%s: %a@." name pp_kernel
      (Kernel.create ?limits
         ~inputs:[ input 0 shape ]
         ~values:[ value 2 (s1c 1) (load 0) ]
         ~outputs:[ tid 2 ]
         ())
  in
  (* Each factor is individually in range and the mathematical product is about
     2^90 — far past Int64.max_int, so a fold that multiplied first and compared
     afterwards would compare a wrapped value. *)
  let big = 1 lsl 15 in
  case "product exceeds Int64.max_int" (s big big big big big big);
  (* Factors in range, product exactly 2^31, reachable even at the largest
     permitted custom limit. *)
  case "product just past 2^31" (s 1 1 1 (1 lsl 16) (1 lsl 15) 1);
  (* The per-extent guard goes through a TIGHTENED limit rather than an extent
     near 2^31. Building one directly is not portable: js_of_ocaml's [int] is 32
     bits, so [1 lsl 31] is [min_int] there and [Dim.extent] raises before any
     bound is consulted. That boundary is unconstructible on that backend by
     definition — which is the whole reason the runtime domain stops below it. *)
  let tight =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:4096 ~max_depth:128 ~max_values:16
         ~max_dep_depth:16 ~max_inputs:16 ~max_outputs:16 ~max_extent:1000L
         ~max_numel:0x7FFF_FFFFL)
  in
  case ~limits:tight "extent over a tightened limit" (s1c 2000);
  [%expect
    {|
    product exceeds Int64.max_int: t0 element count exceeds the limit
    product just past 2^31: t0 element count exceeds the limit
    extent over a tightened limit: t0 extent C=2000 exceeds the limit |}]

let%expect_test "Kernel: budgets are checked before any unmetered traversal" =
  (* Deep enough to blow the stack in [Fold.sources], which [create] would run
     first if the order were reversed. It must report the LIMIT instead. *)
  let deep =
    let e = ref (load 0) in
    for _ = 1 to 5000 do
      e := Expr.Value.add !e (Expr.Value.const 1.0)
    done;
    !e
  in
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~inputs:[ input 0 (s1c 1) ]
       ~values:[ value 2 (s1c 1) deep ]
       ~outputs:[ tid 2 ]
       ());
  [%expect {| t2: depth exceeds limit 128 |}]

(* ---- the adapter ---------------------------------------------------------- *)

let adapt ?limits ?select ?outputs p =
  Kernel_adapt.of_stage_program ?limits ?select ?outputs p

let%expect_test "Kernel_adapt: every fixture adapts to a validated kernel" =
  (* The sweep omits [~outputs], taking the derived required list.
     [multi_output] is the graph that distinguishes it from
     [Stage_program.outputs]: its max-pool index stage is routed to [Discard],
     so nothing consumes it, and a rule keeping only graph outputs would leave
     it unreachable.

     Two fixtures are REJECTED, and correctly. [Graph_builder.new_edge] gives
     every op output [~fmt:f32], so a stage cannot declare another format
     through the builder at all; these two call [Graph_fixtures.with_fmt], which
     patches a format afterwards to reach a state the builder cannot, in order
     to exercise a permute-compatibility branch elsewhere. That patched
     signature is exactly the mismatch the stored-value rule exists to catch —
     [Schedule.ground] would materialise f32 for it while every analysis read
     f16 — so their appearance here is the rule firing on real input rather than
     only on a hand-built case. *)
  let failures =
    List.filter_map
      (fun (name, g) ->
        let p = Eval_symbolic.run (g ()) in
        match adapt p with
        | Ok _ -> None
        | Error e ->
            Some
              (Format.asprintf "%s: %a" name
                 (Core.Pretty.error_kind Kernel_adapt.pp_error)
                 e))
      Graph_fixtures.all
  in
  Format.printf "@[<v>fixtures: %d@,rejected: %d@,%a@]@."
    (List.length Graph_fixtures.all)
    (List.length failures)
    (Fmt.list ~sep:Fmt.cut Fmt.string)
    failures;
  [%expect
    {|
    fixtures: 45
    rejected: 2
    bypass_permute_mixed_compatibility: t3: a stored value must be f32 and unquantized, got f16
    reuse_permute_backtrack_candidate: t2: a stored value must be f32 and unquantized, got f16 |}]

let%expect_test "Kernel_adapt: multi_output promotes its discarded terminal" =
  let p = Eval_symbolic.run (Graph_fixtures.multi_output ()) in
  Format.printf "@[<v>program outputs: %a@,required:        %a@,%a@]@." pp_ids
    p.Stage_program.outputs pp_ids
    (Err.or_raise ~pp_error:Kernel_adapt.pp_error
       (Kernel_adapt.required_outputs p))
    pp_adapt (adapt p);
  [%expect
    {|
    program outputs: t3
    required:        t3,t2
    input t0 : caller
    t1 = round_f32(max_pool2d_value(t0; k=2x2 s=2x2 p=0x0; out=[N,T,D,H,W,C]))
    t2 = round_f32(max_pool2d_index(t0; k=2x2 s=2x2 p=0x0; out=[N,T,D,H,W,C]))
    t3 = round_f32(select((t1[N,T,D,H,W,C] < 0), 0, t1[N,T,D,H,W,C]))
    outputs: t3, t2 |}]

let%expect_test "Kernel_adapt: a pass-through graph output is rejected" =
  (* An input used directly as a graph output is legal today
     (graph_view_test.ml). It has no stage to name, so it must be REJECTED
     rather than filtered — silently returning a kernel that computes fewer
     outputs than its source program is the failure nothing downstream catches. *)
  let g =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      (Graph_builder.build ~name:"passthrough"
         ~outputs:(fun o -> [ o ])
         (Graph_builder.input ~shape:(s1c 4) ()))
  in
  Format.printf "%a@." pp_adapt (adapt (Eval_symbolic.run g));
  [%expect {| graph output t0 is a boundary input, not a stage result |}]

(* Two independent branches over separate inputs, so a selection can be shown
   not to inherit the other branch's boundary. *)
let branches () =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name:"branches" ~outputs:(fun (l, r) -> [ l; r ])
      @@
      let* a = input ~shape:(s1c 3) () in
      let* b = input ~shape:(s1c 3) () in
      let* l = relu a in
      let* r = relu b in
      return (l, r))

let%expect_test "Kernel_adapt: a selection derives its own inputs" =
  let p = Eval_symbolic.run (branches ()) in
  let left =
    match p.Stage_program.stages with
    | st :: _ -> Tensor_id.Set.singleton st.Stage_program.Stage.id
    | [] -> Tensor_id.Set.empty
  in
  Format.printf "whole program:@.%a@." pp_adapt (adapt p);
  (* The other branch's input must be absent: the evaluator validates every
     declared input before computing anything, so inheriting it would turn into
     a spurious [Unbound_input]. *)
  Format.printf "left branch only:@.%a@." pp_adapt (adapt ~select:left p);
  [%expect
    {|
    whole program:
    input t0 : caller
    input t1 : caller
    t2 = round_f32(select((t0[N,T,D,H,W,C] < 0), 0, t0[N,T,D,H,W,C]))
    t3 = round_f32(select((t1[N,T,D,H,W,C] < 0), 0, t1[N,T,D,H,W,C]))
    outputs: t2, t3
    left branch only:
    input t0 : caller
    t2 = round_f32(select((t0[N,T,D,H,W,C] < 0), 0, t0[N,T,D,H,W,C]))
    outputs: t2 |}]

let%expect_test "Kernel_adapt: a selection naming no stage is rejected" =
  let p = Eval_symbolic.run (branches ()) in
  Format.printf "%a@." pp_adapt
    (adapt ~select:(Tensor_id.Set.singleton (tid 99)) p);
  [%expect {| selection names t99, which is not a stage |}]

(* A producer feeding a max-pool, whose dependency exists ONLY inside the
   intrinsic descriptor. Both selection boundaries below are invisible to a
   loads-only rule. *)
let pool_chain () =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name:"pool_chain" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 4 4 2) () in
      let* r = relu x in
      let params : Pool.MaxPool2dWithIndices.params =
        {
          ceil_mode = false;
          kernel = { h = Dim.extent 2; w = Dim.extent 2 };
          stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
          pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
        }
      in
      let* v, i = max_pool2d_with_indices params r in
      let* () = discard i in
      return v)

let stage_ids (p : Stage_program.t) =
  List.map
    (fun (st : Stage_program.Stage.t) -> st.Stage_program.Stage.id)
    p.Stage_program.stages

let%expect_test
    "Kernel_adapt: selection boundaries read every source, not loads" =
  let p = Eval_symbolic.run (pool_chain ()) in
  let ids = stage_ids p in
  let producer = List.nth ids 0 in
  let consumers = List.filteri (fun i _ -> i > 0) ids in
  (* Consumer side selected, producer outside: the producer must become a
     synthetic Caller input even though no [Load] names it. *)
  Format.printf "@[<v>stages: %a@,@,intrinsic consumers selected:@,%a@]@."
    pp_ids ids pp_adapt
    (adapt
       ~select:
         (List.fold_left
            (fun s id -> Tensor_id.Set.add id s)
            Tensor_id.Set.empty consumers)
       p);
  (* Producer selected, its max-pool consumers outside: the producer is
     externally live and must be required as an output. Passing an output list
     without it must fail. *)
  let sel = Tensor_id.Set.singleton producer in
  Format.printf "@[<v>producer selected, required: %a@,with no outputs: %a@]@."
    pp_ids
    (Err.or_raise ~pp_error:Kernel_adapt.pp_error
       (Kernel_adapt.required_outputs ~select:sel p))
    pp_adapt
    (adapt ~select:sel ~outputs:[] p);
  [%expect
    {|
    stages: t1,t2,t3

    intrinsic consumers selected:
    input t1 : caller
    t2 = round_f32(max_pool2d_value(t1; k=2x2 s=2x2 p=0x0; out=[N,T,D,H,W,C]))
    t3 = round_f32(max_pool2d_index(t1; k=2x2 s=2x2 p=0x0; out=[N,T,D,H,W,C]))
    outputs: t2, t3
    producer selected, required: t1
    with no outputs: outputs must begin with the required list; expected t1 next |}]

let%expect_test "Kernel_adapt: caller outputs must begin with the required list"
    =
  let p = Eval_symbolic.run (branches ()) in
  let req =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error
      (Kernel_adapt.required_outputs p)
  in
  let show name outputs =
    Format.printf "%s: %a@." name
      (Core.Pretty.err_result
         ~ok:(fun ppf (k : Kernel.t) ->
           pp_ids ppf
             (List.map
                (fun (o : Kernel.Output.t) -> o.Kernel.Output.value)
                k.Kernel.outputs))
         ~error:Kernel_adapt.pp_error)
      (adapt ~outputs p)
  in
  Format.printf "required: %a@." pp_ids req;
  show "exact" req;
  show "reordered" (List.rev req);
  show "one dropped" (match req with _ :: tl -> tl | [] -> []);
  show "extra appended" (req @ [ List.hd req ]);
  [%expect
    {|
    required: t2,t3
    exact: t2,t3
    reordered: outputs must begin with the required list; expected t2 next
    one dropped: outputs must begin with the required list; expected t2 next
    extra appended: t2,t3,t2 |}]

let%expect_test "Kernel_adapt: a selected forward source is caught" =
  (* Hand-built: the whole-program path would be caught downstream by
     [Kernel.create], so only a SELECTION exercises this hole — an unselected
     later stage becomes a synthetic boundary input, and the forward-reference
     rule never sees it. *)
  let p = Eval_symbolic.run (branches ()) in
  let a, b =
    match stage_ids p with a :: b :: _ -> (a, b) | _ -> (tid 0, tid 1)
  in
  let broken =
    {
      p with
      Stage_program.stages =
        List.map
          (fun (st : Stage_program.Stage.t) ->
            if Tensor_id.equal st.id a then
              {
                st with
                Stage_program.Stage.computation =
                  Region_program.with_output st.computation
                    (load (Tensor_id.to_int b));
              }
            else st)
          p.Stage_program.stages;
    }
  in
  Format.printf "whole program: %a@.selection only: %a@." pp_adapt
    (adapt broken) pp_adapt
    (adapt ~select:(Tensor_id.Set.singleton a) broken);
  [%expect
    {|
    whole program: stage program invalid at t2: reads a later stage
    selection only: stage program invalid at t2: reads a later stage |}]

let%expect_test "Kernel_adapt: an oversized body is caught in both entries" =
  let p = Eval_symbolic.run (branches ()) in
  let a = List.hd (stage_ids p) in
  let deep =
    let e = ref (Expr.Value.const 1.0) in
    for _ = 1 to 5000 do
      e := Expr.Value.add !e (Expr.Value.const 1.0)
    done;
    !e
  in
  let broken =
    {
      p with
      Stage_program.stages =
        List.map
          (fun (st : Stage_program.Stage.t) ->
            if Tensor_id.equal st.id a then
              {
                st with
                Stage_program.Stage.computation =
                  Region_program.with_output st.computation deep;
              }
            else st)
          p.Stage_program.stages;
    }
  in
  (* Selecting the OTHER stage still has to fail: liveness scans the unselected
     body, so the preflight must cover it too. *)
  let other = List.nth (stage_ids p) 1 in
  Format.printf "of_stage_program: %a@." pp_adapt (adapt broken);
  Format.printf "required_outputs: %a@."
    (Core.Pretty.err_result ~ok:pp_ids ~error:Kernel_adapt.pp_error)
    (Kernel_adapt.required_outputs broken);
  Format.printf "unselected body: %a@." pp_adapt
    (adapt ~select:(Tensor_id.Set.singleton other) broken);
  [%expect
    {|
    of_stage_program: t2: depth exceeds limit 128
    required_outputs: t2: depth exceeds limit 128
    unselected body: t2: depth exceeds limit 128 |}]

(* ---- the untrusted boundary table ------------------------------------------

   [Stage_program.t] is a public record, and the adapter's interface claims the
   whole definition table is untrusted. These are the collisions that claim
   covers; before they were checked, a selection could still launder a malformed
   boundary. *)

let stage_program ?(inputs = []) ?(consts = []) ?(stages = []) ?(outputs = [])
    () =
  {
    Stage_program.inputs;
    input_kinds = Tensor_id.Map.empty;
    consts;
    stages;
    outputs;
  }

let stage id shape body =
  {
    Stage_program.Stage.id = tid id;
    sg = sg id shape;
    computation = Region_program.pixel body;
  }

let%expect_test "Kernel_adapt: the boundary table rejects collisions" =
  let case name p =
    Format.printf "%s: %a@." name
      (Core.Pretty.err_result ~ok:pp_ids ~error:Kernel_adapt.pp_error)
      (Kernel_adapt.required_outputs p)
  in
  let body = load 0 in
  let one_stage = [ stage 2 (s1c 3) body ] in
  case "duplicate input"
    (stage_program
       ~inputs:[ (tid 0, sg 0 (s1c 3)); (tid 0, sg 0 (s1c 3)) ]
       ~stages:one_stage
       ~outputs:[ tid 2 ]
       ());
  case "duplicate const"
    (stage_program
       ~inputs:[ (tid 0, sg 0 (s1c 3)) ]
       ~consts:[ (sg 1 (s1c 3), 0.); (sg 1 (s1c 3), 0.) ]
       ~stages:one_stage
       ~outputs:[ tid 2 ]
       ());
  case "input/const collision"
    (stage_program
       ~inputs:[ (tid 0, sg 0 (s1c 3)) ]
       ~consts:[ (sg 0 (s1c 3), 0.) ]
       ~stages:one_stage
       ~outputs:[ tid 2 ]
       ());
  (* An input tuple whose id disagrees with its signature id: [sg.id] is the
     binding key everywhere else, so the two must not diverge. *)
  case "input signature mismatch"
    (stage_program
       ~inputs:[ (tid 0, sg 9 (s1c 3)) ]
       ~stages:one_stage
       ~outputs:[ tid 2 ]
       ());
  case "well formed"
    (stage_program
       ~inputs:[ (tid 0, sg 0 (s1c 3)) ]
       ~stages:one_stage
       ~outputs:[ tid 2 ]
       ());
  [%expect
    {|
    duplicate input: stage program invalid at t0: defined twice
    duplicate const: stage program invalid at t1: defined twice
    input/const collision: stage program invalid at t0: defined twice
    input signature mismatch: stage program invalid at t0: signature id disagrees
    well formed: t2 |}]

let%expect_test "Kernel_adapt: a declared output must name something" =
  (* An id in NEITHER table was silently skipped, so a malformed program adapted
     cleanly to a kernel with a DIFFERENT public result — t2 promoted as a dead
     terminal while t99 vanished. That is the hazard [Passthrough_output] exists
     to prevent, reached by another route. *)
  let p =
    stage_program
      ~inputs:[ (tid 0, sg 0 (s1c 3)) ]
      ~stages:[ stage 2 (s1c 3) (load 0) ]
      ~outputs:[ tid 99 ]
      ()
  in
  Format.printf "unknown output: %a@." pp_adapt (adapt p);
  Format.printf "required_outputs: %a@."
    (Core.Pretty.err_result ~ok:pp_ids ~error:Kernel_adapt.pp_error)
    (Kernel_adapt.required_outputs p);
  [%expect
    {|
    unknown output: graph output t99 is neither a stage nor a boundary
    required_outputs: graph output t99 is neither a stage nor a boundary |}]

let%expect_test "Kernel_adapt: raw lists are bounded before they are traversed"
    =
  (* Only [stages] used to be bounded, so [inputs], [consts] and [outputs] could
     be arbitrarily large and the adapter would do unbounded work before
     [Kernel.create] noticed — and [required_outputs] never reaches [create] at
     all, so the same limits meant different things per entry point. *)
  let limits =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:4096 ~max_depth:128 ~max_values:16
         ~max_dep_depth:16 ~max_inputs:2 ~max_outputs:1 ~max_extent:0x7FFF_FFFFL
         ~max_numel:0x7FFF_FFFFL)
  in
  let many_inputs =
    stage_program
      ~inputs:(List.init 4 (fun i -> (tid i, sg i (s1c 3))))
      ~stages:[ stage 9 (s1c 3) (load 0) ]
      ~outputs:[ tid 9 ]
      ()
  in
  let many_outputs =
    stage_program
      ~inputs:[ (tid 0, sg 0 (s1c 3)) ]
      ~stages:[ stage 2 (s1c 3) (load 0); stage 3 (s1c 3) (load 0) ]
      ~outputs:[ tid 2; tid 3 ]
      ()
  in
  let show name r =
    Format.printf "%s: %a@." name
      (Core.Pretty.err_result ~ok:pp_ids ~error:Kernel_adapt.pp_error)
      r
  in
  show "raw inputs, required_outputs"
    (Kernel_adapt.required_outputs ~limits many_inputs);
  show "raw outputs, required_outputs"
    (Kernel_adapt.required_outputs ~limits many_outputs);
  Format.printf "raw inputs, of_stage_program: %a@." pp_adapt
    (Kernel_adapt.of_stage_program ~limits many_inputs);
  [%expect
    {|
    raw inputs, required_outputs: more than 2 inputs
    raw outputs, required_outputs: more than 1 outputs
    raw inputs, of_stage_program: more than 2 inputs |}]

let%expect_test "Kernel_adapt: the derived interface is bounded too" =
  (* The raw lists are bounded before traversal; the DERIVED interface is
     bounded separately, because it is not the same set. A selection invents a
     synthetic input per referenced unselected stage, so a program within the
     raw input limit can still exceed it once adapted — and [required_outputs]
     must apply the derived output bound itself, since it never reaches
     [Kernel.create]. *)
  let limits ~max_inputs ~max_outputs =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:4096 ~max_depth:128 ~max_values:16
         ~max_dep_depth:16 ~max_inputs ~max_outputs ~max_extent:0x7FFF_FFFFL
         ~max_numel:0x7FFF_FFFFL)
  in
  (* One raw input, but three stages; selecting only the last turns the two
     unselected producers into synthetic Caller inputs. *)
  let p =
    stage_program
      ~inputs:[ (tid 0, sg 0 (s1c 3)) ]
      ~stages:
        [
          stage 1 (s1c 3) (load 0);
          stage 2 (s1c 3) (load 0);
          stage 3 (s1c 3) (Expr.Value.add (load 1) (load 2));
        ]
      ~outputs:[ tid 3 ]
      ()
  in
  let last = Tensor_id.Set.singleton (tid 3) in
  Format.printf "raw inputs = 1, derived = 2, limit 2: %a@." pp_adapt
    (Kernel_adapt.of_stage_program
       ~limits:(limits ~max_inputs:2 ~max_outputs:8)
       ~select:last p);
  Format.printf "same, limit 1: %a@." pp_adapt
    (Kernel_adapt.of_stage_program
       ~limits:(limits ~max_inputs:1 ~max_outputs:8)
       ~select:last p);
  (* Derived outputs: three dead terminals are promoted, exceeding a limit the
     raw output list (one id) sits comfortably under. *)
  let dead =
    stage_program
      ~inputs:[ (tid 0, sg 0 (s1c 3)) ]
      ~stages:
        [
          stage 1 (s1c 3) (load 0);
          stage 2 (s1c 3) (load 0);
          stage 3 (s1c 3) (load 0);
        ]
      ~outputs:[ tid 1 ]
      ()
  in
  Format.printf "raw outputs = 1, derived = 3, limit 2 (required_outputs): %a@."
    (Core.Pretty.err_result ~ok:pp_ids ~error:Kernel_adapt.pp_error)
    (Kernel_adapt.required_outputs
       ~limits:(limits ~max_inputs:8 ~max_outputs:2)
       dead);
  Format.printf "same via of_stage_program: %a@." pp_adapt
    (Kernel_adapt.of_stage_program
       ~limits:(limits ~max_inputs:8 ~max_outputs:2)
       dead);
  [%expect
    {|
    raw inputs = 1, derived = 2, limit 2: input t1 : caller
                                          input t2 : caller
                                          t3 = round_f32((t1[N,T,D,H,W,C] + t2[N,T,D,H,W,C]))
                                          outputs: t3
    same, limit 1: more than 1 inputs
    raw outputs = 1, derived = 3, limit 2 (required_outputs): more than 2 outputs
    same via of_stage_program: more than 2 outputs |}]

let%expect_test "Kernel.over_limit: bounded cardinality, shared budget" =
  (* The bounded SCAN cannot be distinguished from [List.length] by any test at
     a feasible size, and neither can the absence of the 32-bit sum: both are
     structural properties of the source. What is testable is the contract, and
     in particular that the two-list form shares ONE budget rather than checking
     each list separately — [over_limit 2 a || over_limit 2 b] would answer
     false for two two-element lists, which together exceed it. *)
  let l n = List.init n (fun i -> i) in
  Format.printf "0/2:%b 2/2:%b 3/2:%b@."
    (Kernel.over_limit 2 (l 0))
    (Kernel.over_limit 2 (l 2))
    (Kernel.over_limit 2 (l 3));
  Format.printf "shared 2+0:%b 1+1:%b 2+1:%b 2+2:%b@."
    (Kernel.over_limit_2 2 (l 2) (l 0))
    (Kernel.over_limit_2 2 (l 1) (l 1))
    (Kernel.over_limit_2 2 (l 2) (l 1))
    (Kernel.over_limit_2 2 (l 2) (l 2));
  (* The first list alone over the limit short-circuits without touching the
     second, which is what lets the pair be bounded without a sum. *)
  Format.printf "first alone over: %b@." (Kernel.over_limit_2 1 (l 5) (l 0));
  (* A negative limit is exceeded by everything, the empty list included: zero
     cells is more than a negative count. Production callers only ever pass
     validated positive limits, but a public helper answers on its own terms. *)
  Format.printf "negative limit: empty:%b one:%b pair:%b@."
    (Kernel.over_limit (-1) (l 0))
    (Kernel.over_limit (-1) (l 1))
    (Kernel.over_limit_2 (-1) (l 0) (l 0));
  [%expect
    {|
    0/2:false 2/2:false 3/2:true
    shared 2+0:false 1+1:false 2+1:true 2+2:true
    first alone over: true
    negative limit: empty:true one:true pair:true |}]
