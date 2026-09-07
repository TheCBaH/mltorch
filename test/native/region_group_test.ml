(* Project step 19 / Section B: [Region_group] is general, LSTM-agnostic
   infrastructure (_ai_/shared_multi_output_impl.md §3), so this pins its
   contract with a small SYNTHETIC two-emitter group -- one emitter shaped
   like LSTM's time-first sequence output (singleton batch on W, a whole
   axis on H), one shaped like its batch-first counterpart (singleton batch
   on H, a whole axis on W) -- rather than exercising it only through LSTM. *)

let env =
  {
    Expr.Eval.Env.load = (fun _ _ -> assert false);
    load_index = (fun _ _ -> assert false);
  }

let max_size = 64
let max_depth = 16
let canonical_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:4 ~c:1

let whole axes =
  match Region_partition.of_whole_axes axes with
  | Ok p -> p
  | Error _ -> assert false

(* Canonical [W] read directly -- the shared local every emitter reads. *)
let batch_value =
  Expr.Value.value_of_index
    (Expr.Index.of_position (Expr.Index.output Expr.Axis.W))

let shared_local () =
  let id, _ =
    Expr.Builder.run_from Expr.Builder.initial Expr.Builder.fresh_local
  in
  Region_local.scalar ~id
    ~value:(Expr.Value.add batch_value (Expr.Value.const 1.))

let emitter_a : Region_group.Emitter.t =
  {
    output_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:4 ~c:1;
    partition = whole [ Expr.Axis.H ];
    key_axes = [ (Expr.Axis.W, Expr.Axis.W) ];
    output = Expr.Value.const 0.;
    (* overwritten per-test below with the real local read *)
  }

let emitter_b : Region_group.Emitter.t =
  {
    output_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:3 ~c:1;
    partition = whole [ Expr.Axis.W ];
    key_axes = [ (Expr.Axis.W, Expr.Axis.H) ];
    output = Expr.Value.const 0.;
  }

let with_output (e : Region_group.Emitter.t) output =
  { e with Region_group.Emitter.output }

let build_group () =
  let local = shared_local () in
  let output = Expr.Value.local local.Region_local.id in
  Region_group.create ~max_size ~max_depth ~canonical_shape ~locals:[ local ]
    ~emitters:[ with_output emitter_a output; with_output emitter_b output ]

let%expect_test
    "region group: construction succeeds and both projections agree with the \
     canonical batch mapping" =
  let group = Err.or_raise ~pp_error:Region_group.pp_error (build_group ()) in
  let project ordinal =
    Err.or_raise ~pp_error:Region_group.pp_error
      (Region_group.project ~max_size ~max_depth group ordinal)
  in
  let program_a = project 0 and program_b = project 1 in
  let tensor_a =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_eval.materialize program_a
         ~output_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:4 ~c:1)
         ~env)
  in
  let tensor_b =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_eval.materialize program_b
         ~output_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:3 ~c:1)
         ~env)
  in
  (* [emitter_a]'s singleton axis is W (identity mapping): the value at any
     [(h, w, 0)] must be [w+1], invariant in [h]. *)
  Fmt.pr "a: h=0 -> %g,%g,%g,%g@."
    (Tensor.read tensor_a (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0))
    (Tensor.read tensor_a (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:1 ~c:0))
    (Tensor.read tensor_a (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:2 ~c:0))
    (Tensor.read tensor_a (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:3 ~c:0));
  Fmt.pr "a: h=2 -> %g,%g,%g,%g@."
    (Tensor.read tensor_a (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:2 ~w:0 ~c:0))
    (Tensor.read tensor_a (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:2 ~w:1 ~c:0))
    (Tensor.read tensor_a (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:2 ~w:2 ~c:0))
    (Tensor.read tensor_a (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:2 ~w:3 ~c:0));
  (* [emitter_b]'s singleton axis is H, mapped from canonical W: the value at
     any [(h, w, 0)] must be [h+1], invariant in [w] -- the swap the whole
     coordinate-mapping machinery exists for. *)
  Fmt.pr "b: w=0 -> %g,%g,%g,%g@."
    (Tensor.read tensor_b (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0))
    (Tensor.read tensor_b (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:1 ~w:0 ~c:0))
    (Tensor.read tensor_b (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:2 ~w:0 ~c:0))
    (Tensor.read tensor_b (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:3 ~w:0 ~c:0));
  Fmt.pr "b: w=2 -> %g,%g,%g,%g@."
    (Tensor.read tensor_b (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:2 ~c:0))
    (Tensor.read tensor_b (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:1 ~w:2 ~c:0))
    (Tensor.read tensor_b (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:2 ~w:2 ~c:0))
    (Tensor.read tensor_b (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:3 ~w:2 ~c:0));
  [%expect
    {|
    a: h=0 -> 1,2,3,4
    a: h=2 -> 1,2,3,4
    b: w=0 -> 1,2,3,4
    b: w=2 -> 1,2,3,4 |}]

let%expect_test "region group: empty emitters is rejected" =
  let local = shared_local () in
  (match
     Region_group.create ~max_size ~max_depth ~canonical_shape ~locals:[ local ]
       ~emitters:[]
   with
  | Error e -> Fmt.pr "%a@." Region_group.pp_error (Err.Error.kind e)
  | Ok _ -> Fmt.pr "unexpectedly accepted@.");
  [%expect {| a region group needs at least one emitter |}]

let%expect_test
    "region group: a shared local reading a non-canonical axis is rejected" =
  let id, _ =
    Expr.Builder.run_from Expr.Builder.initial Expr.Builder.fresh_local
  in
  let non_canonical =
    Expr.Value.value_of_index
      (Expr.Index.of_position (Expr.Index.output Expr.Axis.H))
  in
  let local =
    Region_local.scalar ~id ~value:(Expr.Value.add batch_value non_canonical)
  in
  let output = Expr.Value.local id in
  (match
     Region_group.create ~max_size ~max_depth ~canonical_shape ~locals:[ local ]
       ~emitters:[ with_output emitter_a output; with_output emitter_b output ]
   with
  | Error e -> Fmt.pr "%a@." Region_group.pp_error (Err.Error.kind e)
  | Ok _ -> Fmt.pr "unexpectedly accepted@.");
  [%expect {| local #0 varies over whole axis H |}]

let%expect_test "region group: unknown emitter ordinal is rejected" =
  let group = Err.or_raise ~pp_error:Region_group.pp_error (build_group ()) in
  (match Region_group.project ~max_size ~max_depth group 5 with
  | Error e -> Fmt.pr "%a@." Region_group.pp_error (Err.Error.kind e)
  | Ok _ -> Fmt.pr "unexpectedly accepted@.");
  [%expect {| unknown emitter ordinal 5 |}]

(* One deliberately broken mapping per [mapping_error] case (CLAUDE.md: "if a
   finding says a check is vacuous, prove the check can fail" -- here the
   rejection itself IS the specified behavior, so each case below is the
   proof rather than a temporary revert). *)
let check_mapping ~label (e : Region_group.Emitter.t) =
  let local = shared_local () in
  let output = Expr.Value.local local.Region_local.id in
  match
    Region_group.create ~max_size ~max_depth ~canonical_shape ~locals:[ local ]
      ~emitters:[ with_output e output ]
  with
  | Error err ->
      Fmt.pr "%s: %a@." label Region_group.pp_error (Err.Error.kind err)
  | Ok _ -> Fmt.pr "%s: unexpectedly accepted@." label

let%expect_test "region group: each mapping_error case fires" =
  check_mapping ~label:"duplicate canonical axis"
    {
      emitter_a with
      key_axes = [ (Expr.Axis.W, Expr.Axis.W); (Expr.Axis.W, Expr.Axis.H) ];
    };
  check_mapping ~label:"duplicate target axis"
    {
      emitter_a with
      key_axes = [ (Expr.Axis.W, Expr.Axis.H); (Expr.Axis.N, Expr.Axis.H) ];
    };
  check_mapping ~label:"extent mismatch"
    {
      emitter_a with
      output_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:4 ~c:2;
      key_axes = [ (Expr.Axis.W, Expr.Axis.C) ];
    };
  check_mapping ~label:"target not singleton"
    { emitter_a with key_axes = [ (Expr.Axis.W, Expr.Axis.H) ] };
  check_mapping ~label:"uncovered canonical axis"
    { emitter_a with key_axes = [] };
  check_mapping ~label:"uncovered physical axis"
    {
      emitter_a with
      output_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:1 ~w:4 ~c:1;
      partition = whole [];
      key_axes = [ (Expr.Axis.W, Expr.Axis.W) ];
    };
  [%expect
    {|
    duplicate canonical axis: emitter 0: duplicate canonical axis W
    duplicate target axis: emitter 0: duplicate target axis H
    extent mismatch: emitter 0: canonical axis W and physical axis C do not agree in extent
    target not singleton: emitter 0: target axis H is not singleton in the emitter's partition
    uncovered canonical axis: emitter 0: canonical key axis W is not covered by any key mapping
    uncovered physical axis: emitter 0: physical singleton axis D is not covered by any key mapping |}]

(* Project step 19 / Section D, acceptance matrix §7 item 7 ("Rounding"): a
   shared local's value must stay in WORKING (double) precision until each
   emitter's OWN store boundary -- rounding the shared local itself to f32
   before emitters read it would be a silent correctness change, not an
   optimization. [1.0 +. 2.0**(-25.)] is exactly representable in double
   (2^-25 needs 25 fractional bits, well under the 52-bit mantissa) but rounds
   to EXACTLY [1.0] in f32 (2^-25 is below the f32 ULP near 1.0, 2^-23):
   an emitter reading [local] directly and STORING it therefore reads back
   [1.0] either way (expected -- storage rounding is normal and not what this
   pins), but an emitter reading [local -. 1.0] recovers the small nonzero
   [2^-25] only if the SUBTRACTION happened before any f32 rounding of
   [local] itself -- premature rounding would make it exactly [0.0] instead. *)
let%expect_test
    "region group: a shared local's value survives to each emitter's own store \
     boundary at working precision, not rounded in between" =
  let id, _ =
    Expr.Builder.run_from Expr.Builder.initial Expr.Builder.fresh_local
  in
  let local_value = 1.0 +. (2.0 ** -25.) in
  let local = Region_local.scalar ~id ~value:(Expr.Value.const local_value) in
  let read_local = Expr.Value.local id in
  let read_local_minus_one = Expr.Value.sub read_local (Expr.Value.const 1.0) in
  (* The discriminating counterfactual, built into the SAME group rather than
     a separately-reverted production edit: an emitter that explicitly rounds
     [local] to f32 BEFORE subtracting -- what premature rounding of the
     shared local itself would look like -- must read exactly [0.0], proving
     this test really would fail if [evaluate_locals] rounded the shared
     local's slot value instead of keeping it at working precision. *)
  let read_rounded_local_minus_one =
    Expr.Value.sub (Expr.Value.round_f32 read_local) (Expr.Value.const 1.0)
  in
  let emitter_c =
    { emitter_b with Region_group.Emitter.output = Expr.Value.const 0. }
  in
  let group =
    Err.or_raise ~pp_error:Region_group.pp_error
      (Region_group.create ~max_size ~max_depth ~canonical_shape
         ~locals:[ local ]
         ~emitters:
           [
             with_output emitter_a read_local;
             with_output emitter_b read_local_minus_one;
             with_output emitter_c read_rounded_local_minus_one;
           ])
  in
  let project ordinal =
    Err.or_raise ~pp_error:Region_group.pp_error
      (Region_group.project ~max_size ~max_depth group ordinal)
  in
  let materialize ordinal shape =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_eval.materialize (project ordinal) ~output_shape:shape ~env)
  in
  let tensor_a = materialize 0 emitter_a.Region_group.Emitter.output_shape in
  let tensor_b = materialize 1 emitter_b.Region_group.Emitter.output_shape in
  let tensor_c = materialize 2 emitter_c.Region_group.Emitter.output_shape in
  let origin = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0 in
  Fmt.pr "local (double) = %.9g@." local_value;
  Fmt.pr "emitter a (reads local, stored f32) = %.9g@."
    (Tensor.read tensor_a origin);
  Fmt.pr "emitter b (reads local - 1, stored f32) = %.9g@."
    (Tensor.read tensor_b origin);
  Fmt.pr
    "emitter c (reads round_f32(local) - 1, the premature-rounding \
     counterfactual) = %.9g@."
    (Tensor.read tensor_c origin);
  [%expect
    {|
    local (double) = 1.00000003
    emitter a (reads local, stored f32) = 1
    emitter b (reads local - 1, stored f32) = 2.98023224e-08
    emitter c (reads round_f32(local) - 1, the premature-rounding counterfactual) = 0 |}]

(* [Region_group.Ref.t] (project step 19, Section C's wrapper migration): a
   [Solo] ref behaves exactly like the bare program it wraps; a [Grouped]
   ref's [project]/[sources]/[pixel_expression] delegate to this same
   group/ordinal, and never fabricate a projection for [pp]. *)
let%expect_test
    "region group ref: Solo and Grouped agree with their underlying operations"
    =
  let group = Err.or_raise ~pp_error:Region_group.pp_error (build_group ()) in
  let solo_program =
    Err.or_raise ~pp_error:Region_group.pp_error
      (Region_group.project ~max_size ~max_depth group 0)
  in
  let solo = Region_group.Ref.Solo solo_program in
  let grouped = Region_group.Ref.Grouped (group, 0) in
  let project r =
    Err.or_raise ~pp_error:Region_group.pp_error
      (Region_group.Ref.project ~max_size ~max_depth r)
  in
  assert (
    String.equal
      (Fmt.str "%a" Region_program.pp (project solo))
      (Fmt.str "%a" Region_program.pp (project grouped)));
  assert (Region_group.Ref.check ~max_size ~max_depth solo |> Result.is_ok);
  assert (Region_group.Ref.check ~max_size ~max_depth grouped |> Result.is_ok);
  assert (
    Option.is_some (Region_group.Ref.pixel_expression solo)
    = Option.is_some (Region_program.pixel_expression solo_program));
  assert (Option.is_none (Region_group.Ref.pixel_expression grouped));
  assert (
    Expr.Source.Set.equal
      (Region_group.Ref.sources solo)
      (Region_group.Ref.sources grouped));
  Fmt.pr "solo: %a@." Region_group.Ref.pp solo;
  Fmt.pr "grouped: %a@." Region_group.Ref.pp grouped;
  [%expect
    {|
    solo: region [N=singleton T=singleton D=singleton H=whole W=singleton C=singleton]
      let l0 : scalar = (value_of_index(W) + 1)
      emit l0
    grouped: group emitter 0 of
    canonical [N=whole T=whole D=whole H=whole W=singleton C=whole]
      let l0 : scalar = (value_of_index(W) + 1)
      emitter 0 [N=singleton T=singleton D=singleton H=whole W=singleton C=singleton] = l0
      emitter 1 [N=singleton T=singleton D=singleton H=singleton W=whole C=singleton] = l0 |}]
