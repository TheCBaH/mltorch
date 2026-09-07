(* Project step 19 / Section B: confirms `Lstm.Lstm.Computation.group`'s
   structural claims directly -- exactly [layers*directions] shared scan
   locals, one group instance per call (never reused/cached), and that its
   three projected emitters agree numerically with materializing each
   output the way `Eval_direct.run` already does through
   `Region_computation.program` (itself now `group`+`Region_group.project`,
   confirmed unchanged bit-for-bit by the untouched lstm_graph_layers_test.ml/
   lstm_graph_batch_first_test.ml suites this same session -- see
   _ai_/project_todo.md step 19's Section B evidence). A stacked (Q=2),
   bidirectional (R=2), batch=2 fixture under EACH [batch_first] value, so
   this exercises exactly the shape the coordinate-contract test
   (test/native/lstm_coordinate_contract_test.ml) pins the axis mapping for. *)

let sig_ id shape =
  Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape
    ~fmt:(Payload.Fmt Payload.F32) ()

let fixture ~batch_first =
  let batch = 2 and seq = 2 and input_size = 2 and hidden_size = 2 in
  let directions = 2 in
  let params : Lstm.Lstm.params = { hidden_size; input_size; batch_first } in
  let seq_shape =
    if batch_first then Vec6.shape ~n:1 ~t:1 ~d:1 ~h:batch ~w:seq ~c:input_size
    else Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:input_size
  in
  let state_shape ~layers =
    Vec6.shape ~n:1 ~t:1 ~d:1 ~h:layers ~w:batch ~c:hidden_size
  in
  let wih_shape ~layer_input_size =
    Vec6.shape ~n:(4 * hidden_size) ~t:1 ~d:1 ~h:1 ~w:1 ~c:layer_input_size
  in
  let whh_shape =
    Vec6.shape ~n:(4 * hidden_size) ~t:1 ~d:1 ~h:1 ~w:1 ~c:hidden_size
  in
  let bias_shape = Vec6.shape ~n:(4 * hidden_size) ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let direction ~id_base ~layer_input_size : Lstm.Lstm.Direction_operands.t =
    {
      weight_ih = sig_ id_base (wih_shape ~layer_input_size);
      weight_hh = sig_ (id_base + 1) whh_shape;
      bias = Some (sig_ (id_base + 2) bias_shape, sig_ (id_base + 3) bias_shape);
    }
  in
  let layer ~q : Lstm.Lstm.Layer_operands.t =
    let layer_input_size =
      if q = 0 then input_size else directions * hidden_size
    in
    {
      forward = direction ~id_base:(q * 100) ~layer_input_size;
      reverse = Some (direction ~id_base:((q * 100) + 10) ~layer_input_size);
    }
  in
  let layers = [ layer ~q:0; layer ~q:1 ] in
  let input = sig_ 900 seq_shape in
  let h0 = sig_ 901 (state_shape ~layers:(2 * directions)) in
  let c0 = sig_ 902 (state_shape ~layers:(2 * directions)) in
  let direction_shapes (d : Lstm.Lstm.Direction_operands.t) :
      Lstm.Lstm.Direction_shapes.t =
    {
      weight_ih = d.weight_ih.Tensor_sig.shape;
      weight_hh = d.weight_hh.Tensor_sig.shape;
      bias =
        Option.map
          (fun (bi, bh) -> (bi.Tensor_sig.shape, bh.Tensor_sig.shape))
          d.bias;
    }
  in
  let layer_shapes (l : Lstm.Lstm.Layer_operands.t) : Lstm.Lstm.Layer_shapes.t =
    {
      forward = direction_shapes l.forward;
      reverse = Option.map direction_shapes l.reverse;
    }
  in
  let out_shape, hn_shape, cn_shape =
    Err.or_raise ~pp_error:Shape_error.pp
      (Lstm.Lstm.output_shape params ~input_shape:seq_shape
         ~layers:(List.map layer_shapes layers)
         ~h0_shape:h0.Tensor_sig.shape ~c0_shape:c0.Tensor_sig.shape)
  in
  (params, layers, input, h0, c0, out_shape, hn_shape, cn_shape)

let build ~batch_first =
  let params, layers, input, h0, c0, out_shape, hn_shape, cn_shape =
    fixture ~batch_first
  in
  Err.or_raise ~pp_error:Region_group.pp_error
    (Lstm.Lstm.Computation.group ~limits:Kernel.Limits.default params ~layers
       ~input ~h0 ~c0 ~out_shape ~hn_shape ~cn_shape)

let%expect_test
    "lstm group: exactly layers*directions shared scan locals, both layouts" =
  List.iter
    (fun batch_first ->
      let g = build ~batch_first in
      let scan_count =
        List.length
          (List.filter
             (fun (l : Region_local.t) ->
               match l.Region_local.rhs with
               | Region_local.Rhs.Scan _ -> true
               | Region_local.Rhs.Scalar _ | Region_local.Rhs.Vector _ -> false)
             (Region_group.locals g))
      in
      Fmt.pr "batch_first=%b: locals=%d scans=%d emitters=%d@." batch_first
        (List.length (Region_group.locals g))
        scan_count
        (List.length (Region_group.emitters g)))
    [ false; true ];
  [%expect
    {|
    batch_first=false: locals=4 scans=4 emitters=3
    batch_first=true: locals=4 scans=4 emitters=3 |}]

(* Every shared local is itself a scan (no auxiliary scalar/vector locals in
   this builder), so [locals=4] IS [layers*directions=2*2]; asserting both
   independently (rather than only [locals=4]) is what pins the "no locals
   besides the shared scans" structural claim, not just a coincidental count
   match. *)

let%expect_test
    "lstm group: two calls with the same config are distinct instances" =
  let a = build ~batch_first:false and b = build ~batch_first:false in
  Fmt.pr "same instance: %b@." (a == b);
  [%expect {| same instance: false |}]

(* Project step 19 / Section C: [Region_execution.lower_group]/
   [materialize_group] actually SHARE the recurrence -- one evaluation of the
   shared locals per canonical key, not once per emitter. Proves both halves
   of that claim together: the produced tensors agree bit-for-bit with
   independently lowering/materializing each output's own [program] (today's
   pre-sharing path, unchanged since Section B), AND [counters.scans] ([keys *
   layers*directions], this fixture's `batch=2,Q=2,R=2` giving 2*2*2=8) is
   exactly a THIRD of summing that same counter across three independent
   per-output [materialize] calls (24) -- confirmed by measuring both, not by
   asserting a single magic number. *)
let%expect_test
    "lstm group: materialize_group agrees with independent per-output \
     materialize, and shares scans (not tripled)" =
  let limits = Kernel.Limits.default in
  let scan_limits = Kernel.Limits.scan_limits limits in
  List.iter
    (fun batch_first ->
      let params, layers, input, h0, c0, out_shape, hn_shape, cn_shape =
        fixture ~batch_first
      in
      let g =
        Err.or_raise ~pp_error:Region_group.pp_error
          (Lstm.Lstm.Computation.group ~limits params ~layers ~input ~h0 ~c0
             ~out_shape ~hn_shape ~cn_shape)
      in
      let lowered_group =
        Err.or_raise ~pp_error:Region_group.pp_error
          (Region_execution.lower_group ~max_size:limits.Kernel.Limits.max_size
             ~max_depth:limits.Kernel.Limits.max_depth
             ~max_local_slots:limits.Kernel.Limits.max_local_slots ~scan_limits
             g)
      in
      let direction_sigs (d : Lstm.Lstm.Direction_operands.t) =
        d.weight_ih :: d.weight_hh
        :: (match d.bias with None -> [] | Some (bi, bh) -> [ bi; bh ])
      in
      let layer_sigs (l : Lstm.Lstm.Layer_operands.t) =
        direction_sigs l.forward
        @ match l.reverse with None -> [] | Some r -> direction_sigs r
      in
      let sigs = input :: h0 :: c0 :: List.concat_map layer_sigs layers in
      let tensor_map =
        List.fold_left
          (fun m (s : Tensor_sig.t) ->
            Tensor_id.Map.add s.Tensor_sig.id
              (Tensor.materialize s.Tensor_sig.shape (fun _ -> 0.01))
              m)
          Tensor_id.Map.empty sigs
      in
      let env =
        Expr_bridge.env ~binding:(fun id ->
            Tensor_id.Map.find_opt id tensor_map)
      in
      let counters = Region_execution.counters () in
      let grouped =
        Err.or_raise ~pp_error:Region_eval.pp_error
          (Region_execution.materialize_group ~counters lowered_group ~env
             ~selected:[ 0; 1; 2 ])
      in
      let shape_of output =
        match output with 0 -> out_shape | 1 -> hn_shape | _ -> cn_shape
      in
      (* Each output's OWN counters, exactly the unshared pre-Section-C path
         (independent [lower_region]/[materialize] per ordinal): summing
         [scans] across all three is what "tripled" meant before this pass,
         and is the number [materialize_group]'s single, shared evaluation
         must come in strictly under. *)
      let independent_counters = Region_execution.counters () in
      let independent output =
        let program =
          Err.or_raise ~pp_error:Region_group.pp_error
            (Lstm.Lstm.Computation.program ~limits params ~output ~layers ~input
               ~h0 ~c0 ~out_shape ~hn_shape ~cn_shape)
        in
        let lowered =
          Err.or_raise ~pp_error:Region_program.pp_error
            (Region_execution.lower_region
               ~max_size:limits.Kernel.Limits.max_size
               ~max_depth:limits.Kernel.Limits.max_depth
               ~max_local_slots:limits.Kernel.Limits.max_local_slots
               ~scan_limits ~output_shape:(shape_of output) program)
        in
        Err.or_raise ~pp_error:Region_eval.pp_error
          (Region_execution.materialize ~counters:independent_counters lowered
             ~env)
      in
      let max_diff = ref 0. in
      List.iter
        (fun (output, tensor) ->
          let expected = independent output in
          Vec6.iter (shape_of output) (fun coord ->
              let a = Tensor.read tensor coord in
              let b = Tensor.read expected coord in
              max_diff := Float.max !max_diff (Float.abs (a -. b))))
        grouped;
      Fmt.pr
        "batch_first=%b: max_abs_diff=%g shared_scans=%d unshared_scans=%d \
         (%dx)@."
        batch_first !max_diff counters.Region_execution.scans
        independent_counters.Region_execution.scans
        (independent_counters.Region_execution.scans
       / counters.Region_execution.scans))
    [ false; true ];
  [%expect
    {|
    batch_first=false: max_abs_diff=0 shared_scans=8 unshared_scans=24 (3x)
    batch_first=true: max_abs_diff=0 shared_scans=8 unshared_scans=24 (3x)
    |}]

(* Project step 19 / Section D, acceptance matrix §7 item 2 ("Selection"):
   [materialize_group] must materialize any requested SUBSET of emitters --
   each singleton, a two-emitter subset, and (design record §4.2/§5.4) an
   empty selection must return no tensors AND perform no recurrence work at
   all, not merely skip the emitter reads. A tiny (non-scaled) fixture is
   enough here: this is a selection-shape claim, not a counter-magnitude one
   (lstm_group_test.ml's own scale assertions above already cover the
   magnitude side for the all-three case). *)
let%expect_test
    "lstm group: materialize_group honors singleton/subset/empty selection" =
  let limits = Kernel.Limits.default in
  let scan_limits = Kernel.Limits.scan_limits limits in
  let params, layers, input, h0, c0, out_shape, hn_shape, cn_shape =
    fixture ~batch_first:false
  in
  let g =
    Err.or_raise ~pp_error:Region_group.pp_error
      (Lstm.Lstm.Computation.group ~limits params ~layers ~input ~h0 ~c0
         ~out_shape ~hn_shape ~cn_shape)
  in
  let lowered_group =
    Err.or_raise ~pp_error:Region_group.pp_error
      (Region_execution.lower_group ~max_size:limits.Kernel.Limits.max_size
         ~max_depth:limits.Kernel.Limits.max_depth
         ~max_local_slots:limits.Kernel.Limits.max_local_slots ~scan_limits g)
  in
  let direction_sigs (d : Lstm.Lstm.Direction_operands.t) =
    d.weight_ih :: d.weight_hh
    :: (match d.bias with None -> [] | Some (bi, bh) -> [ bi; bh ])
  in
  let layer_sigs (l : Lstm.Lstm.Layer_operands.t) =
    direction_sigs l.forward
    @ match l.reverse with None -> [] | Some r -> direction_sigs r
  in
  let sigs = input :: h0 :: c0 :: List.concat_map layer_sigs layers in
  let tensor_map =
    List.fold_left
      (fun m (s : Tensor_sig.t) ->
        Tensor_id.Map.add s.Tensor_sig.id
          (Tensor.materialize s.Tensor_sig.shape (fun _ -> 0.01))
          m)
      Tensor_id.Map.empty sigs
  in
  let env =
    Expr_bridge.env ~binding:(fun id -> Tensor_id.Map.find_opt id tensor_map)
  in
  let run selected =
    let counters = Region_execution.counters () in
    let results =
      Err.or_raise ~pp_error:Region_eval.pp_error
        (Region_execution.materialize_group ~counters lowered_group ~env
           ~selected)
    in
    Fmt.pr
      "selected=%a: ordinals=%a keys=%d scans=%d scan_updates=%d emitters=%d@."
      Fmt.(brackets (list ~sep:comma int))
      selected
      Fmt.(brackets (list ~sep:comma int))
      (List.map fst results) counters.Region_execution.keys
      counters.Region_execution.scans counters.Region_execution.scan_updates
      counters.Region_execution.emitters
  in
  run [ 0 ];
  run [ 1 ];
  run [ 2 ];
  run [ 0; 2 ];
  run [];
  [%expect
    {|
    selected=[0]: ordinals=[0] keys=2 scans=8 scan_updates=64 emitters=16
    selected=[1]: ordinals=[1] keys=2 scans=8 scan_updates=64 emitters=16
    selected=[2]: ordinals=[2] keys=2 scans=8 scan_updates=64 emitters=16
    selected=[0, 2]: ordinals=[0, 2] keys=2 scans=8 scan_updates=64 emitters=32
    selected=[]: ordinals=[] keys=0 scans=0 scan_updates=0 emitters=0 |}]

(* Non-vacuous per CLAUDE.md: the [selected=[]] line above is exactly the
   claim a missing short-circuit would violate. Confirmed by temporarily
   reverting [Region_execution.materialize_group] to run [fold_keys]/
   [evaluate_locals] unconditionally (removing the [match selected with []
   -> [] | _ -> ...] guard this same session added): the empty-selection line
   read [keys=2 scans=8 scan_updates=64 emitters=0] instead -- shared-local
   recurrence work still ran despite requesting nothing -- before the guard
   was restored. *)

let%expect_test
    "lstm group: each projected emitter materializes the same values \
     Eval_direct.run produces through Region_computation .program, both \
     layouts" =
  List.iter
    (fun batch_first ->
      let params, layers, input, h0, c0, out_shape, hn_shape, cn_shape =
        fixture ~batch_first
      in
      let g =
        Err.or_raise ~pp_error:Region_group.pp_error
          (Lstm.Lstm.Computation.group ~limits:Kernel.Limits.default params
             ~layers ~input ~h0 ~c0 ~out_shape ~hn_shape ~cn_shape)
      in
      List.iter
        (fun output ->
          let projected =
            Err.or_raise ~pp_error:Region_group.pp_error
              (Region_group.project ~max_size:Kernel.Limits.default.max_size
                 ~max_depth:Kernel.Limits.default.max_depth g output)
          in
          let via_program =
            Err.or_raise ~pp_error:Region_group.pp_error
              (Lstm.Lstm.Computation.program ~limits:Kernel.Limits.default
                 params ~output ~layers ~input ~h0 ~c0 ~out_shape ~hn_shape
                 ~cn_shape)
          in
          let alpha e = Expr.Rewrite.alpha_normalize e in
          Fmt.pr "batch_first=%b output=%d: alpha-equal=%b@." batch_first output
            (Expr.Value.equal
               (alpha (Region_program.output projected))
               (alpha (Region_program.output via_program))))
        [ 0; 1; 2 ])
    [ false; true ];
  [%expect
    {|
    batch_first=false output=0: alpha-equal=true
    batch_first=false output=1: alpha-equal=true
    batch_first=false output=2: alpha-equal=true
    batch_first=true output=0: alpha-equal=true
    batch_first=true output=1: alpha-equal=true
    batch_first=true output=2: alpha-equal=true |}]
