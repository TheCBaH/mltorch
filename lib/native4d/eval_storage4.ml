(* The storage script of a Native4D run, split from [Eval_direct4] under the
   tracked file-size ceiling (scripts/check-file-size.sh): it is that module's
   own fold ([walk], [node_shapes], [node_outs], [alloc_of], [released_ids]) with
   the used constants and graph inputs added and the run's boundaries marked, so
   it opens that module rather than restating them. *)

open Eval_direct4

(* The twin of [Eval_direct.storage_script]: every block a run under [config]
   touches, by role and arena, and when it is allocated and freed. [dry_run]'s
   fold, with the used constants and graph inputs added and the run's boundaries
   marked. *)
let storage_script ?(alignment = Alignment_policy.standard)
    ?(retain = Release_schedule.Retain.All) (config : Storage_script.Config.t)
    (g : Graph.graph) =
  let open Err.Syntax in
  let module S = Storage_script in
  let sched =
    Release_schedule.schedule ~operands:Op.operands ~is_sink ~retain g
  in
  let released =
    List.fold_left
      (fun set id -> Tensor_id.Set.add id set)
      (released_ids g sched)
      (Release_schedule.initial sched)
  in
  let block role id =
    let* a = alloc_of ~policy:alignment g ~released id in
    let kept = not (Tensor_id.Set.mem id released) in
    let role =
      match role with
      | S.Role.Intermediate when kept -> S.Role.Output
      | role -> role
    in
    let quantized = Option.is_some a.Alloc_script.Alloc.signature.quant in
    Err.return
      {
        S.Block.alloc = a;
        role;
        arena = S.arena_of config role ~quantized ~kept;
      }
  in
  let* constants =
    Err.List.map (block S.Role.Constant)
      (List.filter
         (fun id ->
           Graph.input_kind g id = Graph_ir.Input.Constant
           && constant_is_used g id)
         g.Graph.Graph.inputs)
  in
  let* inputs = Err.List.map (block S.Role.Input) (input_ids g) in
  let owned =
    List.fold_left
      (fun m (b : S.Block.t) ->
        if S.Block.reclaimable b then
          Tensor_id.Map.add b.S.Block.alloc.Alloc_script.Alloc.id b m
        else m)
      Tensor_id.Map.empty inputs
  in
  let allocs = List.map (fun b -> S.Event.Alloc b) in
  let head =
    List.rev
      ((S.Event.Boundary S.Boundary.Model_init :: allocs constants)
      @ (S.Event.Boundary S.Boundary.Input_population :: allocs inputs))
  in
  let step (node : Graph.node) (owned, events) =
    let* shapes = node_shapes g node.Graph.Node.op in
    let* outs = node_outs ~sched node shapes in
    let+ blocks =
      Err.List.map (fun (_, oid, _) -> block S.Role.Intermediate oid) outs
    in
    ( List.fold_left
        (fun m (b : S.Block.t) ->
          Tensor_id.Map.add b.S.Block.alloc.Alloc_script.Alloc.id b m)
        owned blocks,
      List.rev_append (allocs blocks) (S.Event.Node node.Graph.Node.id :: events)
    )
  in
  let free ids (owned, events) =
    List.fold_left
      (fun (owned, events) id ->
        if Tensor_id.Map.mem id owned then
          (Tensor_id.Map.remove id owned, S.Event.Free id :: events)
        else (owned, events))
      (owned, events) ids
  in
  let+ owned, events = walk ~sched g ~step ~release:free (owned, head) in
  let tail =
    S.Event.Boundary S.Boundary.Result_publication
    :: S.Event.Boundary S.Boundary.Result_release
    :: List.map (fun (id, _) -> S.Event.Free id) (Tensor_id.Map.bindings owned)
  in
  S.make config alignment (List.rev_append events tail)
