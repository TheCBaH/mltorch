open Graph_ir
module S = Storage_script

type synthetic = { id : Tensor_id.t; value : float; shape : Vec6.shape }

module Output = struct
  type t = {
    ordinal : Output_ordinal.t;
    oid : Tensor_id.t;
    role : S.Role.t;
    arena : S.Arena_id.t option;
  }
end

type invocation = {
  node : Node_id.t;
  outputs : Output.t list;
  placed : Fusion_plan.t;
  program : Loop_program.t;
  edges : Tensor_id.t list;
  synthetics : synthetic list;
}

type error =
  [ Arena_plan.error
  | Eval_direct.error
  | Loop_node_program.error
  | `Plan_mismatch of Alloc_script.Position.t
  | `Region_lower of Loop_region_program.error ]

let pp_error ppf : [< error ] -> unit = function
  | #Arena_plan.error as e -> Arena_run.pp_error ppf (e :> Arena_run.error)
  | #Eval_direct.error as e -> Eval_direct.pp_error ppf e
  | #Loop_node_program.error as e -> Loop_node_program.pp_error ppf e
  | `Plan_mismatch pos ->
      Format.fprintf ppf
        "the supplied storage plan's script differs from this graph's at %a"
        Alloc_script.Position.pp pos
  | `Region_lower e -> Loop_region_program.pp_error ppf e

let default_config : S.Config.t =
  {
    layout = S.Layout.Separate;
    constants = S.Ownership.Copied;
    inputs = S.Ownership.Copied;
  }

type t = {
  graph : graph;
  config : S.Config.t;
  script : S.t;
  plan : Storage_plan.t;
  inputs : Tensor_id.t list;
  constants : Tensor_id.t list;
  outputs : Tensor_id.t list;
  invocations : invocation list;
}

(* Every [(node, Output.t)] the schedule actually
   allocates for a NODE's output, in schedule order -- read off
   [Eval_direct.storage_script]'s own event stream rather than re-walking
   [graph.Graph.nodes] and re-deciding admission/dead-index suppression/arena
   placement a second time. Constant/input blocks (the head section, before
   any [Node] marker) are skipped here: they carry [Role.Constant]/[Role.Input],
   never emitted for a node's own outputs (see [Eval_direct.storage_script]'s
   [step], which always classifies a node's own blocks [Role.Intermediate]
   before promoting a kept one to [Role.Output]), so filtering by role below
   recovers them unambiguously from the very same event list instead of a
   second graph walk. An [Alloc] under a [Node] always names one of that
   node's own outputs -- an invariant of the shared fold, not a contingency
   this function checks for. *)
let schedule_ordinals (g : graph) (events : S.Event.t list) =
  let by_id =
    List.fold_left
      (fun m (n : node) -> Node_id.Map.add n.Node.id n m)
      Node_id.Map.empty g.Graph.nodes
  in
  let output_of_oid (node : node) oid =
    Output_ordinal.indexed node.Node.outputs
    |> List.find (fun (_, o) -> Tensor_id.equal o oid)
    |> fst
  in
  let _, triples =
    List.fold_left
      (fun (current, acc) (event : S.Event.t) ->
        match event with
        | S.Event.Node id -> (Some (Node_id.Map.find id by_id), acc)
        | S.Event.Alloc
            {
              S.Block.alloc;
              role = (S.Role.Intermediate | S.Role.Output) as role;
              arena;
            } -> (
            match current with
            | None -> (current, acc)
            | Some node ->
                let oid = alloc.Alloc_script.Alloc.id in
                let output = output_of_oid node oid in
                ( current,
                  (node, { Output.ordinal = output; oid; role; arena }) :: acc
                ))
        | S.Event.Alloc { role = S.Role.Constant | S.Role.Input; _ }
        | S.Event.Boundary _ | S.Event.Free _ ->
            (current, acc))
      (None, []) events
  in
  List.rev triples

let ids_of_role events role =
  List.filter_map
    (function
      | S.Event.Alloc { S.Block.alloc; role = r; _ } when r = role ->
          Some alloc.Alloc_script.Alloc.id
      | _ -> None)
    events

(* A Region-authored node's invocation, built from signatures alone: the same
   [Region_computation.program]/[Region_execution.lower] steps
   [Eval_direct.region_result] takes, then [Loop_region_program]'s tensor-free
   lowering. The program's own buffer ids are local to it -- its output id is
   minted past its sources' -- so each positional buffer is bound to a graph edge
   through [edges], never by id: a source keeps its graph id, the output takes
   [oid], and a synthetic default (an omitted Sdpa mask, say) is named by its
   own fresh id and described in [synthetics]. *)
let region_program ~limits (g : graph) (node : node) ~(outputs : Output.t list)
    =
  let open Err.Syntax in
  let op = node.Node.op in
  let synthetic_ids = Eval_direct.fresh_synthetic_ids g in
  let filled = ref Tensor_id.Map.empty in
  let fill role value shape =
    let id = List.assoc role synthetic_ids in
    let sg =
      Tensor_sig.create ~id ~name:"bundle optional operand" ~shape
        ~fmt:(Payload.Fmt Payload.F32) ()
    in
    filled := Tensor_id.Map.add id ({ id; value; shape }, sg) !filled;
    sg
  in
  let operand id = Tensor_id.Map.find_opt id g.Graph.tensors in
  let sigs () =
    Tensor_id.Map.union
      (fun _ a _ -> Some a)
      g.Graph.tensors
      (Tensor_id.Map.map snd !filled)
  in
  let* lowered_program =
    match Output_ordinal.indexed node.Node.outputs with
    | _ :: _ :: _ ->
        let* group =
          Region_computation.group ~limits ~op ~operand
          |> Err.map_error (fun e -> (`Region_construction e :> error))
        in
        let* lowered =
          Region_execution.lower_group ~max_size:limits.Kernel.Limits.max_size
            ~max_depth:limits.Kernel.Limits.max_depth
            ~max_local_slots:limits.Kernel.Limits.max_local_slots
            ~scan_limits:(Kernel.Limits.scan_limits limits)
            group
          |> Err.map_error (fun e ->
              (`Region_construction (Region_computation.Invalid_group e)
                :> error))
        in
        let* plan, ids =
          Loop_region_program.plan_group_sigs ~limits ~sigs:(sigs ())
            ~selected:
              (List.map
                 (fun (o : Output.t) ->
                   Region_computation.emitter_of_output o.Output.ordinal)
                 outputs)
            (Region_execution.group lowered)
          |> Err.map_error (fun e -> (`Region_lower e :> error))
        in
        let+ program =
          Loop_lower.lower plan
          |> Err.map_error (fun e -> (`Region_lower (`Lower e) :> error))
        in
        ( plan,
          program,
          List.map2
            (fun (o : Output.t) (_, id) -> (id, o.Output.oid))
            outputs ids )
    | _ -> (
        let ({ Output.ordinal = output; oid; _ } : Output.t) =
          List.hd outputs
        in
        let* out_shape =
          match Tensor_id.Map.find_opt oid g.Graph.tensors with
          | Some sg -> Err.return sg.Tensor_sig.shape
          | None ->
              Err.fail
                (`Missing_tensor
                   { Eval_direct.context = Eval_direct.Sig_shape; id = oid })
        in
        let* program =
          Region_computation.program ~limits ~op ~output ~output_shape:out_shape
            ~operand ~fill
          |> Err.map_error (fun e -> (`Region_construction e :> error))
        in
        let* lowered =
          Region_execution.lower ~max_size:limits.Kernel.Limits.max_size
            ~max_depth:limits.Kernel.Limits.max_depth
            ~max_local_slots:limits.Kernel.Limits.max_local_slots
            ~scan_limits:(Kernel.Limits.scan_limits limits)
            ~output_shape:out_shape program
          |> Err.map_error (fun e ->
              (`Region_construction (Region_computation.Invalid_program e)
                :> error))
        in
        match lowered with
        | Region_execution.Pixel_loop _ -> assert false
        | Region_execution.Region_loop l ->
            let* plan =
              Loop_region_program.plan_sigs ~limits ~sigs:(sigs ()) ~out_shape
                (Region_execution.program l)
              |> Err.map_error (fun e -> (`Region_lower e :> error))
            in
            let+ program =
              Loop_lower.lower plan
              |> Err.map_error (fun e -> (`Region_lower (`Lower e) :> error))
            in
            let out =
              List.find
                (fun (b : Loop_buffer.t) ->
                  b.Loop_buffer.role = Loop_buffer.Output)
                program.Loop_program.buffers
            in
            (plan, program, [ (out.Loop_buffer.id, oid) ]))
  in
  let plan, program, bound_outputs = lowered_program in
  let synthetics =
    List.filter_map
      (fun (b : Loop_buffer.t) ->
        Option.map fst (Tensor_id.Map.find_opt b.Loop_buffer.id !filled))
      program.Loop_program.buffers
  in
  let edges =
    List.map
      (fun (b : Loop_buffer.t) ->
        match
          List.find_opt
            (fun (local, _) -> Tensor_id.equal b.Loop_buffer.id local)
            bound_outputs
        with
        | Some (_, oid) -> oid
        | None -> b.Loop_buffer.id)
      program.Loop_program.buffers
  in
  Err.return (plan, program, edges, synthetics)

let build ?(limits = Kernel.Limits.default) ?(config = default_config) ?plan
    (g : graph) =
  let open Err.Syntax in
  let* script =
    Eval_direct.storage_script
      ~retain:(Release_schedule.Retain.Only Tensor_id.Set.empty) config g
    |> Err.map_error ~pos:__POS__ (fun e -> (e :> error))
  in
  let* plan =
    match plan with
    | None ->
        Storage_plan.create ~limits script
        |> Err.map_error ~pos:__POS__ (fun e -> (e :> error))
    | Some supplied -> (
        (* A caller's plan is accepted only if the script it was witnessed
           against is this graph's own, config and alignment policy included. *)
        match S.first_difference (Storage_plan.script supplied) script with
        | None -> Err.return supplied
        | Some pos -> Err.fail (`Plan_mismatch pos))
  in
  let events = S.events script in
  let scheduled = schedule_ordinals g events in
  (* A grouped Region node's scheduled outputs share one recurrence, so they are
     one invocation; every other node's outputs are lowered one at a time. *)
  let rec batches = function
    | [] -> []
    | ((node : node), o) :: rest
      when Region_computation.is_region_authored node.Node.op ->
        let mine, others =
          List.partition
            (fun ((n : node), _) -> Node_id.equal n.Node.id node.Node.id)
            rest
        in
        (node, o :: List.map snd mine) :: batches others
    | ((node : node), o) :: rest -> (node, [ o ]) :: batches rest
  in
  let+ invocations =
    Err.List.map
      (fun ((node : node), (outputs : Output.t list)) ->
        if Region_computation.is_region_authored node.Node.op then
          let+ plan, program, edges, synthetics =
            region_program ~limits g node ~outputs
          in
          {
            node = node.Node.id;
            outputs;
            placed = plan;
            program;
            edges;
            synthetics;
          }
        else
          let ({ Output.ordinal = output; _ } : Output.t) = List.hd outputs in
          let+ plan, program =
            Loop_node_program.lower_placed ~limits g node ~output
            |> Err.map_error (fun e -> (e :> error))
          in
          {
            node = node.Node.id;
            outputs;
            placed = plan;
            program;
            edges =
              List.map
                (fun (b : Loop_buffer.t) -> b.Loop_buffer.id)
                program.Loop_program.buffers;
            synthetics = [];
          })
      (batches scheduled)
  in
  let inputs = ids_of_role events S.Role.Input in
  let constants = ids_of_role events S.Role.Constant in
  {
    graph = g;
    config;
    script;
    plan;
    inputs;
    constants;
    outputs = g.Graph.outputs;
    invocations;
  }
