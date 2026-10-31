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
  | `Region_lower of Loop_region_program.error
  | `Ungranted of Node_id.t ]

(* The row [build_with] reports for any dialect. The builder owns the Region
   validation after a dialect hands back a group or program, and the output
   signature lookup, so those have shared arms; only the storage script's
   failures are the dialect's own, behind [`Dialect]. *)
type 'err generic_error =
  [ Arena_plan.error
  | Loop_node_program.error
  | `Dialect of 'err
  | `Missing_signature of Tensor_id.t
  | `Plan_mismatch of Alloc_script.Position.t
  | `Region_construction of Region_computation.error
  | `Region_lower of Loop_region_program.error
  | `Ungranted of Node_id.t ]

let pp_ungranted ppf node =
  Format.fprintf ppf
    "%a is realized only under a grant, which this bundle was not given"
    Node_id.pp node

let pp_generic_error pp_dialect ppf : [< _ generic_error ] -> unit = function
  | #Arena_plan.error as e -> Arena_run.pp_error ppf (e :> Arena_run.error)
  | #Loop_node_program.error as e -> Loop_node_program.pp_error ppf e
  | `Dialect e -> pp_dialect ppf e
  | `Missing_signature id ->
      Format.fprintf ppf "missing shape lookup tensor %a" Tensor_id.pp id
  | `Plan_mismatch pos ->
      Format.fprintf ppf
        "the supplied storage plan's script differs from this graph's at %a"
        Alloc_script.Position.pp pos
  | `Region_construction e -> Region_computation.pp_error ppf e
  | `Region_lower e -> Loop_region_program.pp_error ppf e
  | `Ungranted node -> pp_ungranted ppf node

let pp_error ppf : [< error ] -> unit = function
  | #Arena_plan.error as e -> Arena_run.pp_error ppf (e :> Arena_run.error)
  | #Eval_direct.error as e -> Eval_direct.pp_error ppf e
  | #Loop_node_program.error as e -> Loop_node_program.pp_error ppf e
  | `Plan_mismatch pos ->
      Format.fprintf ppf
        "the supplied storage plan's script differs from this graph's at %a"
        Alloc_script.Position.pp pos
  | `Region_lower e -> Loop_region_program.pp_error ppf e
  | `Ungranted node -> pp_ungranted ppf node

let default_config : S.Config.t =
  {
    layout = S.Layout.Separate;
    constants = S.Ownership.Copied;
    inputs = S.Ownership.Copied;
  }

let erase (g : _ Graph.t) : unit Graph.t =
  {
    g with
    Graph.nodes =
      List.map (fun (n : _ Node.t) -> { n with Node.op = () }) g.Graph.nodes;
  }

type t = {
  graph : unit Graph.t;
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
let schedule_ordinals (g : _ Graph.t) (events : S.Event.t list) =
  let by_id =
    List.fold_left
      (fun m (n : _ Node.t) -> Node_id.Map.add n.Node.id n m)
      Node_id.Map.empty g.Graph.nodes
  in
  let output_of_oid (node : _ Node.t) oid =
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
let region_program ~(dialect : (_, _) Loop_dialect.t) ~limits (g : _ Graph.t)
    (node : _ Node.t) ~(outputs : Output.t list) =
  let open Err.Syntax in
  let op = node.Node.op in
  let synthetic_ids = dialect.Loop_dialect.fresh_synthetic_ids g in
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
          dialect.Loop_dialect.region_group ~limits ~op ~operand
          |> Err.map_error (fun e -> `Region_construction e)
        in
        let* lowered =
          Region_execution.lower_group ~max_size:limits.Kernel.Limits.max_size
            ~max_depth:limits.Kernel.Limits.max_depth
            ~max_local_slots:limits.Kernel.Limits.max_local_slots
            ~scan_limits:(Kernel.Limits.scan_limits limits)
            group
          |> Err.map_error (fun e ->
              `Region_construction (Region_computation.Invalid_group e))
        in
        let* plan, ids =
          Loop_region_program.plan_group_sigs ~limits ~sigs:(sigs ())
            ~selected:
              (List.map
                 (fun (o : Output.t) ->
                   Region_computation.emitter_of_output o.Output.ordinal)
                 outputs)
            (Region_execution.group lowered)
          |> Err.map_error (fun e -> `Region_lower e)
        in
        let+ program =
          Loop_lower.lower plan
          |> Err.map_error (fun e -> `Region_lower (`Lower e))
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
          | None -> Err.fail (`Missing_signature oid)
        in
        let* program =
          dialect.Loop_dialect.region_program ~limits ~op ~output
            ~output_shape:out_shape ~operand ~fill
          |> Err.map_error (fun e -> `Region_construction e)
        in
        let* lowered =
          Region_execution.lower ~max_size:limits.Kernel.Limits.max_size
            ~max_depth:limits.Kernel.Limits.max_depth
            ~max_local_slots:limits.Kernel.Limits.max_local_slots
            ~scan_limits:(Kernel.Limits.scan_limits limits)
            ~output_shape:out_shape program
          |> Err.map_error (fun e ->
              `Region_construction (Region_computation.Invalid_program e))
        in
        match lowered with
        | Region_execution.Pixel_loop _ -> assert false
        | Region_execution.Region_loop l ->
            let* plan =
              Loop_region_program.plan_sigs ~limits ~sigs:(sigs ()) ~out_shape
                (Region_execution.program l)
              |> Err.map_error (fun e -> `Region_lower e)
            in
            let+ program =
              Loop_lower.lower plan
              |> Err.map_error (fun e -> `Region_lower (`Lower e))
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

(* The placeholder of a node no Loop body realizes (an attention layer):
   a kernel that reads one element of each operand, so the program declares a
   buffer for every one of them (an unread input is never an argument), and
   writes NaN to the output. Only a granted caller is given it, and that caller
   replaces the body (a Machine IR site); no route is meant to run it as it
   is, and one that did would see NaN, never a plausible value. *)
let granted_stub ~limits (g : _ Graph.t) ~operands ~oid =
  let open Err.Syntax in
  let sig_of id =
    match Tensor_id.Map.find_opt id g.Graph.tensors with
    | Some sg -> Err.return sg
    | None -> Err.fail (`Missing_signature id)
  in
  let* sigs = Err.List.map sig_of (List.sort_uniq Tensor_id.compare operands) in
  let* out = sig_of oid in
  let zero = Expr.Index.zero in
  let origin =
    Expr.Coord.make ~n:zero ~t:zero ~d:zero ~h:zero ~w:zero ~c:zero
  in
  let pixel =
    List.fold_left
      (fun acc sg -> Expr.Value.add acc (Region_context.load sg origin))
      (Expr.Value.const Float.nan)
      sigs
  in
  let inputs =
    List.map
      (fun (sg : Tensor_sig.t) ->
        {
          Kernel.Input.id = sg.Tensor_sig.id;
          sg;
          binding =
            (match
               Tensor_id.Map.find_opt sg.Tensor_sig.id g.Graph.input_kinds
             with
            | Some Graph_common.Input.Constant ->
                Kernel.Binding.Captured_constant
            | Some Graph_common.Input.Input | None -> Kernel.Binding.Caller);
        })
      sigs
  in
  let* kernel =
    Kernel.create ~limits ~inputs
      ~values:
        [
          {
            Kernel.Value.id = oid;
            sg = out;
            computation = Region_group.Ref.Solo (Region_program.pixel pixel);
            result = Kernel.Result_conversion.Round_f32;
          };
        ]
      ~outputs:[ oid ] ()
    |> Err.map_error (fun e -> `Adapt (e :> Kernel_adapt.error))
  in
  let plan = Fusion_plan.default kernel in
  let+ program = Loop_lower.lower plan |> Err.map_error (fun e -> `Lower e) in
  (plan, program)

let build_with ~(dialect : (_, _) Loop_dialect.t)
    ?(limits = Kernel.Limits.default) ?(config = default_config) ?plan
    ?(granted = fun _ -> false) (g : _ Graph.t) =
  let open Err.Syntax in
  let* script =
    dialect.Loop_dialect.storage_script
      ~retain:(Release_schedule.Retain.Only Tensor_id.Set.empty) config g
    |> Err.map_error ~pos:__POS__ (fun e -> `Dialect e)
  in
  let* plan =
    match plan with
    | None ->
        Storage_plan.create ~limits script
        |> Err.map_error ~pos:__POS__ (fun e -> (e :> _ generic_error))
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
    | ((node : _ Node.t), o) :: rest
      when dialect.Loop_dialect.is_region_authored node.Node.op ->
        let mine, others =
          List.partition
            (fun ((n : _ Node.t), _) -> Node_id.equal n.Node.id node.Node.id)
            rest
        in
        (node, o :: List.map snd mine) :: batches others
    | ((node : _ Node.t), o) :: rest -> (node, [ o ]) :: batches rest
  in
  let+ invocations =
    Err.List.map
      (fun ((node : _ Node.t), (outputs : Output.t list)) ->
        if dialect.Loop_dialect.is_region_authored node.Node.op then
          let+ plan, program, edges, synthetics =
            region_program ~dialect ~limits g node ~outputs
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
          let ({ Output.oid; _ } : Output.t) = List.hd outputs in
          let+ plan, program =
            match dialect.Loop_dialect.granted_only node.Node.op with
            | Some operands ->
                if granted node.Node.id then
                  granted_stub ~limits g ~operands ~oid
                else Err.fail (`Ungranted node.Node.id)
            | None ->
                Loop_node_program.lower_placed_of_program ~limits
                  (dialect.Loop_dialect.node_program ~limits g node)
                  ~oid
                |> Err.map_error (fun e -> (e :> _ generic_error))
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
    graph = erase g;
    config;
    script;
    plan;
    inputs;
    constants;
    outputs = g.Graph.outputs;
    invocations;
  }

(* The Native public contract: [build]'s error row is unchanged, so the generic
   arms map back to the [Eval_direct.error] ones they replaced. *)
let build ?limits ?config ?plan ?granted (g : graph) : (t, error) Err.t =
  build_with ~dialect:Loop_dialect.native ?limits ?config ?plan ?granted g
  |> Err.map_error ~pos:__POS__ (function
    | #Arena_plan.error as e -> (e :> error)
    | #Loop_node_program.error as e -> (e :> error)
    | `Dialect e -> (e :> error)
    | `Missing_signature id ->
        `Missing_tensor { Eval_direct.context = Eval_direct.Sig_shape; id }
    | `Plan_mismatch _ as e -> e
    | `Region_construction _ as e -> e
    | `Region_lower _ as e -> e
    | `Ungranted _ as e -> e)
