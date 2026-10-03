(* Executing a scheduled graph: outputs bit for bit those of the original, the
   run follows the selected order with the plan built for it, and a plan is
   never accepted for the order it was not built for. *)

open Graph_ir
module Plan = Arena_schedule_plan

let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty
let pp_plan_error ppf e = Plan.pp_error ppf (Err.Error.kind e)
let pp_error ppf e = Eval_direct.pp_error ppf (Err.Error.kind e)
let ok = function Ok v -> v | Error e -> Fmt.failwith "%a" pp_plan_error e

let config ?(mode = Arena_schedule.Mode.Intermediate) () =
  {
    Arena_schedule.Config.mode;
    retain = only_empty;
    alignment = Alignment_policy.standard;
    limits = Arena_schedule.Limits.default_beam;
  }

let tensor_of_sig (sg : Tensor_sig.t) =
  let v c =
    float_of_int (((Vec6.offset sg.Tensor_sig.shape c :> int) mod 7) - 3) /. 4.
  in
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.I64 ->
      Tensor.materialize_i64 sg.Tensor_sig.shape (fun c ->
          Int64.of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 2))
  | Payload.Fmt Payload.Bool ->
      Tensor.materialize_bool sg.Tensor_sig.shape (fun c -> v c > 0.)
  | _ -> Tensor.materialize sg.Tensor_sig.shape v

let bound (g : graph) kind =
  List.filter_map
    (fun id ->
      if input_kind g id = kind then
        Some (id, tensor_of_sig (Tensor_id.Map.find id g.Graph.tensors))
      else None)
    g.Graph.inputs

let run ?arena ?hooks (g : graph) =
  Eval_direct.run ?arena ?hooks ~retain:only_empty
    ~constants:(bound g Input.Constant) g ~inputs:(bound g Input.Input)

let same_outputs (g : graph) a b =
  match (a, b) with
  | Ok a, Ok b ->
      List.for_all
        (fun id ->
          Tensor.equal_bits (Tensor_id.Map.find id a) (Tensor_id.Map.find id b))
        g.Graph.outputs
  | Error a, Error b ->
      String.equal (Fmt.str "%a" pp_error a) (Fmt.str "%a" pp_error b)
  | _ -> false

let random_graph seed ~nodes =
  let rng = Random.State.make [| seed |] in
  let mp = { Reduce.Mean.dims = [ Axis.C ]; keepdim = true } in
  Graph_fixtures.build (Fmt.str "random%d" seed)
    Graph_builder.(
      let* x = input ~shape:(Graph_fixtures.s1c 8) () in
      let rec go i wide narrow last =
        if i = nodes then return last
        else
          let pick l = List.nth l (Random.State.int rng (List.length l)) in
          match Random.State.int rng 3 with
          | 0 ->
              let* t = relu (pick wide) in
              go (i + 1) (t :: wide) narrow t
          | 1 ->
              let* t = Graph_builder.mean mp (pick wide) in
              go (i + 1) wide (t :: narrow) t
          | _ ->
              let src = if narrow = [] then wide else narrow in
              let* t = add (pick src) (pick src) in
              if src == narrow then go (i + 1) wide (t :: narrow) t
              else go (i + 1) (t :: wide) narrow t
      in
      go 0 [ x ] [] x)

let graphs () =
  List.map (fun (n, b) -> (n, b ())) Graph_fixtures.all
  @ List.init 10 (fun i -> ("random", random_graph (i + 1) ~nodes:9))

let%expect_test "original, scheduled release-only and scheduled arena agree" =
  let bad = ref 0 and moved = ref 0 in
  List.iter
    (fun (name, g) ->
      let original = run g in
      let c = config () in
      let scheduled, _ = ok (Plan.release_only c g) in
      if scheduled != g then incr moved;
      let sel = ok (Plan.choose c g) in
      let in_arena =
        match sel.plan with
        | Some (Plan.Plan.Intermediate plan) -> (
            match
              Err.payload
                (Arena_run.acquire_plan ~admission:Arena.Admission.Best_effort
                   plan)
            with
            | Ok (Arena_run.Arena arena) -> run ~arena sel.graph
            | _ -> Fmt.failwith "no arena for %s" name)
        | _ -> run sel.graph
      in
      if not (same_outputs g original (run scheduled)) then (
        incr bad;
        Fmt.pr "%s: release-only differs@." name);
      if not (same_outputs g original in_arena) then (
        incr bad;
        Fmt.pr "%s: arena differs@." name))
    (graphs ());
  Fmt.pr "bad=%d moved>0=%b@." !bad (!moved > 0);
  [%expect {| bad=0 moved>0=true |}]

let%expect_test "hooks follow the selected order" =
  let g = random_graph 124 ~nodes:9 in
  let sel = ok (Plan.choose (config ()) g) in
  let seen = ref [] in
  let hooks =
    Eval_direct.Hooks
      {
        on_start = (fun n -> seen := n.Node.id :: !seen);
        on_end = (fun _ () -> ());
      }
  in
  ignore (run ~hooks sel.graph);
  let followed =
    List.rev !seen = List.map (fun n -> n.Node.id) sel.graph.Graph.nodes
  in
  let moved =
    List.map (fun n -> n.Node.id) g.Graph.nodes
    <> List.map (fun n -> n.Node.id) sel.graph.Graph.nodes
  in
  Fmt.pr "followed=%b reordered=%b@." followed moved;
  [%expect {| followed=true reordered=true |}]

let%expect_test "a plan is rejected for the order it was not built for" =
  let g = random_graph 124 ~nodes:9 in
  let sel = ok (Plan.choose (config ()) g) in
  let plan =
    match sel.plan with
    | Some (Plan.Plan.Intermediate p) -> p
    | _ -> failwith "no plan"
  in
  let arena =
    match
      Err.payload
        (Arena_run.acquire_plan ~admission:Arena.Admission.Best_effort plan)
    with
    | Ok (Arena_run.Arena a) -> a
    | _ -> failwith "no arena"
  in
  let started = ref 0 in
  let hooks =
    Eval_direct.Hooks
      { on_start = (fun _ -> incr started); on_end = (fun _ () -> ()) }
  in
  (match run ~arena ~hooks g with
  | Ok _ -> Fmt.pr "original order accepted@."
  | Error e -> Fmt.pr "original order: %a@." pp_error e);
  Fmt.pr "nodes started: %d@." !started;
  (* Role storage: the same, against a storage plan. *)
  let roles =
    {
      Storage_script.Config.layout = Separate;
      constants = Copied;
      inputs = Copied;
    }
  in
  let rcfg = config ~mode:(Roles roles) () in
  (* A seed whose role-mode choice differs from the original order. *)
  let g, rsel =
    let rec find seed =
      let g = random_graph seed ~nodes:9 in
      let sel = ok (Plan.choose rcfg g) in
      if
        List.map (fun n -> n.Node.id) sel.graph.Graph.nodes
        <> List.map (fun n -> n.Node.id) g.Graph.nodes
      then (g, sel)
      else find (seed + 1)
    in
    find 1
  in
  let script =
    match rsel.plan with
    | Some (Plan.Plan.Roles p) -> Storage_plan.script p
    | _ -> failwith "no storage plan"
  in
  started := 0;
  (match
     Eval_direct.run_storage ~arenas:[] ~script ~hooks ~retain:only_empty
       ~constants:(bound g Input.Constant) g ~inputs:(bound g Input.Input)
   with
  | Ok _ -> Fmt.pr "original order accepted@."
  | Error e -> Fmt.pr "storage, original order: %a@." pp_error e);
  Fmt.pr "nodes started: %d@." !started;
  [%expect
    {|
    original order: arena: the plan was built for a different run: at event @2 the plan has node n3, this run has node n1
    nodes started: 0
    storage, original order: arena: the storage plan was built for a different run: its script differs at event @5
    nodes started: 0 |}]
