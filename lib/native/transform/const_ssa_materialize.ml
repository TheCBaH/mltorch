open Graph_ir

type error =
  [ `Missing_capture of Const_ssa.Capture.t
  | `Direct of Eval_direct.error
  | `Signature_mismatch of Const_ssa.Value_id.t
  | Constant_store.error ]

type report = { captures : int; applies : int; cache_hits : int }
type resolver = Const_ssa.Capture.t -> (Tensor.packed, error) Err.t

let pp_error ppf : [< error ] -> unit = function
  | `Missing_capture capture ->
      Fmt.pf ppf "missing captured constant %a" Const_ssa.Capture.pp capture
  | `Direct e -> Eval_direct.pp_error ppf e
  | `Signature_mismatch value ->
      Fmt.pf ppf "materialized %a disagrees with its signature"
        Const_ssa.Value_id.pp value
  | #Constant_store.error as e -> Constant_store.pp_error ppf e

let same_shape a b =
  List.for_all
    (fun axis -> Dim.equal (Vec6.get a axis) (Vec6.get b axis))
    Axis.all

let fmt_name (Payload.Fmt f) = Payload.fmt_name f

let matches sg (Tensor.Tensor payload) =
  same_shape sg.Tensor_sig.shape payload.Tensor.shape
  && String.equal (fmt_name sg.fmt)
       (Payload.fmt_name payload.Tensor.payload.fmt)

let materialize ?needed resolve store =
  let open Err.Syntax in
  let plan = Constant_store.plan store in
  let cache = ref (Constant_store.materialized store) in
  let report = ref { captures = 0; applies = 0; cache_hits = 0 } in
  let bump f = report := f !report in
  let rec value id =
    let raw = Const_ssa.Value_id.to_tensor_id id in
    match Tensor_id.Map.find_opt raw !cache with
    | Some payload ->
        bump (fun r -> { r with cache_hits = r.cache_hits + 1 });
        Err.return payload
    | None -> (
        match Const_ssa.find plan id with
        | None ->
            Err.fail
              (`Missing_capture
                 (Const_ssa.Capture.of_string
                    (Format.asprintf "%a" Const_ssa.Value_id.pp id)))
        | Some (Const_ssa.Leaf { leaf = Const_ssa.Captured capture; output }) ->
            let* payload = resolve capture in
            if not (matches output payload) then
              Err.fail (`Signature_mismatch id)
            else (
              cache := Tensor_id.Map.add raw payload !cache;
              bump (fun r -> { r with captures = r.captures + 1 });
              Err.return payload)
        | Some
            (Const_ssa.Leaf
               {
                 leaf = Const_ssa.Literal payload | Opaque_materialized payload;
                 output;
               }) ->
            if not (matches output payload) then
              Err.fail (`Signature_mismatch id)
            else (
              cache := Tensor_id.Map.add raw payload !cache;
              Err.return payload)
        | Some (Const_ssa.Apply { op; output }) ->
            let operands = Graph_ir.operands op in
            let* payloads =
              Err.List.map
                (fun operand -> value (Const_ssa.Value_id.of_tensor_id operand))
                operands
            in
            let tensors =
              List.fold_left2
                (fun m operand _payload ->
                  match
                    Const_ssa.sig_of plan
                      (Const_ssa.Value_id.of_tensor_id operand)
                  with
                  | Some sg -> Tensor_id.Map.add operand sg m
                  | None -> m)
                (Tensor_id.Map.singleton output.id output)
                operands payloads
            in
            let graph =
              {
                Graph.nodes =
                  [
                    { Node.id = Node_id.of_int 0; op; outputs = [ output.id ] };
                  ];
                root =
                  {
                    Group.id = Group_id.of_int 0;
                    label = None;
                    items = [ Group.Node (Node_id.of_int 0) ];
                  };
                tensors;
                inputs = operands;
                input_kinds =
                  List.fold_left
                    (fun m id -> Tensor_id.Map.add id Input.Constant m)
                    Tensor_id.Map.empty operands;
                outputs = [ output.id ];
              }
            in
            let* env =
              Eval_direct.run graph
                ~constants:(List.combine operands payloads)
                ~inputs:[]
              |> Err.map_error (fun e -> `Direct e)
            in
            let payload = Tensor_id.Map.find output.id env in
            if not (matches output payload) then
              Err.fail (`Signature_mismatch id)
            else (
              cache := Tensor_id.Map.add raw payload !cache;
              bump (fun r -> { r with applies = r.applies + 1 });
              Err.return payload))
  in
  let targets =
    Option.value needed
      ~default:
        (Constant_store.bindings store |> List.map fst |> Tensor_id.Set.of_list)
  in
  let* store =
    Tensor_id.Set.fold
      (fun tensor acc ->
        let* store = acc in
        match Constant_store.binding store tensor with
        | None -> Err.return store
        | Some id -> (
            let* payload = value id in
            match Const_ssa.sig_of plan id with
            | None -> Err.return store
            | Some sg -> (
                (* Plan ids are immutable while graph exports can be packed.
                   The payload belongs under the CURRENT graph tensor [tensor],
                   not [sg.id]'s historical plan id; the latter may no longer
                   occur in the graph at all. *)
                let export = { sg with Tensor_sig.id = tensor } in
                (* A preload may already have cached this exported captured
                   leaf. Materialization is an explicit, repeatable boundary,
                   not a request to overwrite state: the cache value was
                   validated at origin time and [value] returned that exact
                   payload, so retaining it is the only idempotent answer. *)
                match
                  Tensor_id.Map.find_opt tensor
                    (Constant_store.materialized store)
                with
                | Some _ -> Err.return store
                | None ->
                    Constant_store.bind_materialized store ~tensor:export
                      payload
                    |> Err.map_error (fun e -> (e :> error)))))
      targets (Err.return store)
  in
  Err.return (store, !report)
