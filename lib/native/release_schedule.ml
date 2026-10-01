(* See release_schedule.mli. Keyed by [Node_id.t]/[Tensor_id.t], never by a
   position, so nothing here needs a bound for js_of_ocaml's 32-bit [int]. *)

open Graph_common

module Retain = struct
  type t = All | Only of Tensor_id.Set.t
end

module Schedule = struct
  type t = {
    after : Tensor_id.t list Node_id.Map.t;
    initial : Tensor_id.t list;
    read : Tensor_id.Set.t; (* graph outputs and non-sink operands *)
  }
end

let add_all set ids =
  List.fold_left (fun s id -> Tensor_id.Set.add id s) set ids

let schedule ~operands ~is_sink ~retain (g : 'op Graph.t) =
  let outputs = add_all Tensor_id.Set.empty g.Graph.outputs in
  let kept =
    match retain with
    | Retain.All -> None
    | Retain.Only s -> Some (Tensor_id.Set.union s outputs)
  in
  let released id =
    match kept with None -> false | Some k -> not (Tensor_id.Set.mem id k)
  in
  (* Backward, so the first reader met is the last in node order. [read] holds
     every edge a later non-sink node reads. Outputs are checked before the
     node's own operands are added, though no op reads its own output. *)
  let after, read =
    List.fold_left
      (fun (after, read) (node : 'op Node.t) ->
        if is_sink node.Node.op then (after, read)
        else
          let dead =
            List.filter
              (fun id -> not (Tensor_id.Set.mem id read))
              node.Node.outputs
          in
          let last, read =
            List.fold_left
              (fun (last, read) id ->
                if Tensor_id.Set.mem id read then (last, read)
                else (id :: last, Tensor_id.Set.add id read))
              ([], read) (operands node.Node.op)
          in
          match List.filter released (dead @ List.rev last) with
          | [] -> (after, read)
          | ids -> (Node_id.Map.add node.Node.id ids after, read))
      (Node_id.Map.empty, Tensor_id.Set.empty)
      (List.rev g.Graph.nodes)
  in
  let read = Tensor_id.Set.union read outputs in
  let initial =
    List.filter
      (fun id -> released id && not (Tensor_id.Set.mem id read))
      g.Graph.inputs
  in
  { Schedule.after; initial; read }

let initial (s : Schedule.t) = s.Schedule.initial

let after (s : Schedule.t) id =
  Option.value ~default:[] (Node_id.Map.find_opt id s.Schedule.after)

let has_reader (s : Schedule.t) id = Tensor_id.Set.mem id s.Schedule.read

(* [numel] is bounded below [Hard.numel] (2^31) and a cell is at most 8 bytes,
   so one edge's bytes cannot overflow. Their sum is a different aggregate and
   gets its own check. *)
let edge_bytes (g : 'op Graph.t) id =
  let open Err.Syntax in
  let* sg =
    Tensor_id.Map.find_opt id g.Graph.tensors
    |> Err.of_option (`Missing_tensor_sig id)
  in
  let+ numel =
    Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel sg.Tensor_sig.shape
  in
  Int64.mul numel (Int64.of_int (Payload.packed_cell_bytes sg.Tensor_sig.fmt))

let add_bytes id acc bytes =
  if Int64.compare acc (Int64.sub Int64.max_int bytes) > 0 then
    Err.fail (`Peak_bytes_overflow id)
  else Err.return (Int64.add acc bytes)

let peak_bytes ~is_index_output (g : 'op Graph.t) (s : Schedule.t) =
  let open Err.Syntax in
  let* resident =
    Err.List.fold_left
      (fun acc id ->
        let* bytes = edge_bytes g id in
        add_bytes id acc bytes)
      0L g.Graph.inputs
  in
  (* [live] holds the bytes of each allocated node output not yet released;
     an input is never in it, so its release frees nothing here. *)
  let+ _, _, peak =
    Err.List.fold_left
      (fun (resident, live, peak) (node : 'op Node.t) ->
        let* resident, live =
          Err.List.fold_left
            (fun (resident, live) (output, id) ->
              if is_index_output node.Node.op output && not (has_reader s id)
              then Err.return (resident, live)
              else
                let* bytes = edge_bytes g id in
                let+ resident = add_bytes id resident bytes in
                (resident, Tensor_id.Map.add id bytes live))
            (resident, live)
            (Output_ordinal.indexed node.Node.outputs)
        in
        let peak = if Int64.compare resident peak > 0 then resident else peak in
        let resident, live =
          List.fold_left
            (fun (resident, live) id ->
              match Tensor_id.Map.find_opt id live with
              | None -> (resident, live)
              | Some bytes ->
                  (Int64.sub resident bytes, Tensor_id.Map.remove id live))
            (resident, live) (after s node.Node.id)
        in
        Err.return (resident, live, peak))
      (resident, Tensor_id.Map.empty, resident)
      g.Graph.nodes
  in
  peak
