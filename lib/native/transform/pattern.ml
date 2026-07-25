(* See pattern.mli. *)

open Graph_ir

type mismatch =
  | Ambiguous_use of Tensor_id.t
  | Escapes of Tensor_id.t
  | No_progress of Tensor_id.t
  | Not_constant of Tensor_id.t
  | Rejected
  | Shared of Tensor_id.t
  | Unproduced of Tensor_id.t
  | Wrong_op of Node_id.t

let pp_mismatch fmt = function
  | Ambiguous_use id ->
      Fmt.pf fmt "@[<h>several consumers of %a match@]" Tensor_id.pp id
  | Escapes id ->
      Fmt.pf fmt "@[<h>%a is used outside the match or is a graph output@]"
        Tensor_id.pp id
  | No_progress id ->
      Fmt.pf fmt "@[<h>chain step did not move upstream of %a@]" Tensor_id.pp id
  | Not_constant id -> Fmt.pf fmt "@[<h>%a is not a constant@]" Tensor_id.pp id
  | Rejected -> Fmt.string fmt "rejected by a guard"
  | Shared id ->
      Fmt.pf fmt "@[<h>%a has more than one consumer@]" Tensor_id.pp id
  | Unproduced id ->
      Fmt.pf fmt "@[<h>%a is a graph input, not a node output@]" Tensor_id.pp id
  | Wrong_op id -> Fmt.pf fmt "@[<h>%a is not the expected op@]" Node_id.pp id

type failure = Invalid of Region.error | Mismatch of mismatch

let pp_failure fmt = function
  | Invalid e -> Region.pp_error fmt e
  | Mismatch m -> pp_mismatch fmt m

(* The view is read-only; only [claimed] grows, and it is threaded rather than
   mutated so a failed alternative simply drops its version of it. *)
type state = { view : Graph_view.t; claimed : Node_id.Set.t }
type 'a t = state -> ('a * state, failure) Stdlib.result

let return x s = Ok (x, s)
let ( let* ) m f s = match m s with Ok (x, s) -> f x s | Error e -> Error e

let ( let+ ) m f s =
  match m s with Ok (x, s) -> Ok (f x, s) | Error e -> Error e

let fail m _ = Error (Mismatch m)
let guard b s = if b then Ok ((), s) else Error (Mismatch Rejected)

let ( <|> ) a b s =
  match a s with
  | Ok r -> Ok r
  (* [Invalid] is a broken graph, not a failed guess, so it does not backtrack. *)
  | Error (Invalid _ as e) -> Error e
  | Error (Mismatch _) -> b s

let choice = function
  | [] -> fail Rejected
  | first :: rest -> List.fold_left ( <|> ) first rest

let optional m s =
  match m s with
  | Ok (x, s) -> Ok (Some x, s)
  | Error (Invalid _ as e) -> Error e
  | Error (Mismatch _) -> Ok (None, s)

let view s = Ok (s.view, s)
let claimed s = Ok (s.claimed, s)

(* Idempotent: a diamond's two paths reach the same producer, and re-claiming it
   has to succeed or no diamond could ever match. *)
let claim (n : node) s =
  Ok ((), { s with claimed = Node_id.Set.add n.Node.id s.claimed })

let producer ~take edge s =
  match Graph_view.def s.view edge with
  | None -> Error (Mismatch (Unproduced edge))
  | Some n -> (
      match take n.Node.op with
      | None -> Error (Mismatch (Wrong_op n.Node.id))
      | Some payload -> Ok ((payload, n), s))

let peek edge take = producer ~take edge

let def edge take =
  let* payload, n = producer ~take edge in
  let+ () = claim n in
  (payload, n)

let uses edge =
  let+ v = view in
  Graph_view.uses v edge

let use edge take =
  let* v = view in
  let candidates =
    List.filter_map
      (fun (n : node) ->
        Option.map (fun payload -> (payload, n)) (take n.Node.op))
      (Graph_view.uses v edge)
  in
  match candidates with
  | [] -> fail Rejected
  | [ (payload, n) ] ->
      let+ () = claim n in
      (payload, n)
  | _ -> fail (Ambiguous_use edge)

let interior edge =
  let* v = view in
  if Graph_view.is_graph_output v edge then fail (Escapes edge)
  else
    match Graph_view.uses v edge with
    | [ _ ] -> return ()
    | _ -> fail (Shared edge)

let constant edge =
  let* v = view in
  if Graph_view.is_constant v edge then return () else fail (Not_constant edge)

let sig_of edge =
  let* v = view in
  match Graph_view.sig_of v edge with
  | Some sg -> return sg
  | None -> fail (Unproduced edge)

let region s =
  match Region.of_nodes s.view s.claimed with
  | Ok r -> Ok (r, s)
  | Error e -> Error (Invalid e.Core.Error.kind)

(* Progress is measured in topological position, which walking up operands
   always decreases; the check is here so a step that returns its own edge (or
   one further down) fails instead of looping. *)
let chain step start =
  let* v = view in
  let position id =
    match Graph_view.def v id with
    | None -> -1
    | Some (n : node) ->
        Option.value (Graph_view.topo_index v n.Node.id) ~default:(-1)
  in
  let rec go edge acc =
    let* result = optional (step edge) in
    match result with
    | None -> return (List.rev acc)
    | Some (value, next) ->
        if position next < position edge then go next (value :: acc)
        else fail (No_progress edge)
  in
  go start []

let run m v =
  match m { view = v; claimed = Node_id.Set.empty } with
  | Error e -> Error e
  | Ok (value, s) -> (
      match Region.of_nodes s.view s.claimed with
      | Ok r -> Ok (value, r)
      | Error e -> Error (Invalid e.Core.Error.kind))

let scan pattern v =
  let anchors =
    List.concat_map
      (fun (n : node) -> n.Node.outputs)
      (Graph_ir.nodes (Graph_view.graph v))
  in
  (* Greedy and disjoint: a later match overlapping an accepted one is dropped
     rather than retried, so the results can all be applied in one recipe. *)
  List.fold_left
    (fun (taken, acc) anchor ->
      match run (pattern anchor) v with
      | Error _ -> (taken, acc)
      | Ok (value, (r : Region.t)) ->
          if Node_id.Set.is_empty (Node_id.Set.inter taken r.nodes) then
            (Node_id.Set.union taken r.nodes, (value, r) :: acc)
          else (taken, acc))
    (Node_id.Set.empty, []) anchors
  |> snd |> List.rev
