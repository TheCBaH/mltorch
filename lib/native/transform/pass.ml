(* See pass.mli. *)

open Graph_ir
open Core.Syntax

type error = [ Rewrite.error | `Not_converged of string ]

let pp_error ppf : [< error ] -> unit = function
  | #Rewrite.error as e -> Rewrite.pp_error ppf e
  | `Not_converged name -> Fmt.pf ppf "@[<h>pass %s did not converge@]" name

type t = {
  name : string;
  run : 'v. 'v Rewrite.t -> ('v Rewrite.step, error) Core.result;
}

type per_node = { on_node : Graph_view.t -> node -> unit Recipe.t option }

let lift r = (r :> ('a, error) Core.result)

(* One sweep: collect the builders a pass offers, plan them one after another so
   their allocations are contiguous, merge, and apply once. Merging is what makes
   the sweep a single step with a single mapping, rather than N steps whose maps
   the caller would have to compose. *)
let sweep state builders =
  let* planned, _ =
    List.fold_left
      (fun acc builder ->
        let* recipes, alloc = acc in
        let+ recipe, alloc = lift (Rewrite.plan state alloc builder) in
        (recipe :: recipes, alloc))
      (Core.return ([], Rewrite.allocator state))
      builders
  in
  match List.rev planned with
  | [] -> Core.return None
  | first :: rest ->
      let* merged =
        List.fold_left
          (fun acc recipe ->
            let* acc = acc in
            lift (Rewrite.merge acc recipe))
          (Core.return first) rest
      in
      let+ step = lift (Rewrite.apply state merged) in
      Some step

let identity_step state = Rewrite.Step (state, Graph_map.identity)

let of_sweep ~name collect =
  {
    name;
    run =
      (fun state ->
        let* result = sweep state (collect (Rewrite.view state)) in
        Core.return
          (match result with None -> identity_step state | Some step -> step));
  }

let per_node ~name { on_node } =
  of_sweep ~name (fun view ->
      List.filter_map (on_node view) (Graph_ir.nodes (Graph_view.graph view)))

let of_pattern ~name ~pattern ~build =
  of_sweep ~name (fun view ->
      Pattern.scan pattern view
      |> List.map (fun (value, region) -> build value region))

(* Convergence is "the graph stopped changing", read off the node and tensor
   counts plus the map being empty — a step that rewrote nothing produces the
   identity map, which is exactly the signal. *)
let changed (map : ('a, 'b) Graph_map.t) =
  not
    (Correspondence.is_empty map.Graph_map.values
    && Node_map.is_empty map.Graph_map.nodes)

(* The accumulated mapping's destination changes every iteration, so it cannot
   be carried as a [('v,'w) Graph_map.t] with 'w fixed. [Rewrite.step] already
   packages a state together with the map reaching it, existentially — so the
   accumulator IS a step, and composing into it keeps the caller's view as one
   mapping from the state it handed in to the final graph. *)
let fixpoint ?(max_iters = 16) inner =
  {
    name = inner.name;
    run =
      (fun state ->
        let rec go : type v.
            int -> v Rewrite.step -> (v Rewrite.step, error) Core.result =
         fun fuel (Rewrite.Step (state, acc)) ->
          if fuel <= 0 then Core.fail (`Not_converged inner.name)
          else
            let* (Rewrite.Step (next, map)) = inner.run state in
            if changed map then
              go (fuel - 1) (Rewrite.Step (next, Graph_map.compose acc map))
            else Core.return (Rewrite.Step (state, acc))
        in
        go max_iters (identity_step state));
  }

let run_all state passes =
  let rec go : type v.
      v Rewrite.t -> t list -> (v Rewrite.step, error) Core.result =
   fun state passes ->
    match passes with
    | [] -> Core.return (identity_step state)
    | pass :: rest ->
        let* (Rewrite.Step (next, map)) = pass.run state in
        let+ (Rewrite.Step (final, rest_map)) = go next rest in
        Rewrite.Step (final, Graph_map.compose map rest_map)
  in
  go state passes
