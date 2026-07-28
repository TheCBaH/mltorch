(* See pass.mli. *)

open Graph_ir
open Core.Syntax

module Verification = struct
  (* Two distinct failures, and the caller needs to tell them apart: the
     verifier itself can error (a map that does not describe its two graphs, a
     missing signature), or it can succeed and the policy reject what it found.
     Both carry the pass name, since the point of verifying per step is to say
     WHICH rewrite is at fault. *)
  type problem = Error of Map_verify.error | Rejected of Map_verify.Report.t
  type t = { pass : string; problem : problem }

  let pp fmt t =
    match t.problem with
    | Error e ->
        Fmt.pf fmt "@[<h>pass %s: verifier failed: %a@]" t.pass
          Map_verify.pp_error e
    | Rejected report ->
        Fmt.pf fmt "@[<v 2>pass %s rejected: %s@,%a@]" t.pass
          (Map_verify.Report.summary report)
          Map_verify.Report.pp_verdicts report
end

type error =
  [ Rewrite.error | `Not_converged of string | `Verification of Verification.t ]

let pp_error ppf : [< error ] -> unit = function
  | #Rewrite.error as e -> Rewrite.pp_error ppf e
  | `Not_converged name -> Fmt.pf ppf "@[<h>pass %s did not converge@]" name
  | `Verification v -> Verification.pp ppf v

type t = {
  name : string;
  run : 'v. 'v Rewrite.t -> ('v Rewrite.step, error) Core.result;
}

type env = { constants : Tensor.packed Tensor_id.Map.t; view : Graph_view.t }
type per_node = { on_node : env -> node -> unit Recipe.t option }

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
        let env =
          { constants = Rewrite.constants state; view = Rewrite.view state }
        in
        let* result = sweep state (collect env) in
        Core.return
          (match result with None -> identity_step state | Some step -> step));
  }

let per_node ~name { on_node } =
  of_sweep ~name (fun env ->
      List.filter_map (on_node env) (Graph_ir.nodes (Graph_view.graph env.view)))

let of_pattern ~name ~pattern ~build =
  of_sweep ~name (fun env ->
      Pattern.scan pattern env.view
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

(* Verify one step against the state it came from, and turn a rejection into an
   error naming the pass. Runs BEFORE the next pass, so the first offender stops
   the pipeline rather than a later composed map hiding which rewrite was
   wrong. Reports for accepted steps are dropped; a caller wanting every report
   calls [Map_verify.step] itself. *)
let verify_step name policy state step =
  match policy with
  | None -> Core.return ()
  | Some policy -> (
      match Map_verify.step state step with
      | Error e ->
          Core.fail
            (`Verification
               {
                 Verification.pass = name;
                 problem = Verification.Error e.Core.Error.kind;
               })
      | Ok report ->
          if Map_verify.Policy.accepts policy report then Core.return ()
          else
            Core.fail
              (`Verification
                 {
                   Verification.pass = name;
                   problem = Verification.Rejected report;
                 }))

let run_all ?verify state passes =
  let rec go : type v.
      v Rewrite.t -> t list -> (v Rewrite.step, error) Core.result =
   fun state passes ->
    match passes with
    | [] -> Core.return (identity_step state)
    | pass :: rest ->
        let* (Rewrite.Step (next, map) as step) = pass.run state in
        let* () = verify_step pass.name verify state step in
        let+ (Rewrite.Step (final, rest_map)) = go next rest in
        Rewrite.Step (final, Graph_map.compose map rest_map)
  in
  go state passes

(* A fixed list of passes as one named pass, so a caller can [fixpoint] the
   whole group — needed when the passes unlock each other and no single
   non-interleaved round through them all is enough. *)
let sequence ~name passes = { name; run = (fun state -> run_all state passes) }
