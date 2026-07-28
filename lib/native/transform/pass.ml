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

(* What verification a run is under. Threaded THROUGH the pass tree rather than
   applied at its boundary, because a composite verified only at the boundary is
   barely verified: a [fixpoint] iteration or a [sequence] member can be wrong
   and cancelled by a later one, and the error would name the composite rather
   than the pass at fault. *)
type ctx = {
  budget : Map_verify.Budget.t option;
  policy : Map_verify.Policy.t option;
}

let no_verification = { budget = None; policy = None }

type t = {
  name : string;
  run : 'v. ctx -> 'v Rewrite.t -> ('v Rewrite.step, error) Core.result;
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

(* Verify one step against the state it came from, naming the pass. An identity
   step is skipped: its map is empty, so there is nothing to check, and on a real
   graph checking every cluster of a no-op sweep is the dominant cost. *)
let verified name ctx state (Rewrite.Step (_, map) as step) =
  match ctx.policy with
  | None -> Core.return step
  | Some _
    when Correspondence.is_empty map.Graph_map.values
         && Node_map.is_empty map.Graph_map.nodes ->
      Core.return step
  | Some policy -> (
      match Map_verify.step ?budget:ctx.budget state step with
      | Error e ->
          Core.fail
            (`Verification
               {
                 Verification.pass = name;
                 problem = Verification.Error e.Core.Error.kind;
               })
      | Ok report ->
          if Map_verify.Policy.accepts policy report then Core.return step
          else
            Core.fail
              (`Verification
                 {
                   Verification.pass = name;
                   problem = Verification.Rejected report;
                 }))

let of_sweep ~name collect =
  {
    name;
    run =
      (fun ctx state ->
        let env =
          { constants = Rewrite.constants state; view = Rewrite.view state }
        in
        let* result = sweep state (collect env) in
        let step =
          match result with None -> identity_step state | Some step -> step
        in
        verified name ctx state step);
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
      (fun ctx state ->
        let rec go : type v.
            int -> v Rewrite.step -> (v Rewrite.step, error) Core.result =
         fun fuel (Rewrite.Step (state, acc)) ->
          if fuel <= 0 then Core.fail (`Not_converged inner.name)
          else
            (* [ctx], so EVERY iteration is verified rather than only the
               composite the caller sees. *)
            let* (Rewrite.Step (next, map)) = inner.run ctx state in
            if changed map then
              go (fuel - 1) (Rewrite.Step (next, Graph_map.compose acc map))
            else Core.return (Rewrite.Step (state, acc))
        in
        go max_iters (identity_step state));
  }

(* Every pass verifies its own step, so this only threads the context. *)
let run_with ctx state passes =
  let rec go : type v.
      v Rewrite.t -> t list -> (v Rewrite.step, error) Core.result =
   fun state passes ->
    match passes with
    | [] -> Core.return (identity_step state)
    | pass :: rest ->
        let* (Rewrite.Step (next, map)) = pass.run ctx state in
        let+ (Rewrite.Step (final, rest_map)) = go next rest in
        Rewrite.Step (final, Graph_map.compose map rest_map)
  in
  go state passes

let run_all ?verify ?verify_budget state passes =
  run_with { budget = verify_budget; policy = verify } state passes

(* A fixed list of passes as one named pass, so a caller can [fixpoint] the
   whole group — needed when the passes unlock each other and no single
   non-interleaved round through them all is enough. [ctx] is forwarded, so a
   member is verified as it runs rather than only through the composite map the
   sequence hands back. *)
let sequence ~name passes =
  { name; run = (fun ctx state -> run_with ctx state passes) }
