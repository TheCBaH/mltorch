(* Domain types for map_verify.ml: Member/Budget/Effort/Coverage/Strength/
   Refutation/Unproved/Verdict/Outcome/Group_path/Entry/Tally/Report/Policy,
   plus the error type. No verification logic here — see map_verify_check.ml
   for that, and map_verify.mli for what is public. Split out. *)

open Graph_ir

module Member = struct
  (* A GADT rather than a pair of [Tensor_id.t]s, which DELETES the [id]
     accessor instead of typing it: [id] erased exactly the distinction this
     type exists to keep, and map_verify_types.ml:11 was a source id resolved
     in the destination's producer map. Reaching an id now means going through
     [resolve] below, which pairs a member with its own side. *)
  type ('src, 'dst) t =
    | Dst : 'dst Correspondence.id -> ('src, 'dst) t
    | Src : 'src Correspondence.id -> ('src, 'dst) t

  (* What a report shows. The report hierarchy is unparameterized and escapes
     into [Pass.outcome] and the interpreter's result record, so the version is
     dropped once graph lookup is done — checking is typed, diagnostics are
     read. The printed form is unchanged. *)
  module Erased = struct
    type t = { id : Tensor_id.t; side : [ `Dst | `Src ] }

    let rank t = match t.side with `Dst -> 0 | `Src -> 1

    let compare a b =
      match Stdlib.compare (rank a) (rank b) with
      | 0 -> Tensor_id.compare a.id b.id
      | n -> n

    let pp fmt t =
      Fmt.pf fmt "%s%a"
        (match t.side with `Dst -> "dst." | `Src -> "src.")
        Tensor_id.pp t.id
  end

  let erase : type s d. (s, d) t -> Erased.t = function
    | Dst e -> { Erased.id = Correspondence.raw e; side = `Dst }
    | Src e -> { Erased.id = Correspondence.raw e; side = `Src }
end

module Budget = struct
  type t = {
    max_coords : int;
    max_nodes : int;
    max_rounds : int;
    sample : int option;
  }

  let default =
    { max_coords = 4096; max_nodes = 200_000; max_rounds = 32; sample = None }

  let cumulative =
    {
      max_coords = 4096;
      max_nodes = 1_000_000;
      max_rounds = 256;
      sample = None;
    }

  (* For a real model, and shaped by what deep expansion is worth there: nothing.
     The clusters a real graph makes checkable are the CONSTANT-shaped ones —
     folded weights and biases, where fold_const and fold_batch_norm act — and
     those close in one or two rounds. An activation-shaped cluster needs as many
     rounds as the network is deep, and each round multiplies a conv's kernel x
     channels into the previous one, so it is hopeless however much budget it
     gets.

     Widening [max_coords] alone is a trap, and measurably so: a late-layer
     activation like [1,512,7,7] is only 25088 elements, so it passes a generous
     coord budget and then expands the entire network behind it. Verification
     ran over 25x the whole transform. Cutting [max_rounds] is what actually
     bounds it — deep clusters give up after two cheap rounds while shallow
     constant ones still close. *)
  let release =
    { max_coords = 65_536; max_nodes = 50_000; max_rounds = 2; sample = Some 8 }

  let pp fmt t =
    Fmt.pf fmt "@[<h>{coords<=%d nodes<=%d rounds<=%d sample=%a}@]" t.max_coords
      t.max_nodes t.max_rounds
      (Fmt.option ~none:(Fmt.any "none") Fmt.int)
      t.sample
end

(* A named point on the cost/coverage curve, so a caller picks how hard to look
   without knowing what a coord or a round is. Effort drives the probe count as
   well as the budget: a deeper frontier is worth more draws. *)
module Effort = struct
  type t = Quick | Standard | Thorough

  let all = [ Quick; Standard; Thorough ]

  let to_string = function
    | Quick -> "quick"
    | Standard -> "standard"
    | Thorough -> "thorough"

  let of_string s =
    match String.lowercase_ascii s with
    | "quick" -> Ok Quick
    | "standard" -> Ok Standard
    | "thorough" -> Ok Thorough
    | other -> Error (`Unknown_effort other)

  let budget = function
    | Quick ->
        {
          Budget.max_coords = 4096;
          max_nodes = 20_000;
          max_rounds = 1;
          sample = Some 4;
        }
    | Standard -> Budget.release
    | Thorough ->
        {
          Budget.max_coords = 65_536;
          max_nodes = 200_000;
          max_rounds = 6;
          sample = Some 32;
        }

  let probe = function Quick -> 2 | Standard -> 4 | Thorough -> 8

  (* A per-cluster budget does not cap an export: a real model may rewrite many
     times, and every changed rewrite used to receive a fresh budget.  Keep the
     per-pass evidence useful, but bound how many such reports an interactive
     export attempts.  The final composed report is still produced separately. *)
  (* The browser already receives the composed report, which is the evidence it
     renders on the final graph.  Auditing even one MobileNet rewrite can take
     longer than an interactive export budget, so the interactive profiles
     reserve their work for that end-to-end report.  Non-interactive callers do
     not pass this option and keep the complete per-step audit trail. *)
  let max_verified_steps = function Quick | Standard | Thorough -> 0
  let max_clusters = function Quick -> 1 | Standard -> 4 | Thorough -> 16
  let pp fmt t = Fmt.string fmt (to_string t)
end

module Coverage = struct
  type t = Exhaustive | Not_applicable | Sampled of int

  let exhaustive = Exhaustive
  let not_applicable = Not_applicable

  let sampled n =
    if n >= 1 then Err.return (Sampled n) else Err.fail (`Invalid_coverage n)

  let pp fmt = function
    | Exhaustive -> Fmt.string fmt "exhaustive"
    | Not_applicable -> Fmt.string fmt "n/a"
    | Sampled n -> Fmt.pf fmt "sampled %d" n
end

module Strength = struct
  type proof = Constants | Structural
  type test = Agrees of float | Disagrees of Ground_expr.Valuation.t

  let pp_proof fmt = function
    | Constants -> Fmt.string fmt "structural, for these constants"
    | Structural -> Fmt.string fmt "structural"

  let pp_test fmt = function
    | Agrees tol -> Fmt.pf fmt "agrees (%g)" tol
    | Disagrees v ->
        Fmt.pf fmt "@[<h>disagrees at %a@]" Ground_expr.Valuation.pp v
end

module Refutation = struct
  type t =
    | Shape of {
        lhs : Member.Erased.t;
        lhs_shape : Vec6.shape;
        rhs : Member.Erased.t;
        rhs_shape : Vec6.shape;
      }
    | Value of {
        coord : Vec6.coord;
        lhs : Member.Erased.t;
        rhs : Member.Erased.t;
        valuation : Ground_expr.Valuation.t;
      }

  let pp fmt = function
    | Shape s ->
        Fmt.pf fmt "@[<h>shape %a:%a vs %a:%a@]" Member.Erased.pp s.lhs
          Vec6.pp_shape s.lhs_shape Member.Erased.pp s.rhs Vec6.pp_shape
          s.rhs_shape
    | Value v ->
        Fmt.pf fmt "@[<h>value at %a: %a vs %a under %a@]" Vec6.pp_coord v.coord
          Member.Erased.pp v.lhs Member.Erased.pp v.rhs Ground_expr.Valuation.pp
          v.valuation
end

module Unproved = struct
  type t =
    | Eval of Ground_eval.error
    | Exhausted of {
        coord : Vec6.coord;
        lhs : Member.Erased.t;
        rhs : Member.Erased.t;
      }
    | Max_nodes of int
    | Max_rounds
    | Max_clusters of int
    | Out_of_bounds of Ground_expr.Cell.t
    | Too_large of int
    | Unbound_constant of Ground_expr.Cell.t
    | Unsupported_format of {
        blocked : Ground_expr.Cell.t;
        member : Member.Erased.t;
      }
    | Unsupported_relation of Correspondence.relation

  let pp fmt = function
    | Eval e -> Fmt.pf fmt "@[<h>eval: %a@]" Ground_eval.pp_error e
    | Exhausted e ->
        Fmt.pf fmt "@[<h>exhausted at %a: %a vs %a@]" Vec6.pp_coord e.coord
          Member.Erased.pp e.lhs Member.Erased.pp e.rhs
    | Max_nodes n -> Fmt.pf fmt "over max_nodes (%d)" n
    | Max_rounds -> Fmt.string fmt "over max_rounds"
    | Max_clusters n ->
        Fmt.pf fmt "global verification budget exhausted (%d clusters)" n
    | Out_of_bounds c ->
        Fmt.pf fmt "@[<h>out of bounds: %a@]" Ground_expr.Cell.pp c
    | Too_large n -> Fmt.pf fmt "too large (%d coords)" n
    | Unbound_constant c ->
        Fmt.pf fmt "@[<h>unbound constant: %a@]" Ground_expr.Cell.pp c
    | Unsupported_format f ->
        Fmt.pf fmt "@[<h>format blocks collapse: %a in %a@]" Ground_expr.Cell.pp
          f.blocked Member.Erased.pp f.member
    | Unsupported_relation r ->
        Fmt.pf fmt "@[<h>unsupported relation: %a@]" Correspondence.pp_relation
          r

  (* Why, with the payload dropped, so verdicts can be counted by reason. *)
  let reason = function
    | Eval _ -> "grounding failed"
    | Exhausted _ -> "frontier exhausted"
    | Max_nodes _ -> "over max_nodes"
    | Max_rounds -> "over max_rounds"
    | Max_clusters _ -> "global verification budget exhausted"
    | Out_of_bounds _ -> "out of bounds"
    | Too_large _ -> "too large"
    | Unbound_constant _ -> "unbound constant"
    | Unsupported_format _ -> "format blocks collapse"
    | Unsupported_relation _ -> "unsupported relation"

  (* Every string [reason] can return, in constructor order. It has to be a
     literal list: each constructor carries a payload ([Ground_eval.error], a
     [Vec6.coord], a [Member.Erased.t]) that no enumeration can invent, so the
     alternative is sample values this module has no business owning. What keeps
     it honest is the exhaustiveness test beside it — [reason] applied to every
     constructor must land in here — which fails as soon as the two diverge. *)
  let reasons =
    [
      "grounding failed";
      "frontier exhausted";
      "over max_nodes";
      "over max_rounds";
      "global verification budget exhausted";
      "out of bounds";
      "too large";
      "unbound constant";
      "format blocks collapse";
      "unsupported relation";
    ]
end

module Verdict = struct
  type t =
    | Proved of Strength.proof
    | Refuted of Refutation.t
    | Tested of Strength.test
    | Unproved of Unproved.t
    | Vacuous

  let pp fmt = function
    | Proved p -> Fmt.pf fmt "proved (%a)" Strength.pp_proof p
    | Refuted r -> Fmt.pf fmt "refuted: %a" Refutation.pp r
    | Tested t -> Fmt.pf fmt "tested: %a" Strength.pp_test t
    | Unproved u -> Fmt.pf fmt "unproved: %a" Unproved.pp u
    | Vacuous -> Fmt.string fmt "vacuous"

  (* A payload-free label: the outcome and its reason, with ids, coordinates and
     valuations dropped so verdicts can be counted. Those belong in
     [pp_verdicts], which shows the individual cluster. *)
  let label = function
    | Proved Strength.Constants -> "proved (for these constants)"
    | Proved Strength.Structural -> "proved (structural)"
    | Refuted (Refutation.Shape _) -> "refuted (shape)"
    | Refuted (Refutation.Value _) -> "refuted (counterexample)"
    | Tested (Strength.Agrees _) -> "tested (coefficients agree)"
    | Tested (Strength.Disagrees _) -> "tested (disagrees)"
    | Unproved u -> "unproved (" ^ Unproved.reason u ^ ")"
    | Vacuous -> "vacuous"

  (* The finite set [label] can return, in the canonical order a decoder sorts
     by. [label]'s own contract is what makes this finite — it drops every
     payload — so this states a property that already holds rather than
     imposing a new one. Its consumer is [Pass.Outcome_counts.of_bindings],
     which without it could not tell a well-formed decoded label from an
     arbitrary string and would be trusting the wire. *)
  let labels =
    [
      "proved (for these constants)";
      "proved (structural)";
      "refuted (shape)";
      "refuted (counterexample)";
      "tested (coefficients agree)";
      "tested (disagrees)";
    ]
    @ List.map (fun r -> "unproved (" ^ r ^ ")") Unproved.reasons
    @ [ "vacuous" ]

  (* The WEAKER of two verdicts about the same cluster, since a cluster is only
     as verified as its least-verified coordinate. [Refuted] dominates
     everything: one counterexample settles the cluster however many other
     coordinates agreed. Ties keep the earlier verdict, so a report names the
     first coordinate that failed rather than the last. *)
  (* How weak a verdict is; higher wins a join. Total, so a join is never
     order-dependent — [Vacuous] and [Proved] used to tie, which made a node
     whose outputs were one of each print whichever came first. [Vacuous] sits
     lowest because it asserts nothing: a node with a vacuous output and a
     proved one is better described as proved.

     A witness outranks agreement: a cluster where one coordinate's coefficients
     agree and another has a disagreeing valuation is NOT "coefficients agree",
     which is what ranking them equal reported for a batch-norm cluster whose
     first channel agreed. *)
  let rank = function
    | Refuted _ -> 6
    | Unproved _ -> 5
    | Tested (Strength.Disagrees _) -> 4
    | Tested (Strength.Agrees _) -> 3
    | Proved Strength.Constants -> 2
    | Proved Strength.Structural -> 1
    | Vacuous -> 0

  let join a b = if rank b > rank a then b else a
end

(* Whether anything further could change a cluster's answer. Only a
   counterexample can settle it, and only because it is already the weakest
   verdict there is.

   A budget verdict deliberately does NOT stop the traversal, though it is
   tempting: [max_nodes] and [max_rounds] are spent per COMPARISON and reset for
   the next one, so a deep member or coordinate exhausting its budget says
   nothing about a shallower one that might still produce a witness. (The one
   genuinely cluster-global budget, [Too_large], is decided before the traversal
   starts and never reaches here.) *)
let settled = function Verdict.Refuted _ -> true | _ -> false

module Outcome = struct
  type t = { coverage : Coverage.t; verdict : Verdict.t }

  (* The weaker outcome ENTIRE, verdict and coverage together. Joining the two
     independently pairs a verdict with coverage from a different outcome — an
     [Unproved Too_large], whose coverage is [Not_applicable] because nothing
     was examined, could come out marked [sampled n] from a sibling. It also
     leaves the result order-dependent wherever two verdicts rank equal. *)
  let join a b =
    if Verdict.rank b.verdict > Verdict.rank a.verdict then b else a

  (* The counting key: the verdict's payload-free label, suffixed with the
     sample count when there is one. Factored out of [Tally.of_entries]
     UNCHANGED — it was inline there, and [Pass.Outcome_counts] needs the same
     key, so leaving it inline would have meant the format ("<verdict>
     [sampled n]") existing twice and drifting silently. One function, two
     callers, no vocabulary to keep in step. *)
  let label t =
    match t.coverage with
    | Coverage.Sampled n ->
        Fmt.str "%s [sampled %d]" (Verdict.label t.verdict) n
    | Coverage.Exhaustive | Coverage.Not_applicable -> Verdict.label t.verdict

  let pp fmt t =
    match t.coverage with
    | Coverage.Not_applicable -> Verdict.pp fmt t.verdict
    | Coverage.Exhaustive | Coverage.Sampled _ ->
        Fmt.pf fmt "%a [%a]" Verdict.pp t.verdict Coverage.pp t.coverage
end

(* Where a cluster sits in the destination's structural hierarchy. The IR's
   groups are what a reader recognises — "layer1.0", "features.3" — so a report
   over a real model is only legible when its clusters are attributed to them.
   The root is [[]]. *)
module Group_path = struct
  type t = string list

  let compare = List.compare String.compare
  let to_string = function [] -> "(root)" | path -> String.concat "/" path
  let pp fmt t = Fmt.string fmt (to_string t)

  (* Node -> path, from the group tree. An unlabelled group is named by its id,
     so a path is always printable. *)
  let index (g : 'op Graph_common.Graph.t) =
    let rec walk path (group : Graph_ir.Group.t) acc =
      let path =
        match group.Graph_ir.Group.label with
        | Some l -> path @ [ l ]
        | None ->
            if group.Graph_ir.Group.id = g.Graph.root.Graph_ir.Group.id then
              path
            else path @ [ Fmt.str "g%a" Group_id.pp group.Graph_ir.Group.id ]
      in
      List.fold_left
        (fun acc -> function
          | Graph_ir.Group.Group sub -> walk path sub acc
          | Graph_ir.Group.Node id -> Node_id.Map.add id path acc)
        acc group.Graph_ir.Group.items
    in
    walk [] g.Graph.root Node_id.Map.empty

  (* Which node produced an edge, so a cluster can be placed. Keyed by
     DESTINATION edge, so the rule below is a type rather than a comment. *)
  let producers ~(edge : Tensor_id.t -> 'dst Correspondence.id option)
      (g : 'op Graph_common.Graph.t) =
    List.fold_left
      (fun acc (n : 'op Graph_common.Node.t) ->
        (* An output with no destination edge contributes no producer. *)
        List.filter_map edge n.Node.outputs
        |> List.fold_left
             (fun acc e -> Correspondence.Map.update e n.Node.id acc)
             acc)
      Correspondence.Map.empty g.Graph_common.Graph.nodes

  (* A cluster is placed by its DESTINATION edges only — that is the graph whose
     group tree this is. An edge with no producer is a graph input, which belongs
     to no group and lands at the root, and so does a cluster with no destination
     edge at all (a deletion).

     Falling back to the source ids would resolve them through the destination's
     producer map, which is a different id universe: a source id that happens to
     collide numerically with an unrelated destination edge would file the
     cluster under that edge's group. Wrong attribution is worse than the root,
     because it reads as a real answer. [producers] is keyed by ['dst] edge, so
     that fallback no longer type-checks. *)
  let of_cluster ~index ~producers (c : ('src, 'dst) Correspondence.Cluster.t) =
    Option.value ~default:[]
      (Correspondence.Set.fold
         (fun e acc ->
           match acc with
           | Some _ -> acc
           | None ->
               Option.bind (Correspondence.Map.find_opt e producers)
                 (fun node -> Node_id.Map.find_opt node index))
         c.dst None)
end

module Entry = struct
  type t = {
    cluster : Correspondence.Cluster.Erased.t;
    group : Group_path.t;
    outcome : Outcome.t;
  }
end

(* Counts by outcome label, in a deterministic order. What a summary is FOR:
   "which of these came back proved, and for the rest, why". *)
module Tally = struct
  module By_label = Map.Make (String)

  type t = int By_label.t

  let of_entries entries =
    List.fold_left
      (fun acc (e : Entry.t) ->
        (* The key is [Outcome.label] — the same function [Pass.Outcome_counts]
           uses, which is what keeps the two vocabularies one. *)
        let key = Outcome.label e.outcome in
        By_label.update key
          (function None -> Some 1 | Some n -> Some (n + 1))
          acc)
      By_label.empty entries

  let bindings t = By_label.bindings t
  let total t = By_label.fold (fun _ n acc -> acc + n) t 0

  let pp fmt t =
    Fmt.pf fmt "@[<v>%a@]"
      (Fmt.list ~sep:Fmt.cut (fun fmt (label, n) ->
           Fmt.pf fmt "%5d  %s" n label))
      (bindings t)
end

module Report = struct
  type t = { entries : Entry.t list }

  let proved t =
    List.for_all
      (fun (e : Entry.t) ->
        match (e.outcome.Outcome.verdict, e.outcome.Outcome.coverage) with
        | Verdict.Proved _, Coverage.Exhaustive -> true
        | Verdict.Vacuous, _ -> true
        | _ -> false)
      t.entries

  let refuted t =
    List.exists
      (fun (e : Entry.t) ->
        match e.outcome.Outcome.verdict with
        | Verdict.Refuted _ -> true
        | _ -> false)
      t.entries

  let tally t = Tally.of_entries t.entries

  (* Grouped, in [Group_path] order so a golden is stable. *)
  let by_group t =
    let module M = Map.Make (Group_path) in
    List.fold_left
      (fun acc (e : Entry.t) ->
        M.update e.Entry.group
          (function None -> Some [ e ] | Some es -> Some (e :: es))
          acc)
      M.empty t.entries
    |> M.bindings
    |> List.map (fun (path, es) -> (path, Tally.of_entries (List.rev es)))

  let summary t =
    let tally = tally t in
    Fmt.str "@[<h>%d clusters: %a@]" (Tally.total tally)
      (Fmt.list ~sep:(Fmt.any ", ") (fun fmt (label, n) ->
           Fmt.pf fmt "%d %s" n label))
      (Tally.bindings tally)

  let pp fmt t = Fmt.string fmt (summary t)
  let pp_summary fmt t = Tally.pp fmt (tally t)

  let pp_groups fmt t =
    Fmt.pf fmt "@[<v>%a@]"
      (Fmt.list ~sep:Fmt.cut (fun fmt (path, tally) ->
           Fmt.pf fmt "@[<v 2>%a@,%a@]" Group_path.pp path Tally.pp tally))
      (by_group t)

  let pp_verdicts fmt t =
    Fmt.pf fmt "@[<v>%a@]"
      (Fmt.list ~sep:Fmt.cut (fun fmt (e : Entry.t) ->
           Fmt.pf fmt "%a: %a" Correspondence.Cluster.Erased.pp e.Entry.cluster
             Outcome.pp e.Entry.outcome))
      t.entries
end

module Policy = struct
  type t = Reject_refuted | Require_proved

  let accepts t report =
    match t with
    | Reject_refuted -> not (Report.refuted report)
    | Require_proved -> Report.proved report

  let pp fmt = function
    | Reject_refuted -> Fmt.string fmt "reject_refuted"
    | Require_proved -> Fmt.string fmt "require_proved"
end

type error = [ Graph_map.error | `Missing_signature of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | #Graph_map.error as e -> Graph_map.pp_error fmt e
  | `Missing_signature id ->
      Fmt.pf fmt "missing signature for %a" Tensor_id.pp id
