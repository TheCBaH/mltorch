(* See map_verify.mli. *)

open Graph_ir

module Member = struct
  (* A GADT rather than a pair of [Tensor_id.t]s, which DELETES the [id]
     accessor instead of typing it: [id] erased exactly the distinction this
     type exists to keep, and map_verify.ml:316 was a source id resolved in the
     destination's producer map. Reaching an id now means going through
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
  let pp fmt t = Fmt.string fmt (to_string t)
end

module Coverage = struct
  type t = Exhaustive | Not_applicable | Sampled of int

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
    | Out_of_bounds _ -> "out of bounds"
    | Too_large _ -> "too large"
    | Unbound_constant _ -> "unbound constant"
    | Unsupported_format _ -> "format blocks collapse"
    | Unsupported_relation _ -> "unsupported relation"
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
          | Graph_ir.Group.Node id -> Node_id.Map.add id path acc
          | Graph_ir.Group.Group sub -> walk path sub acc)
        acc group.Graph_ir.Group.items
    in
    walk [] g.Graph.root Node_id.Map.empty

  (* Which node produced an edge, so a cluster can be placed. Keyed by
     DESTINATION edge, so the rule below is a type rather than a comment. *)
  let producers ~(edge : Tensor_id.t -> 'dst Correspondence.id option)
      (g : 'op Graph_common.Graph.t) =
    List.fold_left
      (fun acc (n : 'op Graph_common.Node.t) ->
        List.fold_left
          (fun acc out ->
            match edge out with
            | Some e -> Correspondence.Map.update e n.Node.id acc
            | None -> acc)
          acc n.Node.outputs)
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
        let key =
          match e.outcome.Outcome.coverage with
          | Coverage.Sampled n ->
              Fmt.str "%s [sampled %d]"
                (Verdict.label e.outcome.Outcome.verdict)
                n
          | Coverage.Exhaustive | Coverage.Not_applicable ->
              Verdict.label e.outcome.Outcome.verdict
        in
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
  | `Missing_signature id ->
      Fmt.pf fmt "missing signature for %a" Tensor_id.pp id
  | #Graph_map.error as e -> Graph_map.pp_error fmt e

(* ---- the local frontier ---------------------------------------------------

   Each cluster is a LOCAL obligation: does this transformation compute the same
   function of its dependencies, given that those dependencies correspond? So a
   cell is replaced by its cluster's variable — the two graphs' names for one
   value become one name — and the obligation is then universally quantified
   over the variables that remain.

   Two edges do not get one:

   - a member of the cluster UNDER TEST, unless it is a user-data graph input.
     Otherwise the cluster names both its sides the same thing and discharges
     itself. Its non-input members keep expanding through their producers
     instead;
   - an edge of a vacuous cluster — a creation or a deletion — which relates one
     side to nothing and so names no shared value. Those are expanded through,
     which is how a rewrite's own scratch edges are crossed.

   The exception for user inputs is σ, "corresponding inputs are fed the same
   data": a hypothesis rather than an obligation, and the only one here. It is
   also what keeps a trim provable — {t0,t1} -> {t0} needs src.t0 and dst.t0 to
   be one variable while src.t1 expands through its load of t0.

   .ai/native_transform_verify.md §7a has the soundness argument: local proofs
   compose because structural equality forces both sides to mention the same
   variables, so the quotient dependency relation embeds in each graph's own DAG
   order and the induction over it is well-founded. *)

let boundary_of ~index ~under_test ~lookup ~env origin =
  match Ground_expr.Origin.edge origin with
  | None -> None (* already projected; nothing to decide *)
  | Some id ->
      Option.bind (lookup index id) (fun v ->
          (* [under_test] is [None] only for a VACUOUS cluster, and
             [Boundary_index] records no member of one — so [lookup] has already
             answered [None] above and this arm cannot then grant a variable. *)
          let discharges_itself =
            Option.fold ~none:false ~some:(Cluster_var.equal v) under_test
          in
          if discharges_itself && not (Ground_eval.Env.is_user_input env id)
          then None
          else Some v)

(* ---- one cluster ---------------------------------------------------------- *)

(* Two envs per graph. [env] leaves model constants as free cells, so a proof
   through it quantifies over every payload; [with_constants] substitutes them,
   which is strictly weaker and therefore only tried when the first fails.

   Indexed by the version it belongs to, which is what lets [resolve] below
   insist that a member and the side it is looked up in agree. *)
type 'v side = {
  env : Ground_eval.Env.t;
  (* The SIGNATURE LOOKUP, not the snapshot. That one substitution is what makes
     this record dialect-free: [shape_for] was its only reader, and the two
     dialects have different snapshot types but the same [Tensor_sig.t]. The
     universe keeps ['v] a real parameter, so [resolve]'s rank-2 field still
     refuses to pair a [Dst] member with the source side. *)
  sig_of : Tensor_id.t -> Tensor_sig.t option;
  edges : 'v Correspondence.Universe.t;
      (* Never read. It is here to keep ['v] a real parameter of the record:
         with the snapshot gone, nothing else mentions the version, and an
         unparameterised [side] would let [resolve]'s rank-2 field pair a [Dst]
         member with the source side — the bug the rank-2 exists to prevent. *)
  with_constants : Ground_eval.Env.t;
}
[@@warning "-69"]

type ('src, 'dst) sides = { dst : 'dst side; src : 'src side }

(* RANK-2 on purpose. [f] is polymorphic in the version, so it can only be
   applied to a side and an edge that agree — which makes pairing [Dst] with the
   source side a type error rather than the bug at map_verify.ml:316 written
   again. The result is version-free, so nothing leaks back out.

   [side_of] cannot exist in its place: "the side this member belongs to" has a
   constructor-dependent type, and a function returning it would have to pick one
   version for both arms. *)
type 'a resolve = { f : 'v. 'v side -> 'v Correspondence.id -> 'a }

let resolve : type s d. (s, d) sides -> (s, d) Member.t -> 'a resolve -> 'a =
 fun sides member r ->
  match member with
  | Member.Dst e -> r.f sides.dst e
  | Member.Src e -> r.f sides.src e

(* Everything a comparison needs from a member, with the version discharged. *)
type located = {
  loc_boundary : Ground_expr.Origin.t -> Cluster_var.t option;
  loc_env : Ground_eval.Env.t;
  loc_id : Tensor_id.t;
  loc_with_constants : Ground_eval.Env.t;
}

(* [under_test] is this cluster's own variable, which its non-input members are
   denied. The side picks which half of the index to read — a comparison can
   legitimately run between two SOURCE members, so this is not "lhs vs rhs". *)
let locate ~index ~under_test sides member =
  let lookup =
    match (Member.erase member).Member.Erased.side with
    | `Dst -> Boundary_index.dst
    | `Src -> Boundary_index.src
  in
  resolve sides member
    {
      f =
        (fun side e ->
          {
            (* [side.env] and [side.with_constants] agree on which ids are user
               inputs — binding payloads changes what a constant grounds to, not
               what kind it is — so either serves here. *)
            loc_boundary = boundary_of ~index ~under_test ~lookup ~env:side.env;
            loc_env = side.env;
            loc_id = Correspondence.raw e;
            loc_with_constants = side.with_constants;
          });
    }

let shape_for sides member =
  resolve sides member
    {
      f =
        (fun side e ->
          let id = Correspondence.raw e in
          match side.sig_of id with
          | Some (sg : Tensor_sig.t) -> Core.return sg.Tensor_sig.shape
          | None -> Core.fail (`Missing_signature id));
    }

(* The correspondence variables a PROJECTED term is a function of. Sorted and
   deduplicated so two sides can be compared as lists; there are a handful per
   obligation, so a set type of its own would not earn the module. *)
let frontier_vars e =
  Ground_expr.Cell.Set.fold
    (fun (c : Ground_expr.Cell.t) acc ->
      match c.Ground_expr.Cell.origin with
      | Ground_expr.Origin.Boundary v -> v :: acc
      | Ground_expr.Origin.Dst _ | Ground_expr.Origin.Src _ -> acc)
    (Ground_expr.cells e) []
  |> List.sort_uniq Cluster_var.compare

(* Deepen until the two terms agree, or until there is nothing left to expand
   or no rounds left. Structural equality is tried BEFORE normalising and again
   after: two identical terms carrying the same uncollapsible [Round (Cell _)]
   are equal and must be proved, not rejected for the blocked collapse.

   TWO REPRESENTATIONS, and keeping them apart is the whole of it. The RAW terms
   are side-qualified, and everything that asks a question about one graph reads
   them: [normalise], because [stored_f32] needs the real storage edge;
   [expand], because a stage belongs to a graph. The PROJECTED terms carry the
   map's claim that two edges name one value, and only comparison reads them.
   Projecting before normalising would strand every [Round] on a boundary cell,
   since an unknown cell is [stored_f32 = false] and refuses to collapse. *)
let rec settle ~budget ~probe ~tolerance ~label ~proof ~rounds ~lhs ~rhs
    ~lhs_env ~rhs_env ~lhs_boundary ~rhs_boundary ~coord ~members =
  let lhs_member, rhs_member = members in
  let seen () =
    ( Ground_expr.project ~boundary:lhs_boundary lhs,
      Ground_expr.project ~boundary:rhs_boundary rhs )
  in
  let projected_lhs, projected_rhs = seen () in
  if Ground_expr.equal projected_lhs projected_rhs then Verdict.Proved proof
  else
    let ln =
      Ground_expr.normalise ~stored_f32:(Ground_eval.Env.stored_f32 lhs_env) lhs
    and rn =
      Ground_expr.normalise ~stored_f32:(Ground_eval.Env.stored_f32 rhs_env) rhs
    in
    let projected_ln = Ground_expr.project ~boundary:lhs_boundary ln.expr
    and projected_rn = Ground_expr.project ~boundary:rhs_boundary rn.expr in
    if Ground_expr.equal projected_ln projected_rn then Verdict.Proved proof
    else
      (* RECONCILE THE FRONTIERS. A variable naming a value only one side reads
         is not a shared dependency at all: the obligation would read
         [forall v0 v1 v2. f(v0,v1) = g(v0,v2)], false for almost any f and g,
         and a probe assigning v1 and v2 independently separates a correct
         rewrite. [reuse_permute] is the case — the source reads t1 where the
         destination reads t3, and t3 = Q(t1) is exactly the fact a local
         frontier drops. So a one-sided variable is denied for EXPANSION and the
         cell is crossed instead, until both sides name the same variables or
         one runs out of producers.

         Denied for expansion only: [project] keeps the full boundary, or a
         shared variable would stop being unified and nothing would ever match.
         This terminates because each crossing strictly descends that graph's
         own DAG, the same argument §7 makes for σ.

         Once it settles, a variable still on one side alone belongs to a cell
         with no producer — a graph input, since an unbound constant is caught
         before the tiers. σ relates corresponding inputs, and these are in
         DIFFERENT clusters, so nothing constrains them to agree and a probe
         separating them is a genuine counterexample. That is why there is no
         "frontiers differ" verdict guarding the tiers below: crossing removes
         the case where a witness would have been spurious, and the case that
         remains deserves the refutation.
         See .ai/native_transform_local_verify_plan.md §13. *)
      let common =
        let lvars = frontier_vars projected_ln
        and rvars = frontier_vars projected_rn in
        fun v ->
          List.exists (Cluster_var.equal v) lvars
          && List.exists (Cluster_var.equal v) rvars
      in
      let crossing boundary o =
        match boundary o with Some v when common v -> Some v | _ -> None
      in
      let lhs_crossing = crossing lhs_boundary
      and rhs_crossing = crossing rhs_boundary in
      let expandable =
        Ground_eval.expandable ~boundary:lhs_crossing lhs_env lhs
        || Ground_eval.expandable ~boundary:rhs_crossing rhs_env rhs
      in
      if expandable && rounds >= budget.Budget.max_rounds then
        Verdict.Unproved Unproved.Max_rounds
      else if expandable then
        let cap = budget.Budget.max_nodes in
        let lhs =
          Ground_eval.expand ~boundary:lhs_crossing ~budget:cap lhs_env lhs
        and rhs =
          Ground_eval.expand ~boundary:rhs_crossing ~budget:cap rhs_env rhs
        in
        let size = Ground_expr.size lhs + Ground_expr.size rhs in
        if size > budget.Budget.max_nodes then
          Verdict.Unproved (Unproved.Max_nodes size)
        else
          settle ~budget ~probe ~tolerance ~label ~proof ~rounds:(rounds + 1)
            ~lhs ~rhs ~lhs_env ~rhs_env ~lhs_boundary ~rhs_boundary ~coord
            ~members
      else
        (* The LOCAL frontier is complete and the terms still differ. That is
           the prover failing, not a counterexample: no assignment has been
           exhibited. A blocked collapse is a likelier explanation than a real
           difference, so report it when one is present — and report the RAW
           cell, which names an edge a reader can go and look at, where the
           projected one would name a variable. *)
        match
          ( Ground_expr.Cell.Set.min_elt_opt ln.blocked,
            Ground_expr.Cell.Set.min_elt_opt rn.blocked )
        with
        | Some blocked, _ ->
            Verdict.Unproved
              (Unproved.Unsupported_format { blocked; member = lhs_member })
        | None, Some blocked ->
            Verdict.Unproved
              (Unproved.Unsupported_format { blocked; member = rhs_member })
        | None, None -> (
            (* Two reasons the value tiers below — coefficient agreement and the
               probe — must not run at all. Both are evidence ABOUT VALUES, so
               each is about there being no value question to answer, and both
               sit here rather than merely in front of the probe: coefficients
               over two unrelated variables disagree for the same non-reason a
               probe separates them.

               [Unverifiable] asserts nothing about values, so there is nothing
               to gather evidence for or against; structural equality was still
               worth trying, and did not close.

               An unbound model constant is free only because nobody supplied
               its payload. The tiers read a free cell as "may take any value",
               which would refute two constants that may hold identical bytes. *)
            match (label, unbound_constant_at ~lhs_env ~rhs_env ~lhs ~rhs) with
            | Correspondence.Unverifiable, _ ->
                Verdict.Unproved (Unproved.Unsupported_relation label)
            | _, Some cell -> Verdict.Unproved (Unproved.Unbound_constant cell)
            | _, None ->
                (* PROJECTED, so the two sides' names for one dependency are one
                   variable and a draw gives it one value. Over the raw terms a
                   probe would assign src.t2 and dst.t2 independently and
                   "separate" every corresponding pair there is. *)
                value_tiers ~probe ~tolerance ~label ~coord ~members
                  ~lhs:projected_ln ~rhs:projected_rn ~ln:projected_ln
                  ~rn:projected_rn)

(* The local frontier is complete, so every remaining cell is a free variable of
   this obligation and a disagreeing assignment refutes it. This is the only
   place a counterexample may be built — and it is a counterexample to the local
   TRANSFER FUNCTION, not to the two graphs' values: an upstream cluster may
   confine the variable to a range where the two sides agree. See
   .ai/native_transform_verify.md §8b.

   The frontier being SETTLED is what licenses that. [settle] crosses a variable
   only one side names (see its note), so by the time this runs the two sides are
   functions of the same variables and every remaining cell has no producer — a
   graph input, an unbound constant having been caught earlier. σ does not relate
   inputs in different clusters, so nothing constrains them to agree and a draw
   separating them is genuine.

   An earlier revision guarded these tiers instead, reporting "frontiers differ"
   rather than building a witness. That treated the symptom; crossing removes the
   cause, and the guard was measurably unreachable. See
   .ai/native_transform_local_verify_plan.md §13. *)
and value_tiers ~probe ~tolerance ~label ~coord ~members ~lhs ~rhs ~ln ~rn =
  let lhs_member, rhs_member = members in
  let witness () =
    match counterexample ~probe ~lhs ~rhs with
    | None ->
        Verdict.Unproved
          (Unproved.Exhausted { coord; lhs = lhs_member; rhs = rhs_member })
    | Some valuation ->
        if label = Correspondence.Identical then
          Verdict.Refuted
            (Refutation.Value
               { coord; lhs = lhs_member; rhs = rhs_member; valuation })
        else Verdict.Tested (Strength.Disagrees valuation)
  in
  (* The NORMALISED terms, not the raw ones: folding is what turns
     [sqrt (Const _)] — batch norm's normaliser — into a coefficient rather than
     an opaque generator the polynomial view cannot see through. *)
  if Coeff_form.agree ~tolerance ln rn then
    (* Coefficient agreement is never a proof, and for [Identical] it is not even
       the right question — that claim is about bits, so a probe still gets to
       refute it. *)
    if label = Correspondence.Identical then
      match witness () with
      | Verdict.Unproved _ -> Verdict.Tested (Strength.Agrees tolerance)
      | refuted -> refuted
    else Verdict.Tested (Strength.Agrees tolerance)
  else witness ()

(* A cell either side reads that is a model constant with no payload bound in
   its OWN env — each side is asked about its own cells, since a payload map is
   per-graph. [Set.find_first_opt] is not usable: it requires a monotone
   predicate, the same reason [Ground_eval.out_of_bounds] goes through a list. *)
and unbound_constant_at ~lhs_env ~rhs_env ~lhs ~rhs =
  let first env e =
    List.find_opt
      (Ground_eval.Env.unbound_constant env)
      (Ground_expr.Cell.Set.elements (Ground_expr.cells e))
  in
  match first lhs_env lhs with Some _ as c -> c | None -> first rhs_env rhs

(* Try [probe] draws over the union of both terms' free cells, bitwise because
   [Identical] is about bits. Returns the first assignment that separates them. *)
and counterexample ~probe ~lhs ~rhs =
  let cells =
    Ground_expr.Cell.Set.union (Ground_expr.cells lhs) (Ground_expr.cells rhs)
  in
  let rec draw n =
    if n >= probe then None
    else
      let v = Ground_expr.Valuation.draw n cells in
      let bits e = Int64.bits_of_float (Ground_expr.eval e v) in
      if Int64.equal (bits lhs) (bits rhs) then draw (n + 1) else Some v
  in
  draw 0

(* A grounding failure is a verdict ABOUT this cluster, not an error for the
   caller, so it is converted here rather than short-circuiting [run]. *)
let compare_at ~budget ~index ~probe ~tolerance ~label ~under_test sides
    ~canonical ~other coord =
  let lhs_at = locate ~index ~under_test sides canonical
  and rhs_at = locate ~index ~under_test sides other in
  (* Erased once, here: [settle] only ever REPORTS these. *)
  let members = (Member.erase canonical, Member.erase other) in
  let attempt proof lhs_env rhs_env =
    match
      ( Ground_eval.at lhs_env lhs_at.loc_id coord,
        Ground_eval.at rhs_env rhs_at.loc_id coord )
    with
    | Error e, _ | Ok _, Error e ->
        Verdict.Unproved (Unproved.Eval e.Core.Error.kind)
    | Ok lhs, Ok rhs -> (
        let out_of_bounds =
          match Ground_eval.out_of_bounds lhs_env lhs with
          | Some c -> Some c
          | None -> Ground_eval.out_of_bounds rhs_env rhs
        in
        match out_of_bounds with
        | Some c -> Verdict.Unproved (Unproved.Out_of_bounds c)
        | None ->
            settle ~budget ~probe ~tolerance ~label ~proof ~rounds:0 ~lhs ~rhs
              ~lhs_env ~rhs_env ~lhs_boundary:lhs_at.loc_boundary
              ~rhs_boundary:rhs_at.loc_boundary ~coord ~members)
  in
  (* Unqualified first, purely to STRENGTHEN: a proof with the constants left
     free is a statement about every payload, and binding them would silently
     narrow it to the model's own.

     Its failures are discarded, and that is a soundness requirement, not a
     preference. With constants unbound the probe would be free to assign a
     known weight any value it likes and "refute" a fold that is perfectly
     correct for the weight the model actually carries — the same manufactured
     counterexample that expanding a truncated frontier would produce. So every
     verdict other than [Proved] comes from the constant-bound attempt, where
     the cells still free really are free inputs. *)
  match attempt Strength.Structural lhs_at.loc_env rhs_at.loc_env with
  | Verdict.Proved _ as proved -> proved
  | _ ->
      attempt Strength.Constants lhs_at.loc_with_constants
        rhs_at.loc_with_constants

(* Compare every member against a canonical one. [{t0,t1} -> {t0}] must check
   t1, which is the trimmed edge the map's claim is actually about. Stops at the
   first non-[Proved] verdict, so the report names the coordinate that failed
   rather than the last one examined. *)

(* Exactly [min n numel] coordinates, spread evenly: position [i*numel/n] for
   [i] in [0, n). Deterministic because a sampled verdict has to be reproducible
   from a golden, and spread rather than clustered because the errors this looks
   for are index-shaped.

   Taking every [numel/n]-th coordinate instead would be off whenever [n] does
   not divide [numel] — at [numel = 10, n = 8] the stride rounds to 1 and all
   ten are selected while the report still says "sampled 8". *)
let sampled_coords shape n =
  let numel = (Vec6.numel shape :> int) in
  let n = Stdlib.max 1 (Stdlib.min n numel) in
  let wanted = List.init n (fun i -> i * numel / n) in
  let _, _, picked =
    Vec6.fold_coords shape ~init:(0, wanted, [])
      ~f:(fun (i, wanted, acc) coord ->
        match wanted with
        | w :: rest when w = i -> (i + 1, rest, coord :: acc)
        | _ -> (i + 1, wanted, acc))
  in
  (n, List.rev picked)

let check_members ~budget ~index ~probe ~tolerance ~label ~under_test sides
    ~canonical ~others ~shape =
  let numel = (Vec6.numel shape :> int) in
  if numel > budget.Budget.max_coords then
    (Verdict.Unproved (Unproved.Too_large numel), Coverage.Not_applicable)
  else
    let coverage, coords =
      match budget.Budget.sample with
      | Some n when n > 0 && numel > n ->
          let taken, coords = sampled_coords shape n in
          (Coverage.Sampled taken, Some coords)
      | _ -> (Coverage.Exhaustive, None)
    in
    (* Keep looking past an inconclusive coordinate. Stopping at the first
       non-[Proved] result would leave a later coordinate's counterexample
       unexamined, and [Reject_refuted] would then accept a broken rewrite on
       the strength of one coordinate it happened not to decide. *)
    let step other v coord =
      if settled v then v
      else
        Verdict.join v
          (compare_at ~budget ~index ~probe ~tolerance ~label ~under_test sides
             ~canonical ~other coord)
    in
    let over_coords other verdict =
      match coords with
      | Some coords -> List.fold_left (step other) verdict coords
      | None -> Vec6.fold_coords shape ~init:verdict ~f:(step other)
    in
    let verdict =
      List.fold_left
        (fun v other -> if settled v then v else over_coords other v)
        (Verdict.Proved Strength.Structural) others
    in
    (verdict, coverage)

let check_cluster ~budget ~index ~probe ~tolerance sides
    (c : ('src, 'dst) Correspondence.Cluster.t) =
  let members =
    List.map (fun e -> Member.Src e) (Correspondence.Set.elements c.src)
    @ List.map (fun e -> Member.Dst e) (Correspondence.Set.elements c.dst)
  in
  let outcome verdict coverage = Core.return { Outcome.coverage; verdict } in
  if Correspondence.Set.is_empty c.src || Correspondence.Set.is_empty c.dst then
    outcome Verdict.Vacuous Coverage.Not_applicable
  else
    (* [Unverifiable] is NOT short-circuited here. It asserts nothing about
       values, but structural equality is not a statement about values — it
       observes that the two sides compute the same term, which is exactly what
       an unchanged transfer function downstream of a value-destroying rewrite
       has to say for itself. [settle] declines the tiers below structural; see
       its [Unverifiable] arm. *)
    let open Core.Syntax in
    match members with
    | [] -> outcome Verdict.Vacuous Coverage.Not_applicable
    | canonical :: others -> (
        (* This cluster's own variable — the one its non-input members are
           denied. Any member resolves it, since [Boundary_index] gives every
           member of a cluster the same one. *)
        let under_test =
          let e = Member.erase canonical in
          match e.Member.Erased.side with
          | `Dst -> Boundary_index.dst index e.Member.Erased.id
          | `Src -> Boundary_index.src index e.Member.Erased.id
        in
        let shape_for m = shape_for sides m in
        let* shape = shape_for canonical in
        (* Shapes are checked BEFORE the coordinate budget, since [numel] is
           ambiguous when the members disagree on shape. *)
        let* mismatch =
          Core.List.fold_left
            (fun acc m ->
              match acc with
              | Some _ -> Core.return acc
              | None ->
                  let+ s = shape_for m in
                  if Stdlib.( = ) s shape then None else Some (m, s))
            None others
        in
        match mismatch with
        | Some (m, s) ->
            outcome
              (Verdict.Refuted
                 (Refutation.Shape
                    {
                      lhs = Member.erase canonical;
                      lhs_shape = shape;
                      rhs = Member.erase m;
                      rhs_shape = s;
                    }))
              Coverage.Not_applicable
        | None ->
            let verdict, coverage =
              check_members ~budget ~index ~probe ~tolerance ~label:c.label
                ~under_test sides ~canonical ~others ~shape
            in
            outcome verdict coverage)

(* ---- entry points --------------------------------------------------------- *)

let default_coefficient_tolerance = 1e-5

(* Parameterised over a PAIR of [Side.S], which is what makes a cross-dialect
   map checkable. Everything below the two [Stage_program.t]s is already
   dialect-free — the term language, the grounding, the coefficient tier, the
   probe — so the parameterisation reaches only as far as obtaining them.

   The one structural change this forced: [side] used to hold a [Snapshot.t],
   and with two dialects that type differs per side, so [('src,'dst) sides]
   would not typecheck. It holds a signature LOOKUP instead. [shape_for] was its
   only reader, and a [Tensor_sig.t] is a [Tensor_sig.t] in either dialect. *)
module Make_pair (Src : Side.S) (Dst : Side.S) = struct
  module Map_pair = Graph_map.Make_pair (Src) (Dst)

  let run ?(budget = Budget.default)
      ?(coefficient_tolerance = default_coefficient_tolerance) ?(probe = 4)
      ?(src_constants = Tensor_id.Map.empty)
      ?(dst_constants = Tensor_id.Map.empty) map ~(src : 'src Src.Snapshot.t)
      ~(dst : 'dst Dst.Snapshot.t) : (Report.t, error) Core.result =
    let open Core.Syntax in
    (* Endpoint validation now happens in [Graph_map.create], but closure does not
       survive [Graph_map.compose] — which takes no snapshots and so cannot
       re-check — and a composed map is exactly what a cumulative run receives. *)
    let* () =
      (Map_pair.check_claim_closure map ~src ~dst :> (unit, error) Core.result)
    in
    let clusters = Map_pair.clusters_over map ~src ~dst in
    (* CLUSTER MEMBERSHIP decides what a raw id may mean, and nothing else does.

       The two layers this replaces — a static comparison of the graphs' own
       definitions, plus a set of edges that became "shared" once their cluster was
       proved — both let a proof about one cluster travel to another. The static
       one keyed on raw-id equality, which is the false proof this line of work is
       named after: [src: t2 = add(a,b)] and [dst: t2 = sub(a,b)] made {t3} ↔ {t3}
       ground to [relu (cell t2)] on both sides and prove Identical. The dynamic
       one needed the cluster DAG to be acyclic, which two graphs quotiented by a
       correspondence need not be.

       Each cluster is now a LOCAL obligation whose dependencies are free
       variables, so neither layer is needed and neither can mislead: no ordering
       assumption, no cascade, and a cluster that fails cannot be papered over by
       one that passed. See .ai/native_transform_local_verify_plan.md §5-6. *)
    let index = Boundary_index.create clusters in
    (* One side, built from whatever that dialect supplies. The two dialects have
       different snapshot types, so this cannot be one polymorphic function over
       [Snapshot.t] the way it was — it is one per side, each closing over its own
       [Side.S]. *)
    let src_side =
      let program = Src.symbolic src in
      {
        env = Ground_eval.Env.of_program program ~side:`Src;
        sig_of = Src.sig_of src;
        edges = Src.Snapshot.edges src;
        with_constants =
          Ground_eval.Env.of_program ~constants:src_constants program ~side:`Src;
      }
    and dst_side =
      let program = Dst.symbolic dst in
      {
        env = Ground_eval.Env.of_program program ~side:`Dst;
        sig_of = Dst.sig_of dst;
        edges = Dst.Snapshot.edges dst;
        with_constants =
          Ground_eval.Env.of_program ~constants:dst_constants program ~side:`Dst;
      }
    in
    let sides = { dst = dst_side; src = src_side } in
    (* Cluster order is free, and that is the point. The destination-topological
       traversal this replaces existed so a cluster could lean on conclusions
       reached strictly earlier; a local obligation leans on nothing, so the
       clusters are checked in report order and each verdict stands alone. *)
    let* checked =
      Core.List.fold_left
        (fun acc c ->
          let+ outcome =
            check_cluster ~budget ~index ~probe ~tolerance:coefficient_tolerance
              sides c
          in
          outcome :: acc)
        [] clusters
    in
    let outcomes = List.rev checked in
    let dst_graph = Dst.Snapshot.graph dst in
    let index = Group_path.index dst_graph
    and producers =
      Group_path.producers ~edge:(Dst.Snapshot.edge dst) dst_graph
    in
    Core.return
      {
        Report.entries =
          List.map2
            (fun cluster outcome ->
              {
                Entry.cluster = Correspondence.Cluster.erase cluster;
                group = Group_path.of_cluster ~index ~producers cluster;
                outcome;
              })
            clusters outcomes;
      }
end

(* The Native-to-Native specialization: what every existing caller uses. *)
include Make_pair (Native_side) (Native_side)

let step ?budget ?coefficient_tolerance ?probe before
    (Rewrite.Step (after, map)) =
  run ?budget ?coefficient_tolerance ?probe map ~src:(Rewrite.snapshot before)
    ~src_constants:(Rewrite.constants before) ~dst:(Rewrite.snapshot after)
    ~dst_constants:(Rewrite.constants after)
