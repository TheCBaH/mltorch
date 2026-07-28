(* See map_verify.mli. *)

open Graph_ir

module Member = struct
  type t = Dst of Tensor_id.t | Src of Tensor_id.t

  let id = function Dst id -> id | Src id -> id
  let tag = function Dst _ -> 0 | Src _ -> 1

  let compare a b =
    match Stdlib.compare (tag a) (tag b) with
    | 0 -> Tensor_id.compare (id a) (id b)
    | n -> n

  let pp fmt m =
    Fmt.pf fmt "%s%a"
      (match m with Dst _ -> "dst." | Src _ -> "src.")
      Tensor_id.pp (id m)
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

  (* Structural equality of the ground terms — [Round]s included, constants
     compared bitwise — means the two edges compute the same bits, which
     discharges every relation in the lattice. [Unverifiable] is excluded
     earlier, as it asserts nothing to discharge. *)
  let proves _ = function
    | Correspondence.Unverifiable -> false
    | Correspondence.Approximate _ | Correspondence.Equivalent
    | Correspondence.Identical ->
        true

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
        lhs : Member.t;
        lhs_shape : Vec6.shape;
        rhs : Member.t;
        rhs_shape : Vec6.shape;
      }
    | Value of {
        coord : Vec6.coord;
        lhs : Member.t;
        rhs : Member.t;
        valuation : Ground_expr.Valuation.t;
      }

  let pp fmt = function
    | Shape s ->
        Fmt.pf fmt "@[<h>shape %a:%a vs %a:%a@]" Member.pp s.lhs Vec6.pp_shape
          s.lhs_shape Member.pp s.rhs Vec6.pp_shape s.rhs_shape
    | Value v ->
        Fmt.pf fmt "@[<h>value at %a: %a vs %a under %a@]" Vec6.pp_coord v.coord
          Member.pp v.lhs Member.pp v.rhs Ground_expr.Valuation.pp v.valuation
end

module Unproved = struct
  type t =
    | Eval of Ground_eval.error
    | Exhausted of { coord : Vec6.coord; lhs : Member.t; rhs : Member.t }
    | Max_nodes of int
    | Max_rounds
    | Out_of_bounds of Ground_expr.Cell.t
    | Too_large of int
    | Unsupported_format of { blocked : Ground_expr.Cell.t; member : Member.t }
    | Unsupported_relation of Correspondence.relation

  let pp fmt = function
    | Eval e -> Fmt.pf fmt "@[<h>eval: %a@]" Ground_eval.pp_error e
    | Exhausted e ->
        Fmt.pf fmt "@[<h>exhausted at %a: %a vs %a@]" Vec6.pp_coord e.coord
          Member.pp e.lhs Member.pp e.rhs
    | Max_nodes n -> Fmt.pf fmt "over max_nodes (%d)" n
    | Max_rounds -> Fmt.string fmt "over max_rounds"
    | Out_of_bounds c ->
        Fmt.pf fmt "@[<h>out of bounds: %a@]" Ground_expr.Cell.pp c
    | Too_large n -> Fmt.pf fmt "too large (%d coords)" n
    | Unsupported_format f ->
        Fmt.pf fmt "@[<h>format blocks collapse: %a in %a@]" Ground_expr.Cell.pp
          f.blocked Member.pp f.member
    | Unsupported_relation r ->
        Fmt.pf fmt "@[<h>unsupported relation: %a@]" Correspondence.pp_relation
          r
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

  (* The WEAKER of two verdicts about the same cluster, since a cluster is only
     as verified as its least-verified coordinate — so a coordinate that needed
     bound constants is not overwritten by a later one that proved structurally.
     [Refuted] dominates: one counterexample settles the cluster however many
     other coordinates agreed. Ties keep the earlier verdict, so a report names
     the first coordinate that failed rather than the last. *)
  let join a b =
    let rank = function
      | Refuted _ -> 5
      | Unproved _ -> 4
      (* A witness outranks agreement: a cluster where one coordinate's
         coefficients agree and another has a disagreeing valuation is NOT
         "coefficients agree". Ranking them equal and keeping the earlier
         verdict reported exactly that for a batch-norm cluster whose first
         channel agreed. *)
      | Tested (Strength.Disagrees _) -> 3
      | Tested (Strength.Agrees _) -> 2
      | Proved Strength.Constants -> 1
      | Proved Strength.Structural -> 0
      | Vacuous -> 0
    in
    if rank b > rank a then b else a
end

(* Whether anything further could change a cluster's answer. A counterexample is
   final; so is running out of budget, because no later coordinate can be probed
   either and re-deriving the same budget verdict at every remaining coordinate
   is pure cost. Every other verdict keeps looking — a later coordinate may hold
   the counterexample this one could not find. *)
let settled = function
  | Verdict.Refuted _ -> true
  | Verdict.Unproved
      (Unproved.Max_nodes _ | Unproved.Max_rounds | Unproved.Too_large _) ->
      true
  | _ -> false

module Outcome = struct
  type t = { coverage : Coverage.t; verdict : Verdict.t }

  let pp fmt t =
    match t.coverage with
    | Coverage.Not_applicable -> Verdict.pp fmt t.verdict
    | Coverage.Exhaustive | Coverage.Sampled _ ->
        Fmt.pf fmt "%a [%a]" Verdict.pp t.verdict Coverage.pp t.coverage
end

module Report = struct
  type t = { clusters : (Correspondence.Cluster.t * Outcome.t) list }

  let proved t =
    List.for_all
      (fun (_, (o : Outcome.t)) ->
        match (o.verdict, o.coverage) with
        | Verdict.Proved _, Coverage.Exhaustive -> true
        | Verdict.Vacuous, _ -> true
        | _ -> false)
      t.clusters

  let refuted t =
    List.exists
      (fun (_, (o : Outcome.t)) ->
        match o.verdict with Verdict.Refuted _ -> true | _ -> false)
      t.clusters

  let summary t =
    let count f = List.length (List.filter f t.clusters) in
    let is p (_, (o : Outcome.t)) = p o.verdict in
    Printf.sprintf
      "%d proved, %d refuted, %d tested, %d unproved, %d vacuous of %d"
      (count (is (function Verdict.Proved _ -> true | _ -> false)))
      (count (is (function Verdict.Refuted _ -> true | _ -> false)))
      (count (is (function Verdict.Tested _ -> true | _ -> false)))
      (count (is (function Verdict.Unproved _ -> true | _ -> false)))
      (count (is (function Verdict.Vacuous -> true | _ -> false)))
      (List.length t.clusters)

  let pp fmt t = Fmt.string fmt (summary t)

  let pp_verdicts fmt t =
    Fmt.pf fmt "@[<v>%a@]"
      (Fmt.list ~sep:Fmt.cut (fun fmt (c, o) ->
           Fmt.pf fmt "%a: %a" Correspondence.Cluster.pp c Outcome.pp o))
      t.clusters
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

(* ---- sigma over graph inputs ---------------------------------------------

   Renaming both sides of an INTERNAL cluster to a representative would assume
   the very claim under verification; that is only sound under an induction
   over a topological order of the cluster DAG, which two graphs quotiented by
   a correspondence can in principle make cyclic. Graph inputs are different in
   kind: "corresponding inputs are fed the same data" is the hypothesis, not an
   obligation. It matters because [Rewrite.pack] renumbers input ids. *)
let input_renames clusters ~src ~dst =
  let inputs g = Tensor_id.Set.of_list g.Graph.inputs in
  let src_inputs = inputs src and dst_inputs = inputs dst in
  List.fold_left
    (fun (smap, dmap) (c : Correspondence.Cluster.t) ->
      let all = Tensor_id.Set.union c.src c.dst in
      match Tensor_id.Set.min_elt_opt all with
      | None -> (smap, dmap)
      | Some rep ->
          let add keep set map =
            Tensor_id.Set.fold
              (fun id map ->
                if Tensor_id.Set.mem id keep then Tensor_id.Map.add id rep map
                else map)
              set map
          in
          (add src_inputs c.src smap, add dst_inputs c.dst dmap))
    (Tensor_id.Map.empty, Tensor_id.Map.empty)
    clusters

let rename_with map id =
  match Tensor_id.Map.find_opt id map with Some rep -> rep | None -> id

(* ---- one cluster ---------------------------------------------------------- *)

(* Two envs per graph. [env] leaves model constants as free cells, so a proof
   through it quantifies over every payload; [with_constants] substitutes them,
   which is strictly weaker and therefore only tried when the first fails. *)
type side = {
  env : Ground_eval.Env.t;
  graph : graph;
  with_constants : Ground_eval.Env.t;
}

let shape_of side id =
  match Tensor_id.Map.find_opt id side.graph.Graph.tensors with
  | Some (sg : Tensor_sig.t) -> Core.return sg.Tensor_sig.shape
  | None -> Core.fail (`Missing_signature id)

let side_of sides member =
  match member with Member.Src _ -> fst sides | Member.Dst _ -> snd sides

(* Deepen until the two terms agree, or until there is nothing left to expand
   or no rounds left. Structural equality is tried BEFORE normalising and again
   after: two identical terms carrying the same uncollapsible [Round (Cell _)]
   are equal and must be proved, not rejected for the blocked collapse. *)
let rec settle ~budget ~probe ~tolerance ~label ~proof ~rounds ~lhs ~rhs
    ~lhs_env ~rhs_env ~coord ~members =
  let lhs_member, rhs_member = members in
  if Ground_expr.equal lhs rhs then Verdict.Proved proof
  else
    let ln =
      Ground_expr.normalise ~stored_f32:(Ground_eval.Env.stored_f32 lhs_env) lhs
    and rn =
      Ground_expr.normalise ~stored_f32:(Ground_eval.Env.stored_f32 rhs_env) rhs
    in
    if Ground_expr.equal ln.expr rn.expr then Verdict.Proved proof
    else
      let expandable =
        Ground_eval.expandable lhs_env lhs || Ground_eval.expandable rhs_env rhs
      in
      if expandable && rounds >= budget.Budget.max_rounds then
        Verdict.Unproved Unproved.Max_rounds
      else if expandable then
        let cap = budget.Budget.max_nodes in
        let lhs = Ground_eval.expand ~budget:cap lhs_env lhs
        and rhs = Ground_eval.expand ~budget:cap rhs_env rhs in
        let size = Ground_expr.size lhs + Ground_expr.size rhs in
        if size > budget.Budget.max_nodes then
          Verdict.Unproved (Unproved.Max_nodes size)
        else
          settle ~budget ~probe ~tolerance ~label ~proof ~rounds:(rounds + 1)
            ~lhs ~rhs ~lhs_env ~rhs_env ~coord ~members
      else
        (* Frontier is at the graph inputs and the terms still differ. That is
           the prover failing, not a counterexample: no assignment has been
           exhibited. A blocked collapse is a likelier explanation than a real
           difference, so report it when one is present. *)
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
        | None, None ->
            (* The frontier is at the graph inputs, so every remaining cell is
               genuinely free and a disagreeing assignment is realisable. This
               is the only place a value counterexample may be built. *)
            let witness () =
              match counterexample ~probe ~lhs ~rhs with
              | None ->
                  Verdict.Unproved
                    (Unproved.Exhausted
                       { coord; lhs = lhs_member; rhs = rhs_member })
              | Some valuation ->
                  if label = Correspondence.Identical then
                    Verdict.Refuted
                      (Refutation.Value
                         {
                           coord;
                           lhs = lhs_member;
                           rhs = rhs_member;
                           valuation;
                         })
                  else Verdict.Tested (Strength.Disagrees valuation)
            in
            (* The NORMALISED terms, not the raw ones: folding is what turns
               [sqrt (Const _)] — batch norm's normaliser — into a coefficient
               rather than an opaque generator the polynomial view cannot see
               through. *)
            if Coeff_form.agree ~tolerance ln.expr rn.expr then
              (* Coefficient agreement is never a proof, and for [Identical] it
                 is not even the right question — that claim is about bits, so a
                 probe still gets to refute it. *)
              if label = Correspondence.Identical then
                match witness () with
                | Verdict.Unproved _ ->
                    Verdict.Tested (Strength.Agrees tolerance)
                | refuted -> refuted
              else Verdict.Tested (Strength.Agrees tolerance)
            else witness ()

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
let compare_at ~budget ~probe ~tolerance ~label sides ~canonical ~other coord =
  let lhs_side = side_of sides canonical and rhs_side = side_of sides other in
  let attempt proof lhs_env rhs_env =
    match
      ( Ground_eval.at lhs_env (Member.id canonical) coord,
        Ground_eval.at rhs_env (Member.id other) coord )
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
              ~lhs_env ~rhs_env ~coord ~members:(canonical, other))
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
  match attempt Strength.Structural lhs_side.env rhs_side.env with
  | Verdict.Proved _ as proved -> proved
  | _ ->
      attempt Strength.Constants lhs_side.with_constants rhs_side.with_constants

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
   ten are selected while the report still says "sampled 8" — and divides by
   zero at [n = 0]. *)
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

let check_members ~budget ~probe ~tolerance ~label sides ~canonical ~others
    ~shape =
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
          (compare_at ~budget ~probe ~tolerance ~label sides ~canonical ~other
             coord)
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

let check_cluster ~budget ~probe ~tolerance sides (c : Correspondence.Cluster.t)
    =
  let members =
    List.map (fun id -> Member.Src id) (Tensor_id.Set.elements c.src)
    @ List.map (fun id -> Member.Dst id) (Tensor_id.Set.elements c.dst)
  in
  let outcome verdict coverage = Core.return { Outcome.coverage; verdict } in
  if Tensor_id.Set.is_empty c.src || Tensor_id.Set.is_empty c.dst then
    outcome Verdict.Vacuous Coverage.Not_applicable
  else if c.label = Correspondence.Unverifiable then
    outcome (Verdict.Unproved (Unproved.Unsupported_relation c.label))
      Coverage.Not_applicable
  else
    let open Core.Syntax in
    match members with
    | [] -> outcome Verdict.Vacuous Coverage.Not_applicable
    | canonical :: others -> (
        let shape_for m = shape_of (side_of sides m) (Member.id m) in
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
                      lhs = canonical;
                      lhs_shape = shape;
                      rhs = m;
                      rhs_shape = s;
                    }))
              Coverage.Not_applicable
        | None ->
            let verdict, coverage =
              check_members ~budget ~probe ~tolerance ~label:c.label sides
                ~canonical ~others ~shape
            in
            outcome verdict coverage)

(* ---- entry points --------------------------------------------------------- *)

let default_coefficient_tolerance = 1e-5

let run ?(budget = Budget.default)
    ?(coefficient_tolerance = default_coefficient_tolerance) ?(probe = 4)
    ?(src_constants = Tensor_id.Map.empty)
    ?(dst_constants = Tensor_id.Map.empty) map ~src ~dst =
  let open Core.Syntax in
  let* () = (Graph_map.validate map ~src ~dst :> (unit, error) Core.result) in
  let clusters = Graph_map.clusters_over map ~src ~dst in
  let src_map, dst_map = input_renames clusters ~src ~dst in
  let side graph rename constants =
    let program = Eval_symbolic.run graph and rename = rename_with rename in
    {
      env = Ground_eval.Env.of_program program ~rename;
      graph;
      with_constants = Ground_eval.Env.of_program ~constants program ~rename;
    }
  in
  let sides =
    (side src src_map src_constants, side dst dst_map dst_constants)
  in
  let* outcomes =
    Core.List.map
      (check_cluster ~budget ~probe ~tolerance:coefficient_tolerance sides)
      clusters
  in
  Core.return { Report.clusters = List.combine clusters outcomes }

let step ?budget ?coefficient_tolerance ?probe before
    (Rewrite.Step (after, map)) =
  run ?budget ?coefficient_tolerance ?probe map ~src:(Rewrite.graph before)
    ~src_constants:(Rewrite.constants before) ~dst:(Rewrite.graph after)
    ~dst_constants:(Rewrite.constants after)
