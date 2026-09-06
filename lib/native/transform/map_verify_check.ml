(* The verification algorithm proper: given a cluster and both sides'
   signature lookups, ground each member's transfer function and compare.
   Split out of map_verify.ml; see map_verify_types.ml for the
   Verdict/Outcome/... vocabulary this builds, and map_verify.ml for the
   Make_pair/run/step entry points that call check_cluster. *)

open Graph_ir
open Map_verify_types

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
          | Some (sg : Tensor_sig.t) -> Err.return sg.Tensor_sig.shape
          | None -> Err.fail (`Missing_signature id));
    }

(* The correspondence variables a PROJECTED term is a function of. Sorted and
   deduplicated so two sides can be compared as lists; there are a handful per
   obligation, so a set type of its own would not earn the module. *)
let frontier_vars e =
  Ground_expr.Cell.Set.fold
    (fun (c : Ground_expr.Cell.t) acc ->
      match c.Ground_expr.Cell.origin with
      | Ground_expr.Origin.Boundary v -> v :: acc
      | Ground_expr.Origin.Capture _ | Ground_expr.Origin.Dst _
      | Ground_expr.Origin.Src _ ->
          acc)
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
(* A grounding failure is a verdict ABOUT this cluster, not an error for the
   caller -- the rule [compare_at] already follows for [Ground_eval.at]. The
   conversion drops the [Err.Error.t] wrapper, because a verdict stores only
   the payload; that is a legitimate boundary, but it must not read like the
   backtrace-destroying defect .ai/printer_conventions.md warns about, so it
   gets ONE named helper used at both sites. [Err.map_error] is not it -- the
   destination is a verdict, not another result.

   The two budget tags get their OWN named verdicts rather than falling into
   the generic [Eval] bucket -- the exact mapping the design record's
   "Grounding meter and verdict mapping" specifies -- so a report can count
   and label a budget exhaustion distinctly from an ordinary grounding
   failure. *)
let unproved_of_eval_error (e : Ground_eval.error Err.Error.t) =
  match Err.Error.kind e with
  | `Ground_nodes_over_limit limit ->
      Verdict.Unproved (Unproved.Max_ground_nodes limit)
  | `Pair_nodes_over_limit limit -> Verdict.Unproved (Unproved.Max_nodes limit)
  | other -> Verdict.Unproved (Unproved.Eval other)

let rec settle ~budget ~probe ~tolerance ~label ~proof ~rounds ~meter ~lhs ~rhs
    ~lhs_env ~rhs_env ~lhs_boundary ~rhs_boundary ~coord ~members =
  let lhs_expr = Ground_eval.Term.expression lhs
  and rhs_expr = Ground_eval.Term.expression rhs in
  let lhs_member, rhs_member = members in
  let seen () =
    ( Ground_expr.project ~boundary:lhs_boundary lhs_expr,
      Ground_expr.project ~boundary:rhs_boundary rhs_expr )
  in
  let projected_lhs, projected_rhs = seen () in
  if Ground_expr.equal projected_lhs projected_rhs then Verdict.Proved proof
  else
    let ln =
      Ground_expr.normalise
        ~stored_f32:(Ground_eval.Env.stored_f32 lhs_env)
        lhs_expr
    and rn =
      Ground_expr.normalise
        ~stored_f32:(Ground_eval.Env.stored_f32 rhs_env)
        rhs_expr
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
        Ground_eval.expandable ~boundary:lhs_crossing lhs_env lhs_expr
        || Ground_eval.expandable ~boundary:rhs_crossing rhs_env rhs_expr
      in
      if expandable && rounds >= budget.Budget.max_rounds then
        Verdict.Unproved Unproved.Max_rounds
      else if expandable then
        (* Process left then right EXPLICITLY, not via tuple-argument
           evaluation order: both share [meter]'s running pair total, so
           which one runs first can change whether the other still has
           allowance left. *)
        match Ground_eval.expand ~meter ~boundary:lhs_crossing lhs_env lhs with
        | Error e -> unproved_of_eval_error e
        | Ok lhs' -> (
            match
              Ground_eval.expand ~meter ~boundary:rhs_crossing rhs_env rhs
            with
            | Error e -> unproved_of_eval_error e
            | Ok rhs' ->
                (* Neither side grew, and something is still expandable: no
                   allowance remains under [meter]'s shared [max_nodes], since
                   a genuine grounding failure would already have thrown
                   above. Reporting the LIMIT, not an observed size, matches
                   [Unproved.Max_ground_nodes]'s own payload convention. *)
                if
                  expandable
                  && Ground_expr.equal lhs_expr
                       (Ground_eval.Term.expression lhs')
                  && Ground_expr.equal rhs_expr
                       (Ground_eval.Term.expression rhs')
                then
                  Verdict.Unproved (Unproved.Max_nodes budget.Budget.max_nodes)
                else
                  settle ~budget ~probe ~tolerance ~label ~proof
                    ~rounds:(rounds + 1) ~meter ~lhs:lhs' ~rhs:rhs' ~lhs_env
                    ~rhs_env ~lhs_boundary ~rhs_boundary ~coord ~members)
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
            match
              ( label,
                unbound_constant_at ~lhs_env ~rhs_env ~lhs:lhs_expr
                  ~rhs:rhs_expr )
            with
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
      if
        Core.Float_bits.equal_exact (Ground_expr.eval lhs v)
          (Ground_expr.eval rhs v)
      then draw (n + 1)
      else Some v
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
  (* Fresh per attempt: the Structural attempt and the subsequent
     constant-bound attempt must not share allowance, or a discarded
     attempt's spend would starve the authoritative one. *)
  let attempt proof lhs_env rhs_env =
    let meter =
      Ground_eval.Meter.create
        {
          Ground_eval.Budget.max_ground_nodes = budget.Budget.max_ground_nodes;
          max_nodes = budget.Budget.max_nodes;
        }
    in
    (* Left then right EXPLICITLY, not via tuple-argument evaluation order:
       both roots share [meter]'s pair total, so which one registers first
       can change whether the other still fits. *)
    match Ground_eval.at ~meter lhs_env lhs_at.loc_id coord with
    | Error e -> unproved_of_eval_error e
    | Ok lhs -> (
        match Ground_eval.at ~meter rhs_env rhs_at.loc_id coord with
        | Error e -> unproved_of_eval_error e
        | Ok rhs -> (
            let out_of_bounds =
              match
                Ground_eval.out_of_bounds lhs_env
                  (Ground_eval.Term.expression lhs)
              with
              | Some c -> Some c
              | None ->
                  Ground_eval.out_of_bounds rhs_env
                    (Ground_eval.Term.expression rhs)
            in
            match out_of_bounds with
            | Some c -> Verdict.Unproved (Unproved.Out_of_bounds c)
            | None ->
                settle ~budget ~probe ~tolerance ~label ~proof ~rounds:0 ~meter
                  ~lhs ~rhs ~lhs_env ~rhs_env ~lhs_boundary:lhs_at.loc_boundary
                  ~rhs_boundary:rhs_at.loc_boundary ~coord ~members))
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
  let outcome verdict coverage = Err.return { Outcome.coverage; verdict } in
  if Correspondence.Set.is_empty c.src || Correspondence.Set.is_empty c.dst then
    outcome Verdict.Vacuous Coverage.Not_applicable
  else
    (* [Unverifiable] is NOT short-circuited here. It asserts nothing about
       values, but structural equality is not a statement about values — it
       observes that the two sides compute the same term, which is exactly what
       an unchanged transfer function downstream of a value-destroying rewrite
       has to say for itself. [settle] declines the tiers below structural; see
       its [Unverifiable] arm. *)
    let open Err.Syntax in
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
          Err.List.fold_left
            (fun acc m ->
              match acc with
              | Some _ -> Err.return acc
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
