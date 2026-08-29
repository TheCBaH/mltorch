(* See map_verify.mli.

   This file is now a thin facade over map_verify_types.ml (the Member/
   Budget/.../Policy/error vocabulary) and map_verify_check.ml (the actual
   per-cluster verification algorithm); it keeps only the Make_pair/run/step
   entry points that assemble a Report by calling check_cluster once per
   cluster.. Every aliased module/type below is exactly what
   map_verify.mli already declared, so the .mli did not change. *)

open Graph_ir
module Member = Map_verify_types.Member
module Budget = Map_verify_types.Budget
module Effort = Map_verify_types.Effort
module Coverage = Map_verify_types.Coverage
module Strength = Map_verify_types.Strength
module Refutation = Map_verify_types.Refutation
module Unproved = Map_verify_types.Unproved
module Verdict = Map_verify_types.Verdict
module Outcome = Map_verify_types.Outcome
module Group_path = Map_verify_types.Group_path
module Entry = Map_verify_types.Entry
module Tally = Map_verify_types.Tally
module Report = Map_verify_types.Report
module Policy = Map_verify_types.Policy

type error = Map_verify_types.error

let pp_error = Map_verify_types.pp_error

open Map_verify_check

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

  let run ?(budget = Budget.default) ?max_clusters
      ?(coefficient_tolerance = default_coefficient_tolerance) ?(probe = 4)
      ?(src_constants = Tensor_id.Map.empty)
      ?(dst_constants = Tensor_id.Map.empty) ?src_constant_store
      ?dst_constant_store map ~(src : 'src Src.Snapshot.t)
      ~(dst : 'dst Dst.Snapshot.t) : (Report.t, error) Err.t =
    let open Err.Syntax in
    (* Endpoint validation now happens in [Graph_map.create], but closure does not
       survive [Graph_map.compose] — which takes no snapshots and so cannot
       re-check — and a composed map is exactly what a cumulative run receives. *)
    let* () =
      (Map_pair.check_claim_closure map ~src ~dst :> (unit, error) Err.t)
    in
    let clusters = Map_pair.clusters_over map ~src ~dst in
    let rec take n kept rest =
      if n <= 0 then (List.rev kept, rest)
      else
        match rest with
        | [] -> (List.rev kept, [])
        | x :: xs -> take (n - 1) (x :: kept) xs
    in
    let checked_clusters, skipped_clusters =
      match max_clusters with
      | None -> (clusters, [])
      | Some limit -> take (max 0 limit) [] clusters
    in
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
        env =
          Ground_eval.Env.of_program ?constant_store:src_constant_store program
            ~side:`Src;
        sig_of = Src.sig_of src;
        edges = Src.Snapshot.edges src;
        with_constants =
          Ground_eval.Env.of_program ~constants:src_constants
            ?constant_store:src_constant_store program ~side:`Src;
      }
    and dst_side =
      let program = Dst.symbolic dst in
      {
        env =
          Ground_eval.Env.of_program ?constant_store:dst_constant_store program
            ~side:`Dst;
        sig_of = Dst.sig_of dst;
        edges = Dst.Snapshot.edges dst;
        with_constants =
          Ground_eval.Env.of_program ~constants:dst_constants
            ?constant_store:dst_constant_store program ~side:`Dst;
      }
    in
    let sides = { dst = dst_side; src = src_side } in
    (* Cluster order is free, and that is the point. The destination-topological
       traversal this replaces existed so a cluster could lean on conclusions
       reached strictly earlier; a local obligation leans on nothing, so the
       clusters are checked in report order and each verdict stands alone. *)
    let* checked =
      Err.List.fold_left
        (fun acc c ->
          let+ outcome =
            check_cluster ~budget ~index ~probe ~tolerance:coefficient_tolerance
              sides c
          in
          outcome :: acc)
        [] checked_clusters
    in
    let outcomes = List.rev checked in
    let dst_graph = Dst.Snapshot.graph dst in
    let index = Group_path.index dst_graph
    and producers =
      Group_path.producers ~edge:(Dst.Snapshot.edge dst) dst_graph
    in
    Err.return
      {
        Report.entries =
          (List.map2
             (fun cluster outcome ->
               {
                 Entry.cluster = Correspondence.Cluster.erase cluster;
                 group = Group_path.of_cluster ~index ~producers cluster;
                 outcome;
               })
             checked_clusters outcomes
          @
          match skipped_clusters with
          | [] -> []
          | cluster :: _ ->
              [
                {
                  Entry.cluster = Correspondence.Cluster.erase cluster;
                  group = Group_path.of_cluster ~index ~producers cluster;
                  outcome =
                    {
                      coverage = Coverage.Not_applicable;
                      verdict =
                        Verdict.Unproved
                          (Unproved.Max_clusters (List.length skipped_clusters));
                    };
                };
              ]);
      }
end

(* The Native-to-Native specialization: what every existing caller uses. *)
include Make_pair (Native_side) (Native_side)

let step ?budget ?coefficient_tolerance ?probe before
    (Rewrite.Step (after, map)) =
  run ?budget ?coefficient_tolerance ?probe map ~src:(Rewrite.snapshot before)
    ~src_constants:(Rewrite.constants before) ~dst:(Rewrite.snapshot after)
    ~dst_constants:(Rewrite.constants after)
