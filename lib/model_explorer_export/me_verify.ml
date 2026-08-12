(* Verification results, placed and counted. See the .mli. *)

module MS = Me_session

type error = [ Me_ids.error | Me_limits.over_limit_error | Pass.count_error ]

let pp_error fmt : [< error ] -> unit = function
  | #Me_ids.error as e -> Me_ids.pp_error fmt e
  | `Over_limit o -> Me_limits.Over_limit.pp fmt o
  | `Count_overflow o -> Pass.Count_overflow.pp fmt o

let over_limit = Me_limits.check ~scope:Me_limits.Scope.Verification
let summary = Pass.Outcome_counts.of_report

let audit_status (log : Pass.Audit_log.t) =
  {
    MS.Capability.Pass_audit_status.retained_reports =
      Int64.of_int (Pass.Audit_log.retained log);
    omitted_reports = Pass.Audit_log.omitted log;
    (* [empty] and "nothing overflowed" are the same statement, so the absent
       case needs no separate representation. *)
    omitted_counts =
      (match log.Pass.Audit_log.overflow with
      | None -> Pass.Outcome_counts.empty
      | Some s -> s.Pass.Audit_summary.counts);
  }

(* --- placement --------------------------------------------------------- *)

(* The root is [""] and has no components, which is not the same as one empty
   component: [String.split_on_char] would give [[""]] and then every path would
   share a spurious first level. *)
let split ns = if ns = "" then [] else String.split_on_char '/' ns

let common a b =
  let rec go acc = function
    | x :: xs, y :: ys when String.equal x y -> go (x :: acc) (xs, ys)
    | _ -> List.rev acc
  in
  go [] (a, b)

let placement ~namespace_of ~producer (e : Map_verify.Entry.t) =
  (* The DESTINATION side only: the source side's ids belong to a graph version
     that is not the one being rendered, and its namespaces would name groups
     this projection never emitted. *)
  let paths =
    Graph_ir.Tensor_id.Set.fold
      (fun id acc ->
        match producer id with
        (* Producerless: a graph input or a constant belongs to no group, and
           counting it as "root" would drag every cluster that touches a weight
           up to the root. *)
        | None -> acc
        | Some n -> split (namespace_of n) :: acc)
      e.Map_verify.Entry.cluster.dst []
  in
  match paths with
  | [] -> ""
  | first :: rest -> String.concat "/" (List.fold_left common first rest)

(* Which node produces a tensor. Graph inputs are deliberately absent, which is
   what makes [placement]'s producerless case observable. *)
let producers (g : Graph_ir.graph) =
  let table = Hashtbl.create 256 in
  List.iter
    (fun (n : Graph_ir.node) ->
      List.iter
        (fun t ->
          Hashtbl.replace table (Graph_ir.Tensor_id.to_int t) n.Graph_ir.Node.id)
        n.Graph_ir.Node.outputs)
    (Graph_ir.nodes g);
  fun t -> Hashtbl.find_opt table (Graph_ir.Tensor_id.to_int t)

let by_namespace ~limits (g : Graph_ir.graph) (report : Map_verify.Report.t) =
  let open Err.Syntax in
  let* namespace_of = Me_build.namespace_of ~limits g.Graph_ir.Graph.root in
  let producer = producers g in
  let table = Hashtbl.create 32 in
  let* () =
    Err.List.iter
      (fun (e : Map_verify.Entry.t) ->
        let ns = placement ~namespace_of ~producer e in
        let current =
          Option.value
            (Hashtbl.find_opt table ns)
            ~default:Pass.Outcome_counts.empty
        in
        let+ next =
          Pass.Outcome_counts.add current e.Map_verify.Entry.outcome
        in
        Hashtbl.replace table ns next)
      report.Map_verify.Report.entries
  in
  let count = Hashtbl.length table in
  let+ () =
    over_limit Me_limits.Field.Group_node_attributes count
      ~ceiling:limits.Me_limits.Limits.max_groups_per_graph
  in
  (* Sorted, and the counts within an entry already canonically ordered by
     [bindings]: two runs over the same report have to agree byte for byte. *)
  Hashtbl.fold
    (fun ns counts acc ->
      ( ns,
        List.map
          (fun (label, n) -> (label, Int64.to_string n))
          (Pass.Outcome_counts.bindings counts) )
      :: acc)
    table []
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)

(* --- per-node data ------------------------------------------------------ *)

let node_data ~limits ~graph (g : Graph_ir.graph) (report : Map_verify.Report.t)
    =
  let open Err.Syntax in
  (* Destination edge -> the claim the whole pipeline makes about it. *)
  let by_edge = Hashtbl.create 256 in
  List.iter
    (fun (e : Map_verify.Entry.t) ->
      Graph_ir.Tensor_id.Set.fold
        (fun id () ->
          Hashtbl.replace by_edge
            (Graph_ir.Tensor_id.to_int id)
            e.Map_verify.Entry.outcome)
        e.Map_verify.Entry.cluster.dst ())
    report.Map_verify.Report.entries;
  let results =
    List.filter_map
      (fun (n : Graph_ir.node) ->
        match
          List.filter_map
            (fun t -> Hashtbl.find_opt by_edge (Graph_ir.Tensor_id.to_int t))
            n.Graph_ir.Node.outputs
        with
        | [] -> None
        | first :: rest ->
            (* [Outcome.join], so the surviving verdict keeps ITS OWN coverage.
               Joining verdict and coverage separately lets an [Unproved
               Too_large] -- which examined nothing, hence [Not_applicable] --
               come out marked [sampled n] borrowed from a sibling output. *)
            let o = List.fold_left Map_verify.Outcome.join first rest in
            Some
              ( Me_ids.op_node n.Graph_ir.Node.id,
                {
                  MS.Node_data_set.value =
                    float_of_int
                      (Map_verify.Verdict.rank o.Map_verify.Outcome.verdict);
                  label = Some (Map_verify.Outcome.label o);
                } ))
      (Graph_ir.nodes g)
  in
  let+ () =
    over_limit Me_limits.Field.Node_data_results (List.length results)
      ~ceiling:limits.Me_limits.Limits.max_node_data_results_per_graph
  in
  { MS.Node_data_set.name = "verification"; graph; results }
