(* See graph_map.mli. *)

open Graph_ir

type ('src, 'dst) t = {
  values : ('src, 'dst) Correspondence.t;
  nodes : ('src, 'dst) Node_map.t;
  provenance : ('src, 'dst) Provenance.t;
}

type error =
  [ `Cluster_format of Tensor_id.t * Tensor_id.t
  | `Graph_output_arity of int * int
  | `Graph_output_mismatch of Tensor_id.t * Tensor_id.t
  | `Cluster_shape of Tensor_id.t * Tensor_id.t
  | `Node_endpoint of Node_id.t Cluster_relation.issue
  | `Provenance_endpoint of Tensor_id.t Cluster_relation.issue
  | `Unclosed_claim of Tensor_id.t
  | `Value_endpoint of Tensor_id.t Cluster_relation.issue ]

let pp_error ppf : [< error ] -> unit = function
  | `Graph_output_arity (s, d) ->
      Fmt.pf ppf "graph outputs differ in arity: %d source, %d destination" s d
  | `Graph_output_mismatch (s, d) ->
      Fmt.pf ppf
        "source output %a and destination output %a are not in one cluster"
        Tensor_id.pp s Tensor_id.pp d
  | `Cluster_format (a, b) ->
      Fmt.pf ppf
        "@[<h>value map: %a and %a are claimed identical across different \
         formats@]"
        Tensor_id.pp a Tensor_id.pp b
  | `Cluster_shape (a, b) ->
      Fmt.pf ppf "@[<h>value map: %a and %a correspond but differ in shape@]"
        Tensor_id.pp a Tensor_id.pp b
  | `Node_endpoint issue ->
      Fmt.pf ppf "@[<h>node map: %a@]"
        (Cluster_relation.pp_issue Node_id.pp)
        issue
  | `Provenance_endpoint issue ->
      Fmt.pf ppf "@[<h>provenance: %a@]"
        (Cluster_relation.pp_issue Tensor_id.pp)
        issue
  | `Unclosed_claim id ->
      Fmt.pf ppf
        "@[<h>value map: %a is implicitly identical but downstream of a weaker \
         claim@]"
        Tensor_id.pp id
  | `Value_endpoint issue ->
      Fmt.pf ppf "@[<h>value map: %a@]"
        (Cluster_relation.pp_issue Tensor_id.pp)
        issue

let identity =
  {
    values = Correspondence.identity;
    nodes = Node_map.identity;
    provenance = Provenance.empty;
  }

let compose a b =
  {
    values = Correspondence.compose a.values b.values;
    nodes = Node_map.compose a.nodes b.nodes;
    provenance =
      Provenance.compose a.provenance b.provenance ~values:(a.values, b.values);
  }

let nodes t = t.nodes
let provenance t = t.provenance
let values t = t.values
let clusters t = Correspondence.clusters t.values

let mentioned t =
  List.fold_left
    (fun acc (c : ('a, 'b) Correspondence.Cluster.t) ->
      Tensor_id.Set.union acc
        (Tensor_id.Set.union
           (Correspondence.raws c.src)
           (Correspondence.raws c.dst)))
    Tensor_id.Set.empty (clusters t)

(* ---- the snapshot-consuming half, over a PAIR of dialects ------------------

   [t] itself stays outside: its data is [Correspondence]/[Node_map]/
   [Provenance], all keyed on ids and dialect-free, so [compose] keeps working
   ACROSS the boundary — a Native-to-Native4D map composed with a
   Native4D-to-Native4D one — with no existential packaging. Only the three
   functions that read snapshots need to know which dialects they are between.

   Applicative, so [Make_pair (A) (B).create] names the same function wherever
   it is written. *)
module Make_pair (Src : Side.S) (Dst : Side.S) = struct
  (* The implicit identities the relation deliberately does not store: every id
     present in both graphs and mentioned by no cluster. Each side is looked up in
     its own universe, so no retag is needed here. *)
  let clusters_over t ~src ~dst =
    let mentioned = mentioned t in
    let implicit =
      Correspondence.Universe.elements (Src.Snapshot.edges src)
      |> List.filter_map (fun s ->
          let id = Correspondence.raw s in
          if Tensor_id.Set.mem id mentioned then None
          else
            Option.map
              (fun d ->
                {
                  Correspondence.Cluster.src = Correspondence.Set.singleton s;
                  dst = Correspondence.Set.singleton d;
                  label = Correspondence.Identical;
                })
              (Dst.Snapshot.edge dst id))
    in
    clusters t @ implicit

  (* ---- checks -------------------------------------------------------------- *)

  (* The relations answer with a bare [Stdlib.result]; crossing into the Core row
     is where the detection backtrace gets captured. *)
  let lift wrap = function
    | Ok x -> Core.return x
    | Error issue -> Core.fail (wrap issue)

  let check_claim_closure t ~src ~dst =
    let explicit =
      List.fold_left
        (fun acc (c : ('a, 'b) Correspondence.Cluster.t) ->
          Correspondence.Set.fold
            (fun d acc -> Tensor_id.Map.add (Correspondence.raw d) c.label acc)
            c.dst acc)
        Tensor_id.Map.empty (clusters t)
    in
    let closed =
      Dst.Transfer.propagate ~explicit
        ~preserved:(fun id -> Src.Snapshot.edge src id <> None)
        (Dst.Snapshot.graph dst)
    in
    Tensor_id.Map.fold
      (fun id _ acc ->
        let open Core.Syntax in
        let* () = acc in
        (* Everything [propagate] returns is weaker than [Identical]; an id that
           seeded the propagation is one the map already speaks about. *)
        if Tensor_id.Map.mem id explicit then Core.return ()
        else Core.fail (`Unclosed_claim id))
      closed (Core.return ())

  (* Step 9 of .ai/native_transform_design.md §7, over every non-vacuous cluster
     INCLUDING the implicit identities — an id kept across a rewrite that changed
     its shape or format is exactly the case the lens must not treat as capturable
     source bytes. *)
  let check_metadata t ~src ~dst =
    (* Each side's own signature lookup: the two dialects have different view
       types, so there is no one [Graph_view.sig_of] to call. *)
    let src_sig = Src.sig_of src and dst_sig = Dst.sig_of dst in
    let fmt_key (Payload.Fmt f) = Payload.fmt_name f in
    let members (c : ('a, 'b) Correspondence.Cluster.t) =
      List.filter_map
        (fun id -> Option.map (fun sg -> (id, sg)) (src_sig id))
        (Correspondence.raws c.src |> Tensor_id.Set.elements)
      @ List.filter_map
          (fun id -> Option.map (fun sg -> (id, sg)) (dst_sig id))
          (Correspondence.raws c.dst |> Tensor_id.Set.elements)
    in
    let agree ~key ~err = function
      | [] -> Core.return ()
      | (id0, sg0) :: rest ->
          Core.List.iter
            (fun (id, sg) ->
              if key sg = key sg0 then Core.return ()
              else Core.fail (err id0 id))
            rest
    in
    Core.List.iter
      (fun (c : ('a, 'b) Correspondence.Cluster.t) ->
        if
          Correspondence.Set.is_empty c.src || Correspondence.Set.is_empty c.dst
        then Core.return ()
        else
          let ms = members c in
          let open Core.Syntax in
          let* () =
            agree
              ~key:(fun (sg : Tensor_sig.t) -> sg.shape)
              ~err:(fun a b -> `Cluster_shape (a, b))
              ms
          in
          match c.label with
          | Correspondence.Identical ->
              agree
                ~key:(fun (sg : Tensor_sig.t) -> (fmt_key sg.fmt, sg.quant))
                ~err:(fun a b -> `Cluster_format (a, b))
                ms
          | _ -> Core.return ())
      (clusters_over t ~src ~dst)

  (* Output correspondence: POSITIONAL and TWO-SIDED.

     .ai/native_transform_verify.md states the owed check as "every graph output
     on each side belongs to a non-vacuous cluster", and that is necessary but
     not sufficient — it admits crossed outputs ([a;b] against [b';a'], every
     output covered, the graphs returning results in opposite order) and an
     output clustered with a destination INTERMEDIATE. [Graph.outputs] is
     ordered, being the graph's signature, so the rule is equal arity plus
     index-wise membership of one cluster.

     Read off [clusters_over], never [clusters]: the latter is explicit clusters
     only and [normalise] drops identity pairs, so under an id-preserving
     conversion a directly translated output appears in no explicit cluster at
     all and reading [clusters] would reject exactly the cleanest maps.

     In [create] ONLY, not in [Map_verify.run]. Unlike claim closure this
     property IS preserved by composition — if a pairs src[i] with mid[i] and b
     pairs mid[i] with dst[i], the composed clusters pair src[i] with dst[i],
     and arity chains — and [identity] satisfies it trivially. Since [create] is
     the only other constructor, every map reaching the verifier already has it,
     so a check there could never fire. A predicate no caller can falsify is
     exactly what map_verify.mli calls dead API stating the wrong rule. *)
  let check_output_correspondence t ~src ~dst =
    let clusters = clusters_over t ~src ~dst in
    let paired s d =
      List.exists
        (fun (c : (_, _) Correspondence.Cluster.t) ->
          Tensor_id.Set.mem s (Correspondence.raws c.Correspondence.Cluster.src)
          && Tensor_id.Set.mem d
               (Correspondence.raws c.Correspondence.Cluster.dst))
        clusters
    in
    let src_outputs = (Src.Snapshot.graph src).Graph_common.Graph.outputs
    and dst_outputs = (Dst.Snapshot.graph dst).Graph_common.Graph.outputs in
    if List.compare_lengths src_outputs dst_outputs <> 0 then
      Core.fail
        (`Graph_output_arity (List.length src_outputs, List.length dst_outputs))
    else
      Core.List.iter
        (fun (s, d) ->
          if paired s d then Core.return ()
          else Core.fail (`Graph_output_mismatch (s, d)))
        (List.combine src_outputs dst_outputs)

  let create ~src ~dst ~values ~nodes ~provenance =
    let open Core.Syntax in
    let* values =
      Correspondence.of_clusters ~src:(Src.Snapshot.edges src)
        ~dst:(Dst.Snapshot.edges dst) values
      |> lift (fun i -> `Value_endpoint i)
    in
    let* nodes =
      Node_map.of_clusters ~src:(Src.Snapshot.nodes src)
        ~dst:(Dst.Snapshot.nodes dst) nodes
      |> lift (fun i -> `Node_endpoint i)
    in
    let* () =
      Provenance.validate provenance ~src:(Src.Snapshot.edges src)
        ~dst:(Dst.Snapshot.edges dst)
      |> lift (fun i -> `Provenance_endpoint i)
    in
    let t = { values; nodes; provenance } in
    let* () = check_metadata t ~src ~dst in
    let* () = check_claim_closure t ~src ~dst in
    let+ () = check_output_correspondence t ~src ~dst in
    t
end

(* The Native-to-Native specialization: what every existing caller uses. *)
include Make_pair (Native_side) (Native_side)

let pp fmt t =
  Fmt.pf fmt
    "@[<v>@[<v 2>values:@,%a@]@,@[<v 2>nodes:@,%a@]@,@[<v 2>provenance:@,%a@]@]"
    Correspondence.pp t.values Node_map.pp t.nodes Provenance.pp t.provenance
