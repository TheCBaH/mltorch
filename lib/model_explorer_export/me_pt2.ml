(* PT2 → initial Native navigation. See the .mli. *)

type t = {
  entries : Me_session.Mapping_entry.t list;
  created : string list;
  deleted : string list;
}

(* Union-find over the two id spaces at once. They are disjoint by
   construction: a PT2 element is keyed by its rendered id, a native one by
   [Me_ids.op_node], and the two grammars share no string. Keeping them in ONE
   structure is what makes a component span both sides, which is the whole
   point -- a decomposition and a fold meet in the same component. *)
module Uf = struct
  type t = (string, string) Hashtbl.t

  let create () : t = Hashtbl.create 64

  let rec find (uf : t) x =
    match Hashtbl.find_opt uf x with
    | None ->
        Hashtbl.replace uf x x;
        x
    | Some p when String.equal p x -> x
    | Some p ->
        let r = find uf p in
        (* Path compression: the relation is small but a chain of folds can
           make it deep, and the walk below visits every element. *)
        Hashtbl.replace uf x r;
        r

  let union uf a b =
    let ra = find uf a and rb = find uf b in
    if not (String.equal ra rb) then
      (* Deterministic representative: the lexicographically smaller root, so
         the component's identity does not depend on insertion order and two
         runs over the same sidecar agree. *)
      if String.compare ra rb <= 0 then Hashtbl.replace uf rb ra
      else Hashtbl.replace uf ra rb
end

(* A PT2 node identity, rendered the way the source view renders it. Only root
   origins participate: the lowerer is root-only, so a non-root [graph_path]
   names a node in a nested graph that has no native counterpart to navigate
   to, and pairing one would send the right pane somewhere unrelated. *)
let pt2_id (o : Pt2_native_graph.Node_origin.t) =
  Me_ids.pt2_node o.Pt2_native_graph.Node_origin.graph_path
    o.Pt2_native_graph.Node_origin.index

let is_root (o : Pt2_native_graph.Node_origin.t) =
  o.Pt2_native_graph.Node_origin.graph_path = Pt2_native_graph.Graph_path.root

let of_origins ~limits ~source_nodes origins =
  let open Core.Syntax in
  let uf = Uf.create () in
  let native_side = Hashtbl.create 64 in
  let pt2_side = Hashtbl.create 64 in
  let created = ref [] in
  (* One pass over the sidecar: register both sides and join them. A native
     node whose root origins are empty is [created] here rather than being
     inferred later from a set difference, which is what keeps the two derived
     from one relation and stops a node being both. *)
  Graph_ir.Node_id.Map.iter
    (fun node os ->
      let nid = Me_ids.op_node node in
      let roots = List.filter is_root os in
      match roots with
      | [] -> created := nid :: !created
      | _ ->
          Hashtbl.replace native_side nid ();
          List.iter
            (fun o ->
              let pid = pt2_id o in
              Hashtbl.replace pt2_side pid ();
              Uf.union uf nid pid)
            roots)
    origins;
  let total = Hashtbl.length native_side + Hashtbl.length pt2_side in
  let* () =
    if total > limits.Me_limits.Limits.max_mapping_members_total then
      Core.fail (`Over_limit ("mappingMembers", total))
    else Core.return ()
  in
  (* Group by representative, then order everything: components by their
     smallest native id, ids within a side lexicographically. Determinism is
     not decoration here -- the session's "two loads produce identical JSON"
     claim quantifies over this output. *)
  let components = Hashtbl.create 64 in
  let add side id =
    let r = Uf.find uf id in
    let l, r' =
      Option.value (Hashtbl.find_opt components r) ~default:([], [])
    in
    Hashtbl.replace components r
      (match side with `Left -> (id :: l, r') | `Right -> (l, id :: r'))
  in
  Hashtbl.iter (fun id () -> add `Right id) native_side;
  Hashtbl.iter (fun id () -> add `Left id) pt2_side;
  let entries =
    Hashtbl.fold
      (fun _ (l, r) acc -> (List.sort compare l, List.sort compare r) :: acc)
      components []
    |> List.sort (fun (_, a) (_, b) -> compare a b)
    |> List.map (fun (left, right) -> { Me_session.Mapping_entry.left; right })
  in
  let* () =
    Core.List.iter
      (fun (e : Me_session.Mapping_entry.t) ->
        let n =
          List.length e.Me_session.Mapping_entry.left
          + List.length e.Me_session.Mapping_entry.right
        in
        if n > limits.Me_limits.Limits.max_mapping_members_per_entry then
          Core.fail (`Over_limit ("mappingMembersPerEntry", n))
        else Core.return ())
      entries
  in
  (* [deleted] needs the source universe, which the sidecar cannot supply: it
     is indexed by native node, so a PT2 node with no counterpart never appears
     in it. [source_nodes] is therefore a parameter, and this is the set
     difference against the ids that DID appear. *)
  let deleted =
    List.filter (fun id -> not (Hashtbl.mem pt2_side id)) source_nodes
    |> List.sort_uniq compare
  in
  Core.return { entries; created = List.sort compare !created; deleted }

let sync t =
  {
    Me_session.Sync_navigation.entries = t.entries;
    show_diff_highlights = t.created <> [] || t.deleted <> [];
  }
