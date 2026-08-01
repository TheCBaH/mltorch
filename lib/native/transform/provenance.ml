(* See provenance.mli. Keyed by destination id, since every query is "where did
   this come from" and a destination has exactly one derivation. Key and payload
   sit at different versions, which is what [Correspondence.Map] keeps apart. *)

module Set = Correspondence.Set
module Map = Correspondence.Map

type ('src, 'dst) t = { edges : ('dst, 'src Correspondence.set) Map.t }

let empty = { edges = Map.empty }

let add ~sources dst t =
  (* No entry and an empty entry agree: union's identity IS the absent case. *)
  let merged =
    Set.union
      (Option.value ~default:Set.empty (Map.find_opt dst t.edges))
      sources
  in
  if Set.is_empty merged then t else { edges = Map.update dst merged t.edges }

let of_list l =
  List.fold_left (fun acc (sources, dst) -> add ~sources dst acc) empty l

let is_empty t = Map.is_empty t.edges
let edges t = Map.bindings t.edges |> List.map (fun (d, s) -> (s, d))
let sources_of t id = Option.value (Map.find_opt id t.edges) ~default:Set.empty

let targets_of t id =
  Map.fold
    (fun dst sources acc -> if Set.mem id sources then Set.add dst acc else acc)
    t.edges Set.empty

(* An A->C edge exists when an A->B edge feeds something that a B->C edge (or the
   B->C value correspondence) carries forward. Sources are pulled back through
   the first correspondence, targets pushed forward through the second. *)
let compose a b ~values =
  let ab, bc = values in
  let pull_back set =
    Set.fold
      (fun id acc -> Set.union acc (Correspondence.backward ab id))
      set Set.empty
  in
  let push_forward id = Correspondence.forward bc id in
  (* Direct A->B edges, with their destination carried into C. *)
  let from_a =
    Map.fold
      (fun mid sources acc ->
        Set.fold (fun dst acc -> add ~sources dst acc) (push_forward mid) acc)
      a.edges empty
  in
  (* B->C edges, with their sources pulled back into A. *)
  Map.fold
    (fun dst mids acc ->
      let sources =
        Set.fold
          (fun mid acc ->
            (* A middle id that is itself derived contributes its own sources,
               so derivation chains stay transitive. Those are ALREADY source-
               side ids and must not be pulled back again: [backward ab] reads
               its argument as a middle id, and while an unmentioned one comes
               back unchanged, a source id that also occurs as a middle
               destination is silently replaced by that cluster's sources. Only
               [mid] itself is a middle id. *)
            let via = sources_of a mid in
            let direct =
              if Set.is_empty via then pull_back (Set.singleton mid) else via
            in
            Set.union acc direct)
          mids Set.empty
      in
      add ~sources dst acc)
    b.edges from_a

let validate t ~src ~dst =
  let ( let* ) = Result.bind in
  let resolves universe id =
    Correspondence.Universe.find universe (Correspondence.raw id) <> None
  in
  Map.fold
    (fun target sources acc ->
      let* () = acc in
      let* () =
        if resolves dst target then Ok ()
        else Error (Cluster_relation.Dangling_dst (Correspondence.raw target))
      in
      Set.fold
        (fun source acc ->
          let* () = acc in
          if resolves src source then Ok ()
          else Error (Cluster_relation.Dangling_src (Correspondence.raw source)))
        sources (Ok ()))
    t.edges (Ok ())

let pp fmt t =
  if is_empty t then Fmt.string fmt "none"
  else
    let pp_edge fmt (sources, dst) =
      Fmt.pf fmt "@[<h>%a -> %a@]"
        (Fmt.braces (Fmt.list ~sep:Fmt.comma Graph_ir.Tensor_id.pp))
        (Correspondence.raws sources |> Graph_ir.Tensor_id.Set.elements)
        Graph_ir.Tensor_id.pp (Correspondence.raw dst)
    in
    Fmt.(list ~sep:cut pp_edge) fmt (edges t)
