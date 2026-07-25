(* See provenance.mli. Keyed by destination id, since every query is "where did
   this come from" and a destination has exactly one derivation. *)

open Graph_ir

type ('src, 'dst) t = { edges : Tensor_id.Set.t Tensor_id.Map.t }

let empty = { edges = Tensor_id.Map.empty }

let add ~sources dst t =
  let incoming = Tensor_id.Set.of_list sources in
  let merged =
    match Tensor_id.Map.find_opt dst t.edges with
    | None -> incoming
    | Some existing -> Tensor_id.Set.union existing incoming
  in
  if Tensor_id.Set.is_empty merged then t
  else { edges = Tensor_id.Map.add dst merged t.edges }

let of_list l =
  List.fold_left (fun acc (sources, dst) -> add ~sources dst acc) empty l

let is_empty t = Tensor_id.Map.is_empty t.edges
let edges t = Tensor_id.Map.bindings t.edges |> List.map (fun (d, s) -> (s, d))

let sources_of t id =
  Option.value (Tensor_id.Map.find_opt id t.edges) ~default:Tensor_id.Set.empty

let targets_of t id =
  Tensor_id.Map.fold
    (fun dst sources acc ->
      if Tensor_id.Set.mem id sources then Tensor_id.Set.add dst acc else acc)
    t.edges Tensor_id.Set.empty

(* An A->C edge exists when an A->B edge feeds something that a B->C edge (or the
   B->C value correspondence) carries forward. Sources are pulled back through
   the first correspondence, targets pushed forward through the second. *)
let compose a b ~values =
  let ab, bc = values in
  let pull_back set =
    Tensor_id.Set.fold
      (fun id acc -> Tensor_id.Set.union acc (Correspondence.backward ab id))
      set Tensor_id.Set.empty
  in
  let push_forward id = Correspondence.forward bc id in
  (* Direct A->B edges, with their destination carried into C. *)
  let from_a =
    Tensor_id.Map.fold
      (fun mid sources acc ->
        Tensor_id.Set.fold
          (fun dst acc -> add ~sources:(Tensor_id.Set.elements sources) dst acc)
          (push_forward mid) acc)
      a.edges empty
  in
  (* B->C edges, with their sources pulled back into A. *)
  Tensor_id.Map.fold
    (fun dst mids acc ->
      let sources =
        Tensor_id.Set.fold
          (fun mid acc ->
            (* A middle id that is itself derived contributes its own sources,
               so derivation chains stay transitive. *)
            let via = sources_of a mid in
            let direct =
              if Tensor_id.Set.is_empty via then
                pull_back (Tensor_id.Set.singleton mid)
              else pull_back via
            in
            Tensor_id.Set.union acc direct)
          mids Tensor_id.Set.empty
      in
      add ~sources:(Tensor_id.Set.elements sources) dst acc)
    b.edges from_a

let validate t ~src ~dst =
  let ( let* ) = Result.bind in
  Tensor_id.Map.fold
    (fun target sources acc ->
      let* () = acc in
      let* () =
        if Tensor_id.Set.mem target dst then Ok ()
        else Error (Cluster_relation.Dangling_dst target)
      in
      Tensor_id.Set.fold
        (fun source acc ->
          let* () = acc in
          if Tensor_id.Set.mem source src then Ok ()
          else Error (Cluster_relation.Dangling_src source))
        sources (Ok ()))
    t.edges (Ok ())

let pp fmt t =
  if is_empty t then Fmt.string fmt "none"
  else
    let pp_edge fmt (sources, dst) =
      Fmt.pf fmt "@[<h>%a -> %a@]"
        (Fmt.braces (Fmt.list ~sep:Fmt.comma Tensor_id.pp))
        (Tensor_id.Set.elements sources)
        Tensor_id.pp dst
    in
    Fmt.(list ~sep:cut pp_edge) fmt (edges t)
