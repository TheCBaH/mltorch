(* See cluster_relation.mli. A relation is a canonical list of pairwise-disjoint
   clusters: normalisation merges anything sharing an id, so "which cluster is id
   x in" is unambiguous and equality of relations is list equality. *)

type 'id issue =
  | Dangling_dst of 'id
  | Dangling_src of 'id
  | Uncovered_dst of 'id
  | Uncovered_src of 'id
  | Unpaired_dst of 'id
  | Unpaired_src of 'id

let pp_issue pp_id fmt = function
  | Dangling_dst id ->
      Fmt.pf fmt "@[<h>dst %a is not in the destination@]" pp_id id
  | Dangling_src id -> Fmt.pf fmt "@[<h>src %a is not in the source@]" pp_id id
  | Uncovered_dst id ->
      Fmt.pf fmt "@[<h>%a is implicitly identity but absent from the source@]"
        pp_id id
  | Uncovered_src id ->
      Fmt.pf fmt
        "@[<h>%a is implicitly identity but absent from the destination@]" pp_id
        id
  | Unpaired_dst id ->
      Fmt.pf fmt "@[<h>%a is mapped to but unmentioned as a source@]" pp_id id
  | Unpaired_src id ->
      Fmt.pf fmt "@[<h>%a is mapped from but unmentioned as a destination@]"
        pp_id id

module type ID = sig
  type t

  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit

  module Set : Set.S with type elt = t
end

module type LABEL = sig
  type t

  val identity : t
  val join : t -> t -> t
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

module Make (Id : ID) (Label : LABEL) = struct
  (* The version index is erased outright: a ['v id] IS an [Id.t] and a ['v set]
     IS an [Id.Set.t], so everything below is the same code that ran before the
     indexing existed. Retagging is therefore free in here, which is exactly why
     this layer lives inside the functor — see the .mli. *)
  type 'v id = Id.t
  type 'v set = Id.Set.t

  let raw id = id
  let raws set = set

  module Universe = struct
    (* The brand is a type-level witness only (see brand.mli), so it is taken and
       discarded rather than stored. *)
    type 'v t = Id.Set.t

    let create (_ : 'v Brand.t) ids = ids
    let find u id = if Id.Set.mem id u then Some id else None
    let elements u = Id.Set.elements u
    let ids u = u
  end

  module Map = struct
    (* [Stdlib.Map]: nothing shadows it here, but [Set] below shadows [Set]. *)
    module M = Stdlib.Map.Make (Id)

    type ('v, 'a) t = 'a M.t

    let bindings = M.bindings
    let empty = M.empty
    let find_opt = M.find_opt
    let fold = M.fold
    let is_empty = M.is_empty
    let update = M.add
  end

  module Set = struct
    let add = Id.Set.add
    let cardinal = Id.Set.cardinal
    let disjoint = Id.Set.disjoint
    let elements = Id.Set.elements
    let empty = Id.Set.empty
    let equal = Id.Set.equal
    let fold = Id.Set.fold
    let is_empty = Id.Set.is_empty
    let mem = Id.Set.mem
    let min_elt_opt = Id.Set.min_elt_opt
    let of_list = Id.Set.of_list
    let singleton = Id.Set.singleton
    let union = Id.Set.union
  end

  module Cluster = struct
    type ('src, 'dst) t = { src : 'src set; dst : 'dst set; label : Label.t }

    let pp_side fmt s =
      Fmt.braces (Fmt.list ~sep:Fmt.comma Id.pp) fmt (Id.Set.elements s)

    (* The label is omitted when it renders empty, so an unlabelled relation
       (nodes) does not print a trailing separator. *)
    let render fmt (src, dst, label) =
      match Core.Pretty.to_string Label.pp label with
      | "" -> Fmt.pf fmt "@[<h>%a -> %a@]" pp_side src pp_side dst
      | _ ->
          Fmt.pf fmt "@[<h>%a -> %a %a@]" pp_side src pp_side dst Label.pp label

    let pp fmt { src; dst; label } = render fmt (src, dst, label)

    module Erased = struct
      type t = { src : Id.Set.t; dst : Id.Set.t; label : Label.t }

      let pp fmt { src; dst; label } = render fmt (src, dst, label)
    end

    let erase (c : ('src, 'dst) t) =
      { Erased.src = c.src; dst = c.dst; label = c.label }
  end

  open Cluster

  type ('src, 'dst) t = { clusters : ('src, 'dst) Cluster.t list }

  let union a b =
    {
      src = Id.Set.union a.src b.src;
      dst = Id.Set.union a.dst b.dst;
      label = Label.join a.label b.label;
    }

  (* [acc] is pairwise disjoint by induction, so every cluster overlapping [c]
     is found in one pass: anything in [rest] could only reach [c] through an id
     [c] already has, and all of those are in [hit]. *)
  let insert c acc =
    let overlaps o =
      (not (Id.Set.disjoint o.src c.src)) || not (Id.Set.disjoint o.dst c.dst)
    in
    let hit, rest = List.partition overlaps acc in
    List.fold_left union c hit :: rest

  let is_implicit c =
    Label.equal c.label Label.identity
    && Id.Set.equal c.src c.dst
    && Id.Set.cardinal c.src = 1

  (* Deterministic order: by lowest src id, then lowest dst id. A side may be
     empty (creation / deletion), which sorts after any populated side. *)
  let compare_side a b =
    match (Id.Set.min_elt_opt a, Id.Set.min_elt_opt b) with
    | None, None -> 0
    | None, Some _ -> 1
    | Some _, None -> -1
    | Some x, Some y -> Id.compare x y

  let normalise clusters =
    List.fold_left (fun acc c -> insert c acc) [] clusters
    |> List.filter (fun c ->
        (not (is_implicit c))
        && not (Id.Set.is_empty c.src && Id.Set.is_empty c.dst))
    |> List.sort (fun a b ->
        match compare_side a.src b.src with
        | 0 -> compare_side a.dst b.dst
        | c -> c)

  let identity = { clusters = [] }
  let clusters t = t.clusters
  let is_empty t = t.clusters = []

  let find_side get t id =
    List.find_opt (fun c -> Id.Set.mem id (get c)) t.clusters

  let forward t id =
    match find_side (fun c -> c.src) t id with
    | Some c -> c.dst
    | None -> Id.Set.singleton id

  let backward t id =
    match find_side (fun c -> c.dst) t id with
    | Some c -> c.src
    | None -> Id.Set.singleton id

  let side_union get t =
    List.fold_left
      (fun acc c -> Id.Set.union acc (get c))
      Id.Set.empty t.clusters

  let mentioned_src t = side_union (fun c -> c.src) t
  let mentioned_dst t = side_union (fun c -> c.dst) t

  let created t =
    List.fold_left
      (fun acc c ->
        if Id.Set.is_empty c.src then Id.Set.union acc c.dst else acc)
      Id.Set.empty t.clusters

  let deleted t =
    List.fold_left
      (fun acc c ->
        if Id.Set.is_empty c.dst then Id.Set.union acc c.src else acc)
      Id.Set.empty t.clusters

  let invert t =
    {
      clusters =
        normalise
          (List.map (fun c -> { c with src = c.dst; dst = c.src }) t.clusters);
    }

  (* ---- composition ------------------------------------------------------- *)

  (* Composition needs three id roles at once, so it works over tagged ids and
     reuses the same disjoint-merge as [normalise]. *)
  module Tag = struct
    type t = Dst of Id.t | Mid of Id.t | Src of Id.t

    (* Comparison follows the composition direction, source -> middle ->
       destination, not constructor spelling. It fixes [Tset]'s traversal and
       therefore the deterministic normal form of a composed relation. *)
    let rank = function Src _ -> 0 | Mid _ -> 1 | Dst _ -> 2

    let compare a b =
      match (a, b) with
      | Src x, Src y | Mid x, Mid y | Dst x, Dst y -> Id.compare x y
      | _ -> Int.compare (rank a) (rank b)
  end

  (* [Stdlib.Set]: the tagged [Set] above shadows it. *)
  module Tset = Stdlib.Set.Make (Tag)

  let tagged f s = Id.Set.fold (fun id acc -> Tset.add (f id) acc) s Tset.empty

  let tinsert (members, label) acc =
    let overlaps (m, _) = not (Tset.disjoint m members) in
    let hit, rest = List.partition overlaps acc in
    List.fold_left
      (fun (m, l) (m', l') -> (Tset.union m m', Label.join l l'))
      (members, label) hit
    :: rest

  let compose a b =
    let a_dst = mentioned_dst a and b_src = mentioned_src b in
    let a_groups =
      List.map
        (fun c ->
          ( Tset.union
              (tagged (fun i -> Tag.Src i) c.src)
              (tagged (fun i -> Tag.Mid i) c.dst),
            c.label ))
        a.clusters
    in
    let b_groups =
      List.map
        (fun c ->
          ( Tset.union
              (tagged (fun i -> Tag.Mid i) c.src)
              (tagged (fun i -> Tag.Dst i) c.dst),
            c.label ))
        b.clusters
    in
    (* Identity-extension across the middle, guarded: an id the partner declares
       created or deleted did not exist in the middle graph, so extending it
       would invent a correspondence (and could fuse a dead id with a later
       cluster reusing its value). *)
    let b_created = created b and a_deleted = deleted a in
    let extend_right =
      Id.Set.fold
        (fun m acc ->
          if Id.Set.mem m b_src || Id.Set.mem m b_created then acc
          else (Tset.of_list [ Tag.Mid m; Tag.Dst m ], Label.identity) :: acc)
        a_dst []
    in
    let extend_left =
      Id.Set.fold
        (fun m acc ->
          if Id.Set.mem m a_dst || Id.Set.mem m a_deleted then acc
          else (Tset.of_list [ Tag.Src m; Tag.Mid m ], Label.identity) :: acc)
        b_src []
    in
    let merged =
      List.fold_left
        (fun acc g -> tinsert g acc)
        []
        (a_groups @ b_groups @ extend_right @ extend_left)
    in
    let of_group (members, label) =
      Tset.fold
        (fun tag c ->
          match tag with
          | Tag.Src id -> { c with src = Id.Set.add id c.src }
          | Tag.Dst id -> { c with dst = Id.Set.add id c.dst }
          | Tag.Mid _ -> c)
        members
        { src = Id.Set.empty; dst = Id.Set.empty; label }
    in
    { clusters = normalise (List.map of_group merged) }

  (* ---- construction, which is validation ---------------------------------- *)

  (* Internal: [of_clusters] is the only way in, and it always runs this. A
     ['v id] witnesses membership of SOME universe at ['v], which is not the same
     as membership of the one passed to [of_clusters] — two universes can be
     minted from one unpacked brand — so the endpoint checks stay live. *)
  let validate t ~src ~dst =
    let ( let* ) = Result.bind in
    let check_set s ~universe ~err =
      Id.Set.fold
        (fun id acc ->
          let* () = acc in
          if Id.Set.mem id universe then Ok () else Error (err id))
        s (Ok ())
    in
    let m_src = mentioned_src t and m_dst = mentioned_dst t in
    let* () = check_set m_src ~universe:src ~err:(fun id -> Dangling_src id) in
    let* () = check_set m_dst ~universe:dst ~err:(fun id -> Dangling_dst id) in
    (* Implicit identity: an unmentioned id must exist on both sides, and an id
       mentioned on one side while present in both graphs must be mentioned on
       the other — otherwise it would be both explicitly related and implicitly
       identical, which no equivalence can be. *)
    let* () =
      Id.Set.fold
        (fun id acc ->
          let* () = acc in
          if Id.Set.mem id m_src then Ok ()
          else if not (Id.Set.mem id dst) then Error (Uncovered_src id)
          else if Id.Set.mem id m_dst then Error (Unpaired_dst id)
          else Ok ())
        src (Ok ())
    in
    Id.Set.fold
      (fun id acc ->
        let* () = acc in
        if Id.Set.mem id m_dst then Ok ()
        else if not (Id.Set.mem id src) then Error (Uncovered_dst id)
        else if Id.Set.mem id m_src then Error (Unpaired_src id)
        else Ok ())
      dst (Ok ())

  let of_clusters ~src ~dst cs =
    let rel = { clusters = normalise cs } in
    validate rel ~src:(Universe.ids src) ~dst:(Universe.ids dst)
    |> Result.map (fun () -> rel)

  let pp fmt t =
    match t.clusters with
    | [] -> Fmt.string fmt "identity"
    | clusters -> Fmt.(list ~sep:cut Cluster.pp) fmt clusters
end
