(* Naming a version tag in a test.

   [Snapshot.create] and [Brand.fresh] hand back existentials — that is exactly
   what makes a version unforgeable (.ai/native_transform_versioning.md §2) — and
   an existential cannot be bound by a toplevel [let]. A first-class module can
   carry one out, so a test writes

     module A = (val Version_fixture.of_graph src)

   and then refers to [A.v] wherever the tag is needed, instead of threading a
   rank-2 callback through every case.

   Nothing here weakens the guarantee: [ids] mints its own brand, whose tag
   corresponds to no snapshot, and [of_graph] just re-exposes the tag
   [Snapshot.create] already chose. *)

open Graph_ir

module type SNAP = sig
  type v

  val snapshot : v Snapshot.t
end

let of_graph g : (module SNAP) =
  match Core.or_raise Graph_view.pp_error (Snapshot.create g) with
  | Snapshot.Pack (type a) (s : a Snapshot.t) ->
      (module struct
        type v = a

        let snapshot = s
      end)

(* A bare id universe, for tests about the relation algebra itself: cluster
   normalisation, composition, inversion. Those are statements about ids, not
   about any particular graph, and minting a graph to state them would obscure
   what is under test. Tensor and node ids share one brand, exactly as a real
   snapshot's do. *)
module type IDS = sig
  type v

  val edge : int -> v Correspondence.id

  (* A universe at this tag over any id set. Tests that state a relation without
     stating two graphs use the smallest pair the relation could describe — the
     ids it mentions on each side — which makes the endpoint checks vacuous, as
     they should be where normalisation is what is under test. *)
  val edges : Tensor_id.Set.t -> v Correspondence.Universe.t
  val node : int -> v Node_map.id
  val nodes : Node_id.Set.t -> v Node_map.Universe.t
end

let ids n : (module IDS) =
  let upto of_int add empty =
    List.init n Fun.id |> List.fold_left (fun acc i -> add (of_int i) acc) empty
  in
  match Brand.fresh () with
  | Brand.Pack (type a) (b : a Brand.t) ->
      (module struct
        type v = a

        let edges = Correspondence.Universe.create b
        let nodes = Node_map.Universe.create b

        let all_edges =
          edges (upto Tensor_id.of_int Tensor_id.Set.add Tensor_id.Set.empty)

        let all_nodes =
          nodes (upto Node_id.of_int Node_id.Set.add Node_id.Set.empty)

        (* [Option.get] is the right shape here: an id outside the range is a
           broken test, not a case to handle. *)
        let edge i =
          Option.get
            (Correspondence.Universe.find all_edges (Tensor_id.of_int i))

        let node i =
          Option.get (Node_map.Universe.find all_nodes (Node_id.of_int i))
      end)
