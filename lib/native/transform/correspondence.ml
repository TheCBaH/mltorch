(* See correspondence.mli. *)

module Precision = struct
  type t = { fmt : Payload.packed_fmt; quant : Quant.t option }

  let fmt_key (Payload.Fmt f) = Payload.fmt_name f

  (* [quant] is a first-order variant of floats, ints and arrays, so structural
     compare is exact here; the format is compared by name rather than by
     structure because [packed_fmt] is an existential over a GADT. *)
  let compare a b =
    match String.compare (fmt_key a.fmt) (fmt_key b.fmt) with
    | 0 -> Stdlib.compare a.quant b.quant
    | c -> c

  let pp_packed fmt (Payload.Fmt f) = Payload.pp_fmt fmt f

  let pp fmt { fmt = f; quant } =
    match quant with
    | None -> pp_packed fmt f
    | Some q -> Fmt.pf fmt "@[<h>%a[%a]@]" pp_packed f Quant.pp q

  module Set = Set.Make (struct
    type nonrec t = t

    let compare = compare
  end)
end

type relation =
  | Approximate of Precision.Set.t
  | Equivalent
  | Identical
  | Unverifiable

(* The weaker claim wins; two approximations accumulate their lossy
   representations rather than picking one. *)
let join a b =
  match (a, b) with
  | Unverifiable, _ | _, Unverifiable -> Unverifiable
  | Approximate x, Approximate y -> Approximate (Precision.Set.union x y)
  | Approximate x, _ | _, Approximate x -> Approximate x
  | Equivalent, _ | _, Equivalent -> Equivalent
  | Identical, Identical -> Identical

let pp_relation fmt = function
  | Approximate s ->
      Fmt.pf fmt "@[<h>approximate(%a)@]"
        Fmt.(list ~sep:comma Precision.pp)
        (Precision.Set.elements s)
  | Equivalent -> Fmt.string fmt "equivalent"
  | Identical -> Fmt.string fmt "identical"
  | Unverifiable -> Fmt.string fmt "unverifiable"

module Id = struct
  type t = Tensor_id.t

  let compare = Tensor_id.compare
  let pp = Tensor_id.pp

  module Set = Tensor_id.Set
end

module Label = struct
  type t = relation

  let identity = Identical
  let join = join

  let equal a b =
    match (a, b) with
    | Approximate x, Approximate y -> Precision.Set.equal x y
    | Equivalent, Equivalent | Identical, Identical | Unverifiable, Unverifiable
      ->
        true
    | _ -> false

  let pp = pp_relation
end

include Cluster_relation.Make (Id) (Label)

let pair src dst label =
  { Cluster.src = Set.singleton src; dst = Set.singleton dst; label }

let create id =
  { Cluster.src = Set.empty; dst = Set.singleton id; label = Identical }

let delete id =
  { Cluster.src = Set.singleton id; dst = Set.empty; label = Identical }
