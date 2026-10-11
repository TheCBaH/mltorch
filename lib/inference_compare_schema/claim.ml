module W = Wire
module Precision = W.Text (Id.Precision)

type t =
  | Approximate of Id.Precision.t list
  | Equivalent
  | Identical
  | Unverifiable

(* Strength: Identical > Equivalent > Approximate > Unverifiable. [rank] is
   the position in that order, not a numeric claim. *)
let rank = function
  | Identical -> 3
  | Equivalent -> 2
  | Approximate _ -> 1
  | Unverifiable -> 0

let sort_unique ps = List.sort_uniq Id.Precision.compare ps

(* The claim that survives composing two relations: the weaker of the two.
   Two approximate histories pool their representation identities; ordered
   conversion recipes keep repeated rounding steps elsewhere. *)
let join a b =
  match (a, b) with
  | Approximate x, Approximate y -> Approximate (sort_unique (x @ y))
  | _ -> if rank a <= rank b then a else b

let equal a b =
  match (a, b) with
  | Approximate x, Approximate y ->
      List.equal Id.Precision.equal (sort_unique x) (sort_unique y)
  | Equivalent, Equivalent | Identical, Identical | Unverifiable, Unverifiable
    ->
      true
  | _ -> false

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "approximate" ->
          let* ps = W.Obj.required o "precisions" (W.list Precision.decode) in
          Ok (Approximate ps)
      | "equivalent" -> Ok Equivalent
      | "identical" -> Ok Identical
      | "unverifiable" -> Ok Unverifiable
      | tag -> W.unknown_tag ~kind:"claim" tag)

let encode = function
  | Approximate ps ->
      W.tagged "approximate"
        [ W.field "precisions" (W.enc_list Precision.encode) ps ]
  | Equivalent -> W.tagged "equivalent" []
  | Identical -> W.tagged "identical" []
  | Unverifiable -> W.tagged "unverifiable" []
