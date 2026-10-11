open Inference_compare_schema
module Codecs = Inference_compare_codecs
module E = Compose_error

let ( let* ) = Err.bind
let fail = E.fail

let rec all f = function
  | [] -> Ok []
  | x :: rest ->
      let* y = f x in
      let* ys = all f rest in
      Ok (y :: ys)

let nkey (n : Node_ref.t) = Wire.canonical_string (Node_ref.encode n)
let tkey (t : Tensor_ref.t) = Wire.canonical_string (Tensor_ref.encode t)
let sort_tensors l = List.sort_uniq Tensor_ref.compare l
let mem_tensor t l = List.exists (Tensor_ref.equal t) l

(* ---- union-find over string keys ---------------------------------------- *)

module Uf = struct
  type t = (string, string) Hashtbl.t

  let create () : t = Hashtbl.create 64

  let rec find (t : t) x =
    match Hashtbl.find_opt t x with
    | None ->
        Hashtbl.replace t x x;
        x
    | Some p when String.equal p x -> x
    | Some p ->
        let r = find t p in
        Hashtbl.replace t x r;
        r

  let union t a b =
    let ra = find t a and rb = find t b in
    if not (String.equal ra rb) then Hashtbl.replace t ra rb
end

(* ---- derived identifiers ------------------------------------------------- *)

type ctx = {
  hashers : Codecs.Payload_hasher.Registry.t;
  algorithm : Id.Digest_algorithm.t;
}

let short ctx text =
  match Codecs.Payload_hasher.digest_string ctx.hashers ctx.algorithm text with
  | Ok d -> Ok (String.sub (Digest.Payload.hex d) 0 16)
  | Error _ -> fail (E.Invalid_identifier "digest backend unavailable")

let ident of_string prefix ctx text =
  let* h = short ctx text in
  let s = prefix ^ h in
  match of_string s with
  | Ok v -> Ok v
  | Error _ -> fail (E.Invalid_identifier s)

let recipe_id = ident Id.Recipe.of_string "r-"
let unit_id = ident Id.Unit.of_string "c-"
let relation_id = ident Id.Tensor_relation.of_string "rel-"
let route_id = ident Id.Value_route.of_string "route-"
let pair_id = ident Id.Member_pair.of_string "pair-"
let comparison_id = ident Id.Comparison.of_string "cmp-"
let unit_ids us = List.map (fun (u : Matched_unit.t) -> u.unit_id) us
let has_unit id us = List.exists (Id.Unit.equal id) us
let find_first f l = List.find_map f l
