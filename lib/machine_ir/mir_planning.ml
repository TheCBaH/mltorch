module Precision = struct
  type t = F32 | F64

  let name = function F32 -> "f32" | F64 -> "f64"
  let of_name = function "f32" -> Some F32 | "f64" -> Some F64 | _ -> None
end

module Fma = struct
  type t = Dots | Exact | Forbidden | Relaxed_madd

  let name = function
    | Dots -> "dots"
    | Exact -> "exact"
    | Forbidden -> "forbidden"
    | Relaxed_madd -> "relaxed_madd"

  let of_name = function
    | "dots" -> Some Dots
    | "exact" -> Some Exact
    | "forbidden" -> Some Forbidden
    | "relaxed_madd" -> Some Relaxed_madd
    | _ -> None
end

module Contraction_engine = struct
  type t = { desc : Contraction.t; engine : Contraction.Select.Engine.t }
end

module Capability = struct
  type t =
    | Contraction_engine of Contraction_engine.t
    | Fused_multiply_add
    | Helper of string
    | Vector_bits of int64

  let compare = Stdlib.compare

  let to_string = function
    | Contraction_engine c ->
        "contraction:"
        ^ Contraction.Select.Engine.name c.engine
        ^ ":" ^ Contraction.name c.desc
    | Fused_multiply_add -> "fma"
    | Helper h -> "helper:" ^ h
    | Vector_bits b -> "vector_bits:" ^ Int64.to_string b

  let of_string s =
    if s = "fma" then Some Fused_multiply_add
    else
      match String.index_opt s ':' with
      | None -> None
      | Some i -> (
          let tag = String.sub s 0 i in
          let rest = String.sub s (i + 1) (String.length s - i - 1) in
          match tag with
          | "contraction" -> (
              match String.index_opt rest ':' with
              | None -> None
              | Some j -> (
                  let engine =
                    Contraction.Select.Engine.of_name (String.sub rest 0 j)
                  in
                  let desc =
                    Contraction.of_name
                      (String.sub rest (j + 1) (String.length rest - j - 1))
                  in
                  match (engine, desc) with
                  | Some engine, Some desc ->
                      Some (Contraction_engine { desc; engine })
                  | _ -> None))
          | "helper" when rest <> "" -> Some (Helper rest)
          | "vector_bits" ->
              Option.map (fun b -> Vector_bits b) (Int64.of_string_opt rest)
          | _ -> None)
end

type t = {
  subject : string;
  policy : string;
  schedule : string;
  precision : Precision.t;
  lanes : Mir_type.Lanes.t;
  fma : Fma.t;
  capabilities : Capability.t list;
}

let make ~subject ~policy ~schedule ~precision ~lanes ~fma ~capabilities =
  {
    subject;
    policy;
    schedule;
    precision;
    lanes;
    fma;
    capabilities = List.sort_uniq Capability.compare capabilities;
  }

let equal (a : t) b = a = b

let contraction_engines t =
  List.filter_map
    (function Capability.Contraction_engine c -> Some c | _ -> None)
    t.capabilities

let contraction_engine t desc =
  List.find_map
    (fun (c : Contraction_engine.t) ->
      if c.desc = desc then Some c.engine else None)
    (contraction_engines t)

(* A field holds no newline; the serialized form is line-oriented. *)
let clean s =
  if String.contains s '\n' then
    invalid_arg "Mir_planning: a newline in a field"
  else s

let to_string t =
  String.concat "\n"
    [
      "subject=" ^ clean t.subject;
      "policy=" ^ clean t.policy;
      "schedule=" ^ clean t.schedule;
      "precision=" ^ Precision.name t.precision;
      "lanes=" ^ string_of_int (Mir_type.Lanes.to_int t.lanes);
      "fma=" ^ Fma.name t.fma;
      "capabilities="
      ^ String.concat "," (List.map Capability.to_string t.capabilities);
    ]

let of_string s =
  let malformed why = Error (`Malformed_summary why) in
  let fields =
    List.filter_map
      (fun line ->
        match String.index_opt line '=' with
        | None -> None
        | Some i ->
            Some
              ( String.sub line 0 i,
                String.sub line (i + 1) (String.length line - i - 1) ))
      (String.split_on_char '\n' s)
  in
  let get k = List.assoc_opt k fields in
  match
    ( get "subject",
      get "policy",
      get "schedule",
      Option.bind (get "precision") Precision.of_name,
      Option.bind (get "lanes") int_of_string_opt,
      Option.bind (get "fma") Fma.of_name,
      get "capabilities" )
  with
  | ( Some subject,
      Some policy,
      Some schedule,
      Some precision,
      Some lanes,
      Some fma,
      Some caps ) -> (
      let caps =
        if caps = "" then Some []
        else
          List.fold_right
            (fun c acc ->
              match (acc, Capability.of_string c) with
              | Some l, Some c -> Some (c :: l)
              | _ -> None)
            (String.split_on_char ',' caps)
            (Some [])
      in
      if lanes < 1 || lanes > Mir_type.max_lanes then malformed "lanes"
      else if List.length fields <> 7 then malformed "fields"
      else
        match caps with
        | None -> malformed "capabilities"
        | Some capabilities ->
            Ok
              (make ~subject ~policy ~schedule ~precision
                 ~lanes:(Mir_type.Lanes.of_int lanes)
                 ~fma ~capabilities))
  | _ -> malformed "missing or unreadable field"

type mismatch =
  [ `Missing_planning_summary
  | `Planning_contraction_mismatch
  | `Planning_subject_mismatch of string * string
  | `Unauthorized_contraction of Fma.t ]

let pp_mismatch fmt : [< mismatch ] -> unit = function
  | `Missing_planning_summary -> Fmt.string fmt "no planning summary"
  | `Planning_contraction_mismatch ->
      Fmt.string fmt "contraction engines differ from the planned SSA"
  | `Planning_subject_mismatch (e, f) ->
      Fmt.pf fmt "planning summary is for %s, not %s" f e
  | `Unauthorized_contraction f ->
      Fmt.pf fmt "a fused multiply-add under fma=%s" (Fma.name f)

let admit summary ~subject ~contracts =
  match summary with
  | None -> Error `Missing_planning_summary
  | Some t ->
      if not (String.equal t.subject subject) then
        Error (`Planning_subject_mismatch (subject, t.subject))
      else if contracts && t.fma = Fma.Forbidden then
        Error (`Unauthorized_contraction t.fma)
      else Ok t
