(* The worker request. See the .mli. *)

module Hard = Me_limits.Hard

(* Every checked constructor below crosses into Jsont at the end, and this is
   the one place the framework wrapper is dropped. Named, per CLAUDE.md, so it
   reads as a deliberate crossing rather than a lost backtrace; [Err.export]
   supplies the [Export] mark that makes the drop visible. *)
let or_jsont pp r =
  match Err.export ~pos:__POS__ r with
  | Ok v -> v
  | Error k -> Jsont.Error.msgf Jsont.Meta.none "%a" pp k

(* --- request identity --------------------------------------------------- *)

module Request_id = struct
  type t = string

  let is_hex c = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')

  (* Canonical lowercase hyphenated UUID: 8-4-4-4-12. Uppercase is NOT accepted:
     two spellings of one id would make [epoch] a non-injective projection, and
     the epoch is what a delta binds against. *)
  let is_uuid s =
    String.length s = 36
    &&
    let ok = ref true in
    String.iteri
      (fun i c ->
        match i with
        | 8 | 13 | 18 | 23 -> if c <> '-' then ok := false
        | _ -> if not (is_hex c) then ok := false)
      s;
    !ok

  (* 1-10 decimal digits, no leading zero, and BELOW the ceiling. The value goes
     to [Int64.of_string]: an [int] parse wraps under js_of_ocaml before the
     comparison, and a check on a wrapped value is not a check. *)
  let is_seq s =
    let n = String.length s in
    n >= 1 && n <= 10
    && (n = 1 || s.[0] <> '0')
    && String.for_all (fun c -> c >= '0' && c <= '9') s
    &&
    match Int64.of_string_opt s with
    | None -> false
    | Some v -> Int64.compare v Hard.max_seq_exclusive < 0

  let of_string s =
    (* The UUID's own length is fixed, so the split point is fixed too: the
       LAST hyphen would be ambiguous with the four inside a UUID. *)
    if String.length s > 37 && s.[36] = '-' then
      let uuid = String.sub s 0 36 in
      let seq = String.sub s 37 (String.length s - 37) in
      if is_uuid uuid && is_seq seq then Err.return s
      else Err.fail `Malformed_request_id
    else Err.fail `Malformed_request_id

  let epoch t = String.sub t 0 36
  let to_string t = t
  let equal = String.equal

  let jsont =
    Jsont.map ~kind:"requestId"
      ~dec:(fun s ->
        or_jsont
          (fun fmt `Malformed_request_id ->
            Fmt.string fmt "malformed request id")
          (of_string s))
      ~enc:to_string Jsont.string
end

(* --- what a detail request names ---------------------------------------- *)

module Detail_key = struct
  type t = { parent_graph : string; value : Graph_ir.Tensor_id.t }
  type invalid = [ `Parent_too_long | `Derived_id_too_long ]

  let pp_invalid fmt : [< invalid ] -> unit = function
    | `Parent_too_long -> Fmt.string fmt "detail key parent graph is too long"
    | `Derived_id_too_long -> Fmt.string fmt "derived detail id is too long"

  let parent_node t = Core.Pretty.to_string Graph_ir.Tensor_id.pp t.value

  (* The same string [Me_ids.detail] assembles, built here without a ceiling so
     that [id] is total: the LENGTH rule belongs to [create]/[validate], which
     have a profile, and recomputing the string is not where it lives. *)
  let id t =
    Printf.sprintf "expr/%s/%s/t%d" t.parent_graph (parent_node t)
      (Graph_ir.Tensor_id.to_int t.value)

  let equal a b =
    String.equal a.parent_graph b.parent_graph
    && Graph_ir.Tensor_id.equal a.value b.value

  let validate ~limits t =
    let max = limits.Me_limits.Limits.max_id_bytes in
    if String.length t.parent_graph > max then
      Err.fail (`Invalid_detail_key `Parent_too_long)
    else if String.length (id t) > max then
      Err.fail (`Invalid_detail_key `Derived_id_too_long)
    else Err.return ()

  let create ~limits ~parent_graph ~value =
    let open Err.Syntax in
    let t = { parent_graph; value } in
    let+ () = validate ~limits t in
    t

  let jsont =
    (* [Limits.trusted] is the FIXED half: a parameterless codec cannot pick a
       profile, and picking the wrong one is the defect [Wire_limits] exists to
       prevent. [Request.jsont] runs [validate] under the decoded profile. *)
    Jsont.Object.map ~kind:"detailKey" (fun parent_graph value ->
        or_jsont pp_invalid
          (Err.map_error
             (fun (`Invalid_detail_key e) -> e)
             (create ~limits:Me_limits.Limits.trusted ~parent_graph
                ~value:(Graph_ir.Tensor_id.of_int value))))
    |> Jsont.Object.mem "parentGraph" Jsont.string ~enc:(fun t ->
        t.parent_graph)
    |> Jsont.Object.mem "value" Jsont.int ~enc:(fun t ->
        Graph_ir.Tensor_id.to_int t.value)
    |> Jsont.Object.finish
end

(* --- where the bytes came from ------------------------------------------ *)

module Source = struct
  module Origin = struct
    module Catalog = struct
      type t = { url : string; ref_ : string; verified_sha256 : string }
    end

    type t = Local | Catalog of Catalog.t
  end

  type t = {
    origin : Origin.t;
    name : string;
    bytes : int64;
    format : [ `Model_json | `Pt2 ];
  }

  module Invalid = struct
    type kind =
      | Negative_bytes
      | Bad_digest
      | Name_too_long
      | Url_too_long
      | Ref_too_long

    type t = { kind : kind }
  end

  let pp_invalid fmt (i : Invalid.t) =
    Fmt.string fmt
      (match i.Invalid.kind with
      | Invalid.Negative_bytes -> "source byte count is negative"
      | Invalid.Bad_digest -> "source digest is not 64 lowercase hex bytes"
      | Invalid.Name_too_long -> "source name is too long"
      | Invalid.Url_too_long -> "source url is too long"
      | Invalid.Ref_too_long -> "source ref is too long")

  let fail kind = Err.fail (`Invalid_source { Invalid.kind })

  let is_digest s =
    String.length s = Hard.max_digest_bytes
    && String.for_all
         (fun c -> (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
         s

  let create ~limits ~origin ~name ~bytes ~format =
    let open Err.Syntax in
    let* () =
      if Int64.compare bytes 0L < 0 then fail Invalid.Negative_bytes
      else Err.return ()
    in
    let* () =
      if String.length name > limits.Me_limits.Limits.max_label_bytes then
        fail Invalid.Name_too_long
      else Err.return ()
    in
    let+ () =
      match origin with
      | Origin.Local -> Err.return ()
      | Origin.Catalog c ->
          let max = limits.Me_limits.Limits.max_url_bytes in
          if String.length c.Origin.Catalog.url > max then
            fail Invalid.Url_too_long
          else if String.length c.Origin.Catalog.ref_ > max then
            fail Invalid.Ref_too_long
          else if not (is_digest c.Origin.Catalog.verified_sha256) then
            fail Invalid.Bad_digest
          else Err.return ()
    in
    { origin; name; bytes; format }

  let verified_sha256 t =
    match t.origin with
    | Origin.Local -> None
    | Origin.Catalog c -> Some c.Origin.Catalog.verified_sha256

  let format_jsont =
    Jsont.enum ~kind:"sourceFormat"
      [ ("model_json", `Model_json); ("pt2", `Pt2) ]

  let catalog_jsont =
    Jsont.Object.map ~kind:"catalogOrigin" (fun url ref_ verified_sha256 ->
        { Origin.Catalog.url; ref_; verified_sha256 })
    |> Jsont.Object.mem "url" Jsont.string ~enc:(fun c -> c.Origin.Catalog.url)
    |> Jsont.Object.mem "ref" Jsont.string ~enc:(fun c -> c.Origin.Catalog.ref_)
    |> Jsont.Object.mem "verifiedSha256" Jsont.string ~enc:(fun c ->
        c.Origin.Catalog.verified_sha256)
    |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"source" (fun catalog name bytes format ->
        or_jsont pp_invalid
          (Err.map_error
             (fun (`Invalid_source i) -> i)
             (create ~limits:Me_limits.Limits.trusted
                ~origin:
                  (match catalog with
                  | None -> Origin.Local
                  | Some c -> Origin.Catalog c)
                ~name ~bytes ~format)))
    |> Jsont.Object.opt_mem "catalog" catalog_jsont ~enc:(fun t ->
        match t.origin with Origin.Local -> None | Origin.Catalog c -> Some c)
    |> Jsont.Object.mem "name" Jsont.string ~enc:(fun t -> t.name)
    |> Jsont.Object.mem "bytes" Jsont.int64_as_string ~enc:(fun t -> t.bytes)
    |> Jsont.Object.mem "format" format_jsont ~enc:(fun t -> t.format)
    |> Jsont.Object.finish
end

(* --- what changes the projection ---------------------------------------- *)

module Options = struct
  type namespace = Structural | Module

  type t = {
    stages : Me_session.Capability.graph_stage list;
    fold : bool;
    verify_symbolic : Map_verify.Effort.t option;
    namespace : namespace;
  }

  (* NORMALISED against [all_stages], which both removes duplicates and imposes
     the constructor order in one pass -- so the result is bounded by that
     type's cardinality by construction, and two requests differing only in the
     order a user typed them are the same request. *)
  let normalise stages =
    List.filter
      (fun s -> List.exists (fun x -> x = s) stages)
      Me_session.Capability.all_stages

  let create ~stages ~fold ~verify_symbolic ~namespace =
    match normalise stages with
    | [] -> Err.fail `Invalid_options
    | stages -> Err.return { stages; fold; verify_symbolic; namespace }

  let stage_jsont =
    Jsont.enum ~kind:"graphStage"
      (List.map
         (fun s -> (Me_session.Capability.stage_name s, s))
         Me_session.Capability.all_stages)

  let effort_jsont =
    Jsont.enum ~kind:"effort"
      (List.map
         (fun e -> (Map_verify.Effort.to_string e, e))
         Map_verify.Effort.all)

  let namespace_jsont =
    Jsont.enum ~kind:"namespaceMode"
      [ ("structural", Structural); ("module", Module) ]

  let jsont =
    Jsont.Object.map ~kind:"options"
      (fun stages fold verify_symbolic namespace ->
        or_jsont
          (fun fmt `Invalid_options ->
            Fmt.string fmt "a request must ask for at least one stage")
          (create ~stages ~fold ~verify_symbolic ~namespace))
    |> Jsont.Object.mem "stages" (Jsont.list stage_jsont) ~enc:(fun t ->
        t.stages)
    |> Jsont.Object.mem "fold" Jsont.bool ~enc:(fun t -> t.fold)
    |> Jsont.Object.opt_mem "verifySymbolic" effort_jsont ~enc:(fun t ->
        t.verify_symbolic)
    |> Jsont.Object.mem "namespace" namespace_jsont ~enc:(fun t -> t.namespace)
    |> Jsont.Object.finish
end

(* --- the request -------------------------------------------------------- *)

module Request = struct
  module Build_session = struct
    type t = {
      id : Request_id.t;
      source : Source.t;
      options : Options.t;
      limits : Me_limits.Wire_limits.t;
    }
  end

  module Build_detail = struct
    type t = {
      id : Request_id.t;
      source : Source.t;
      options : Options.t;
      limits : Me_limits.Wire_limits.t;
      key : Detail_key.t;
    }
  end

  type t = Build_session of Build_session.t | Build_detail of Build_detail.t

  type error =
    [ `Malformed_request_id
    | `Invalid_options
    | `Invalid_source of Source.Invalid.t
    | `Invalid_limits of Me_limits.error
    | `Invalid_detail_key of Detail_key.invalid ]

  let pp_error fmt : [< error ] -> unit = function
    | `Malformed_request_id -> Fmt.string fmt "malformed request id"
    | `Invalid_options ->
        Fmt.string fmt "a request must ask for at least one stage"
    | `Invalid_source i -> Source.pp_invalid fmt i
    | `Invalid_limits e -> Me_limits.pp_error fmt e
    | `Invalid_detail_key i -> Detail_key.pp_invalid fmt i

  (* REVALIDATION, not a witness check. A [Source.t] keeps no record of the
     profile it was built under, so "was it built under this one" is
     unobtainable; re-running the profile-dependent checks under the request's
     own profile is obtainable and is what [handle] will enforce. *)
  let revalidate ~limits ~source ~key =
    let open Err.Syntax in
    let l = Me_limits.Wire_limits.limits limits in
    let* _ =
      Source.create ~limits:l ~origin:source.Source.origin
        ~name:source.Source.name ~bytes:source.Source.bytes
        ~format:source.Source.format
    in
    match key with
    | None -> Err.return ()
    | Some k -> Detail_key.validate ~limits:l k

  let build_session ~id ~source ~options ~limits =
    let open Err.Syntax in
    let+ () = revalidate ~limits ~source ~key:None in
    Build_session { Build_session.id; source; options; limits }

  let build_detail ~id ~source ~options ~limits ~key =
    let open Err.Syntax in
    let+ () = revalidate ~limits ~source ~key:(Some key) in
    Build_detail { Build_detail.id; source; options; limits; key }

  let id = function
    | Build_session s -> s.Build_session.id
    | Build_detail d -> d.Build_detail.id

  let epoch t = Request_id.epoch (id t)

  let key = function
    | Build_session _ -> None
    | Build_detail d -> Some d.Build_detail.key

  let limits = function
    | Build_session s -> s.Build_session.limits
    | Build_detail d -> d.Build_detail.limits

  let source = function
    | Build_session s -> s.Build_session.source
    | Build_detail d -> d.Build_detail.source

  let options = function
    | Build_session s -> s.Build_session.options
    | Build_detail d -> d.Build_detail.options

  (* STAGED. Jsont decodes an object's fields independently, but [Source.create]
     and [Detail_key.validate] need the DECODED profile -- so the members are
     decoded as raw pieces and validated in a fixed order, each step taking what
     the step before it produced. And the last step is [build_session] /
     [build_detail], so encoder and decoder share one domain rather than two
     that happen to agree. *)
  let jsont =
    Jsont.Object.map ~kind:"request" (fun id limits options source key ->
        or_jsont pp_error
          (let open Err.Syntax in
           let* id = Request_id.of_string id in
           match key with
           | None -> build_session ~id ~source ~options ~limits
           | Some key -> build_detail ~id ~source ~options ~limits ~key))
    |> Jsont.Object.mem "id" Jsont.string ~enc:(fun t ->
        Request_id.to_string (id t))
    |> Jsont.Object.mem "limits" Me_limits.Wire_limits.jsont ~enc:limits
    |> Jsont.Object.mem "options" Options.jsont ~enc:options
    |> Jsont.Object.mem "source" Source.jsont ~enc:source
    |> Jsont.Object.opt_mem "key" Detail_key.jsont ~enc:key
    |> Jsont.Object.finish
end
