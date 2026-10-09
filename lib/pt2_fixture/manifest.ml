open Err.Syntax
module Pin = Pt2_checkpoint_map.Document.Pin
module String_map = Schema_runtime.String_map

type member = { sha256 : Pt2_sha256.Digest.t; size : int64 }

type t = {
  archive : member * string;
  artifact_id : string;
  cases : string list;
  contract_sha256 : Pt2_sha256.Digest.t;
  graph_sha256 : Pt2_sha256.Digest.t;
  map_assets : Pin.t list;
  map_member : string;
  members : member String_map.t;
}

module Wire = struct
  type member = { sha256 : string; size : int64 }
  type archive = { name : string; a_sha256 : string; a_size : int64 }
  type map_v2 = { assets : Pin_wire.t list; member : string }

  type document = {
    archive : archive;
    artifact_id : string;
    cases : string list;
    contract_sha256 : string;
    graph_sha256 : string;
    map_v2 : map_v2;
    members : member String_map.t;
    payload : Jsont.json;
  }

  let member_jsont =
    Jsont.Object.map ~kind:"manifest member" (fun sha256 size ->
        { sha256; size })
    |> Jsont.Object.mem "sha256" Jsont.string
    |> Jsont.Object.mem "size" Jsont.int64
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let archive_jsont =
    Jsont.Object.map ~kind:"manifest archive" (fun name a_sha256 a_size ->
        { name; a_sha256; a_size })
    |> Jsont.Object.mem "name" Jsont.string
    |> Jsont.Object.mem "sha256" Jsont.string
    |> Jsont.Object.mem "size" Jsont.int64
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let map_v2_jsont =
    Jsont.Object.map ~kind:"manifest map_v2" (fun assets member ->
        { assets; member })
    |> Jsont.Object.mem "assets" (Jsont.list Pin_wire.jsont)
    |> Jsont.Object.mem "member" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"manifest"
      (fun
        archive
        artifact_id
        cases
        contract_sha256
        graph_sha256
        map_v2
        members
        payload
      ->
        {
          archive;
          artifact_id;
          cases;
          contract_sha256;
          graph_sha256;
          map_v2;
          members;
          payload;
        })
    |> Jsont.Object.mem "archive" archive_jsont
    |> Jsont.Object.mem "artifact_id" Jsont.string
    |> Jsont.Object.mem "cases" (Jsont.list Jsont.string)
    |> Jsont.Object.mem "contract_sha256" Jsont.string
    |> Jsont.Object.mem "graph_sha256" Jsont.string
    |> Jsont.Object.mem "map_v2" map_v2_jsont
    |> Jsont.Object.mem "members" (Jsont.Object.as_string_map member_jsont)
    |> Jsont.Object.mem "payload" Jsont.json
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish
end

let digest_of s =
  match Pt2_sha256.Digest.of_hex s with
  | Some d -> Err.return d
  | None -> Err.fail (`Bad_digest s)

let member_of (w : Wire.member) =
  let+ sha256 = digest_of w.sha256 in
  { sha256; size = w.size }

let of_string text =
  let* w =
    Jsont_bytesrw.decode_string Wire.jsont text
    |> Err.import ~pos:__POS__ (fun e -> `Manifest_decode e)
  in
  let* () =
    match w.payload with
    | Jsont.Null _ -> Err.return ()
    | _ ->
        Err.fail
          (`Field_clash
             {
               Fault.Field_clash.layer = Fault.Manifest;
               field = Fault.Payload;
               actual = "present";
               expected = "null (a slim bundle)";
             })
  in
  let* archive_digest = digest_of w.archive.a_sha256 in
  let* contract_sha256 = digest_of w.contract_sha256 in
  let* graph_sha256 = digest_of w.graph_sha256 in
  let* map_assets = Err.List.map Pin_wire.to_pin w.map_v2.assets in
  let* members =
    Err.List.map
      (fun (name, m) ->
        let+ m = member_of m in
        (name, m))
      (String_map.bindings w.members)
  in
  Err.return
    {
      archive =
        ({ sha256 = archive_digest; size = w.archive.a_size }, w.archive.name);
      artifact_id = w.artifact_id;
      cases = w.cases;
      contract_sha256;
      graph_sha256;
      map_assets;
      map_member = w.map_v2.member;
      members = String_map.of_seq (List.to_seq members);
    }

let required =
  [
    "captures.json";
    "cases.json";
    "contract.json";
    "data/constants/model_constants_config.json";
    "data/weights/model_weights_config.json";
    "models/model.json";
  ]

(* The files replay reads must be listed, and the two the cohort pins by digest
   (graph, contract) must be the very bytes the manifest lists for them. *)
let check_members (want : Cohort.entry) t =
  let need name =
    match String_map.find_opt name t.members with
    | Some m -> Err.return m
    | None -> Err.fail (`Member_missing name)
  in
  let* () = Err.List.iter (fun n -> Err.map (fun _ -> ()) (need n)) required in
  let* () =
    Err.List.iter
      (fun id ->
        let* _ = need (Printf.sprintf "cases/%s/inputs.pt" id) in
        let+ _ = need (Printf.sprintf "cases/%s/outputs.pt" id) in
        ())
      want.cases
  in
  let* graph = need "models/model.json" in
  let* () =
    Check.digest (Fault.Member "models/model.json") graph.sha256
      want.graph_sha256
  in
  let* contract = need "contract.json" in
  Check.digest (Fault.Member "contract.json") contract.sha256
    want.contract_sha256

let check (want : Cohort.entry) (t : t) =
  let* () =
    Check.field Fault.Manifest Fault.Artifact_id ~actual:t.artifact_id
      ~expected:want.artifact_id
  in
  let* () = Check.digest Fault.Graph t.graph_sha256 want.graph_sha256 in
  let* () =
    Check.digest Fault.Contract t.contract_sha256 want.contract_sha256
  in
  let* () =
    if t.cases = want.cases then Err.return ()
    else
      Err.fail
        (`Field_clash
           {
             Fault.Field_clash.layer = Fault.Manifest;
             field = Fault.Cases;
             actual = String.concat "," t.cases;
             expected = String.concat "," want.cases;
           })
  in
  let archive, name = t.archive in
  let* () =
    Check.field Fault.Archive Fault.File_name ~actual:name
      ~expected:want.archive.name
  in
  let* () = Check.digest Fault.Archive archive.sha256 want.archive.sha256 in
  let* () = Check.size Fault.Archive archive.size want.archive.size in
  let* () = check_members want t in
  let* () =
    Check.field Fault.Manifest Fault.Map_member ~actual:t.map_member
      ~expected:want.map_member
  in
  let* () =
    match String_map.find_opt t.map_member t.members with
    | None -> Err.fail (`Member_missing t.map_member)
    | Some m -> Check.digest Fault.Map m.sha256 want.map_sha256
  in
  Err.List.iter
    (fun (asset : Pin.t) ->
      match
        List.find_opt
          (fun (p : Pin.t) -> String.equal p.name asset.name)
          want.sources
      with
      | None -> Err.fail (`Source_pin_missing asset.name)
      | Some pinned -> Check.pin (Fault.Source asset.name) asset pinned)
    t.map_assets
