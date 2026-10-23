open Err.Syntax

module Access = struct
  type t = {
    dma : bool option;
    guest_load : bool option;
    guest_store : bool option;
    host_load : bool option;
    instruction_fetch : bool option;
  }

  let jsont =
    let flag name = Jsont.Object.mem name (Jsont.option Jsont.bool) in
    Jsont.Object.map ~kind:"Access"
      (fun dma guest_load guest_store host_load instruction_fetch ->
        { dma; guest_load; guest_store; host_load; instruction_fetch })
    |> flag "dma" |> flag "guest_load" |> flag "guest_store" |> flag "host_load"
    |> flag "instruction_fetch" |> Jsont.Object.finish
end

module Region = struct
  type t = {
    access : Access.t;
    base_address : int64;
    kind : string;
    name : string;
    size_bytes : int64;
  }

  let jsont =
    Jsont.Object.map ~kind:"Region"
      (fun access base_address kind name size_bytes ->
        { access; base_address; kind; name; size_bytes })
    |> Jsont.Object.mem "access" Access.jsont
    |> Jsont.Object.mem "base_address" Jsont.int64
    |> Jsont.Object.mem "type" Jsont.string
    |> Jsont.Object.mem "name" Jsont.string
    |> Jsont.Object.mem "size_bytes" Jsont.int64
    |> Jsont.Object.finish
end

module Configuration = struct
  type t = {
    backend : string;
    extensions : string list;
    id : string;
    interface : string;
    name : string;
    regions : Region.t list;
  }

  let simulation =
    Jsont.Object.map ~kind:"Simulation" (fun backend interface ->
        (backend, interface))
    |> Jsont.Object.mem "backend" Jsont.string
    |> Jsont.Object.mem "interface" Jsont.string
    |> Jsont.Object.finish

  let isa =
    Jsont.Object.map ~kind:"Isa" Fun.id
    |> Jsont.Object.mem "extensions" (Jsont.list Jsont.string)
    |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"Configuration"
      (fun (backend, interface) extensions id name regions ->
        { backend; extensions; id; interface; name; regions })
    |> Jsont.Object.mem "simulation" simulation
    |> Jsont.Object.mem "isa" isa
    |> Jsont.Object.mem "id" Jsont.string
    |> Jsont.Object.mem "name" Jsont.string
    |> Jsont.Object.mem "memory_regions" (Jsont.list Region.jsont)
    |> Jsont.Object.finish
end

module Artifact = struct
  type t = {
    command : string list;
    configuration_id : string option;
    path : string;
    sha256 : string;
  }

  let launch =
    Jsont.Object.map ~kind:"Launch" Fun.id
    |> Jsont.Object.mem "command" (Jsont.list Jsont.string)
    |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"Artifact"
      (fun command configuration_id path sha256 ->
        { command; configuration_id; path; sha256 })
    |> Jsont.Object.mem "launch" launch
    |> Jsont.Object.mem "configuration_id" (Jsont.option Jsont.string)
    |> Jsont.Object.mem "path" Jsont.string
    |> Jsont.Object.mem "sha256" Jsont.string
    |> Jsont.Object.finish
end

type t = {
  archive : string;
  artifacts : Artifact.t list;
  configurations : Configuration.t list;
}

module Id_mismatch = struct
  type t = { computed : string; declared : string }
end

type error =
  [ Coral_json.error
  | `Id_mismatch of Id_mismatch.t
  | `Manifest_decode of string
  | `Schema_version of string ]

let pp_error ppf : error -> unit = function
  | #Coral_json.error as e -> Coral_json.pp_error ppf e
  | `Id_mismatch { Id_mismatch.computed; declared } ->
      Fmt.pf ppf "configuration %s: its record hashes to %s" declared computed
  | `Manifest_decode msg -> Fmt.pf ppf "failed to decode the manifest: %s" msg
  | `Schema_version v -> Fmt.pf ppf "unsupported manifest schema_version %S" v

(* The members the whole file is read through: configurations stay generic
   until their IDs are checked. *)
module File = struct
  type t = {
    archive : string;
    artifacts : Artifact.t list;
    configurations : Jsont.json list;
    schema_version : string;
  }

  let archive =
    Jsont.Object.map ~kind:"Archive" Fun.id
    |> Jsont.Object.mem "name" Jsont.string
    |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"Manifest"
      (fun archive artifacts configurations schema_version ->
        { archive; artifacts; configurations; schema_version })
    |> Jsont.Object.mem "archive" archive
    |> Jsont.Object.mem "artifacts" (Jsont.list Artifact.jsont)
    |> Jsont.Object.mem "configurations" (Jsont.list Jsont.json)
    |> Jsont.Object.mem "schema_version" Jsont.string
    |> Jsont.Object.finish
end

let decode codec json =
  Jsont.Json.decode codec json
  |> Err.import ~pos:__POS__ (fun e -> `Manifest_decode e)

let name_only =
  Jsont.Object.map ~kind:"Configuration" Fun.id
  |> Jsont.Object.mem "name" Jsont.string
  |> Jsont.Object.finish

let configuration_id (json : Jsont.json) =
  let* name = decode name_only json in
  let body =
    match json with
    | Object (mems, meta) ->
        Jsont.Object
          (List.filter (fun ((m, _), _) -> not (String.equal m "id")) mems, meta)
    | other -> other
  in
  let* canonical = Coral_json.to_string body in
  let h = Pt2_sha256.create () in
  Pt2_sha256.add_string h canonical;
  let hex = Pt2_sha256.Digest.to_hex (Pt2_sha256.finish h) in
  Err.return ("coralnpu:" ^ name ^ ":" ^ String.sub hex 0 20)

let checked json =
  let* c = decode Configuration.jsont json in
  let* computed = configuration_id json in
  if String.equal computed c.Configuration.id then Err.return c
  else Err.fail (`Id_mismatch { Id_mismatch.computed; declared = c.id })

let of_string s =
  let* (file : File.t) =
    Jsont_bytesrw.decode_string File.jsont s
    |> Err.import ~pos:__POS__ (fun e -> `Manifest_decode e)
  in
  let* () =
    if String.equal file.schema_version "1.0" then Err.return ()
    else Err.fail (`Schema_version file.schema_version)
  in
  let* configurations = Err.List.map checked file.configurations in
  Err.return
    { archive = file.archive; artifacts = file.artifacts; configurations }
