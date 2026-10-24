open Err.Syntax

module Role = struct
  type t = Csr | Ddr | Dtcm | Itcm | Mmio | Ram | Rom

  let of_kind = function
    | "csr" -> Some Csr
    | "ddr" -> Some Ddr
    | "dtcm" -> Some Dtcm
    | "itcm" -> Some Itcm
    | "mmio" -> Some Mmio
    | "ram" -> Some Ram
    | "rom" -> Some Rom
    | _ -> None

  let to_string = function
    | Csr -> "csr"
    | Ddr -> "ddr"
    | Dtcm -> "dtcm"
    | Itcm -> "itcm"
    | Mmio -> "mmio"
    | Ram -> "ram"
    | Rom -> "rom"
end

module Simulator = struct
  type t = Mpact | Rtl | Subsystem

  let of_backend ~interface = function
    | "mpact" -> Some Mpact
    | "verilator" when String.equal interface "subsystem-cli" -> Some Subsystem
    | "verilator" -> Some Rtl
    | _ -> None

  let to_string = function
    | Mpact -> "mpact"
    | Rtl -> "rtl"
    | Subsystem -> "subsystem"
end

module Region = struct
  type t = {
    base : int64;
    dma_reachable : bool;
    htif_reachable : bool;
    loadable : bool;
    name : string;
    role : Role.t;
    size : int64;
    writable : bool;
  }
end

type t = {
  artifact : string;
  artifact_sha256 : string;
  command : string list;
  configuration_id : string;
  extensions : string list;
  name : string;
  regions : Region.t list;
  simulator : Simulator.t;
}

module Artifact_digest = struct
  type t = { actual : string; expected : string; path : string }
end

module Artifact_read = struct
  type t = { message : string; path : string }
end

type error =
  [ Coral_manifest.error
  | `Artifact_digest of Artifact_digest.t
  | `Artifact_read of Artifact_read.t
  | `No_artifact of string
  | `Unknown_backend of string
  | `Unknown_configuration of string
  | `Unknown_region_kind of string ]

let pp_error ppf : error -> unit = function
  | #Coral_manifest.error as e -> Coral_manifest.pp_error ppf e
  | `Artifact_digest { Artifact_digest.actual; expected; path } ->
      Fmt.pf ppf "%s: sha256 is %s, the manifest pins %s" path actual expected
  | `Artifact_read { Artifact_read.message; path = _ } ->
      (* The system's message already names the path. *)
      Fmt.string ppf message
  | `No_artifact id -> Fmt.pf ppf "no artifact simulates %s" id
  | `Unknown_backend b -> Fmt.pf ppf "unknown simulation backend %S" b
  | `Unknown_configuration n -> Fmt.pf ppf "no configuration named %S" n
  | `Unknown_region_kind k -> Fmt.pf ppf "unknown memory region type %S" k

let baseline = [ "core-mini-axi"; "core-mini-highmem-axi"; "mpact-v2" ]
let manifest_file = "hardware-configurations.json"

(* The two facts of the interface comment, by role. A ROM holds the boot image
   it was built with, and the subsystem's DDR is an aperture with no memory
   behind it: neither is loadable. *)
let memory (r : Role.t) =
  match r with Dtcm | Itcm | Ram -> true | Csr | Ddr | Mmio | Rom -> false

(* The subsystem has no HTIF at all. *)
let htif_reachable (s : Simulator.t) (r : Role.t) =
  match (s, r) with
  | (Mpact | Rtl), (Dtcm | Ram) -> true
  | (Mpact | Rtl), (Csr | Ddr | Itcm | Mmio | Rom) | Subsystem, _ -> false

let region simulator (r : Coral_manifest.Region.t) =
  match Role.of_kind r.kind with
  | None -> Err.fail (`Unknown_region_kind r.kind)
  | Some role ->
      Err.return
        {
          Region.base = r.base_address;
          dma_reachable = r.access.dma = Some true;
          htif_reachable = htif_reachable simulator role;
          loadable = memory role && r.access.host_load = Some true;
          name = r.name;
          role;
          size = r.size_bytes;
          writable = r.access.guest_store = Some true;
        }

let of_manifest (m : Coral_manifest.t) name =
  let* (c : Coral_manifest.Configuration.t) =
    List.find_opt
      (fun (c : Coral_manifest.Configuration.t) -> String.equal c.name name)
      m.configurations
    |> Err.of_option (`Unknown_configuration name)
  in
  let* simulator =
    Simulator.of_backend ~interface:c.interface c.backend
    |> Err.of_option (`Unknown_backend c.backend)
  in
  let* (a : Coral_manifest.Artifact.t) =
    List.find_opt
      (fun (a : Coral_manifest.Artifact.t) -> a.configuration_id = Some c.id)
      m.artifacts
    |> Err.of_option (`No_artifact c.id)
  in
  let* regions = Err.List.map (region simulator) c.regions in
  Err.return
    {
      artifact = a.path;
      artifact_sha256 = a.sha256;
      command = a.command;
      configuration_id = c.id;
      extensions = c.extensions;
      name;
      regions;
      simulator;
    }

let read_failure path = function
  | Sys_error message ->
      Err.fail (`Artifact_read { Artifact_read.message; path })
  | e -> raise e

let sha256_file path =
  match
    In_channel.with_open_bin path (fun ic ->
        let h = Pt2_sha256.create () in
        let buf = Bytes.create 65536 in
        let rec go () =
          match In_channel.input ic buf 0 (Bytes.length buf) with
          | 0 -> Pt2_sha256.Digest.to_hex (Pt2_sha256.finish h)
          | n ->
              Pt2_sha256.add_string h ~len:n (Bytes.unsafe_to_string buf);
              go ()
        in
        go ())
  with
  | hex -> Err.return hex
  | exception e -> read_failure path e

let verify ~dir p =
  let path = Filename.concat dir p.artifact in
  let* actual = sha256_file path in
  if String.equal actual p.artifact_sha256 then Err.return ()
  else
    Err.fail
      (`Artifact_digest
         { Artifact_digest.actual; expected = p.artifact_sha256; path })

let load ~dir name =
  let path = Filename.concat dir manifest_file in
  let* text =
    match In_channel.with_open_bin path In_channel.input_all with
    | s -> Err.return s
    | exception e -> read_failure path e
  in
  let* m = Coral_manifest.of_string text in
  let* p = of_manifest m name in
  let* () = verify ~dir p in
  Err.return p

(* A hole is the whole argument or its suffix ([--binary=<elf>]). *)
let fill elf arg =
  let hole = "<elf>" in
  let n = String.length arg and k = String.length hole in
  if n >= k && String.equal (String.sub arg (n - k) k) hole then
    String.sub arg 0 (n - k) ^ elf
  else arg

let argv ~dir ~elf p =
  match p.command with
  | [] -> [ Filename.concat dir p.artifact; elf ]
  | _ :: args -> Filename.concat dir p.artifact :: List.map (fill elf) args

let pp_region ppf (r : Region.t) =
  Fmt.pf ppf "%-14s %-4s 0x%08Lx %8Ld%s%s%s%s" r.name (Role.to_string r.role)
    r.base r.size
    (if r.writable then " writable" else "")
    (if r.loadable then " loadable" else "")
    (if r.htif_reachable then " htif" else "")
    (if r.dma_reachable then " dma" else "")

let pp ppf p =
  Fmt.pf ppf "@[<v>%s %s %s@,%s@,%a@]" p.name
    (Simulator.to_string p.simulator)
    p.configuration_id
    (String.concat "," p.extensions)
    Fmt.(list pp_region)
    p.regions
