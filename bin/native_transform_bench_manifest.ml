(* The environment manifest: enough to tell two JSONL runs
   apart, or to know they are directly comparable. Recorded once per
   invocation, never inferred from a sample.

   The workspace commonly carries unrelated in-flight changes (vendored
   submodule bumps, WIP on other files) — this records the dirty state as
   data rather than resetting it or attributing it to this benchmark. *)

module J = Jsont.Json

let command_output prog args =
  let out = Filename.temp_file "native_transform_bench" ".out" in
  Fun.protect
    ~finally:(fun () -> Sys.remove out)
    (fun () ->
      let cmd =
        Filename.quote_command prog args ~stdout:out ~stderr:"/dev/null"
      in
      if Sys.command cmd = 0 then
        Some (In_channel.with_open_bin out In_channel.input_all)
      else None)

let lines s = String.split_on_char '\n' s |> List.filter (( <> ) "")

let trim_line = function
  | None -> None
  | Some s -> ( match lines s with s :: _ -> Some s | [] -> None)

let contains ~needle haystack =
  let hn = String.length needle and hh = String.length haystack in
  let rec at i =
    i + hn <= hh && (String.sub haystack i hn = needle || at (i + 1))
  in
  hn = 0 || at 0

let git_revision dir =
  trim_line (command_output "git" [ "-C"; dir; "rev-parse"; "HEAD" ])

let git_dirty dir =
  match command_output "git" [ "-C"; dir; "status"; "--porcelain" ] with
  | None -> None
  | Some "" -> Some false
  | Some _ -> Some true

(* [needle] matches by substring against the whole submodule-status line
   rather than requiring the exact registered path, so a caller can pass any
   directory under the submodule (e.g. its "models" subdirectory) without
   having to know where the submodule itself is rooted. *)
let submodule_revision ~repo_root ~needle =
  match command_output "git" [ "-C"; repo_root; "submodule"; "status" ] with
  | None -> None
  | Some out -> (
      match List.find_opt (contains ~needle) (lines out) with
      | None -> None
      | Some line -> (
          (* "[+-U ]<sha1> <path> (<describe>)" — the sha1 alone identifies
             the checked-out commit; leading state marker is stripped. *)
          let line = String.trim line in
          let line =
            if String.length line > 0 && not (Char.equal line.[0] ' ') then
              String.sub line 1 (String.length line - 1)
            else line
          in
          match String.split_on_char ' ' (String.trim line) with
          | sha :: _ -> Some sha
          | [] -> None))

let uname flag = trim_line (command_output "uname" [ flag ])

type t = {
  repo_root : string;
  source_revision : string option;
  source_dirty : bool option;
  producer_revision : string option;
  ocaml_version : string;
  dune_profile : string;
  os : string option;
  cpu : string option;
  command : string list;
  verify_policy : string;
  payload_policy : string;
  warmup : int;
  repeats : int;
}

let create ~repo_root ~producer_path ~dune_profile ~verify_policy
    ~payload_policy ~warmup ~repeats =
  {
    repo_root;
    source_revision = git_revision repo_root;
    source_dirty = git_dirty repo_root;
    producer_revision = submodule_revision ~repo_root ~needle:producer_path;
    ocaml_version = Sys.ocaml_version;
    dune_profile;
    os = uname "-s";
    cpu = uname "-m";
    command = Array.to_list Sys.argv;
    verify_policy;
    payload_policy;
    warmup;
    repeats;
  }

let opt_str = function Some s -> J.string s | None -> J.null ()
let opt_bool = function Some b -> J.bool b | None -> J.null ()

let json t =
  J.object'
    [
      J.mem (J.name "repo_root") (J.string t.repo_root);
      J.mem (J.name "source_revision") (opt_str t.source_revision);
      J.mem (J.name "source_dirty") (opt_bool t.source_dirty);
      J.mem (J.name "producer_revision") (opt_str t.producer_revision);
      J.mem (J.name "ocaml_version") (J.string t.ocaml_version);
      J.mem (J.name "dune_profile") (J.string t.dune_profile);
      J.mem (J.name "os") (opt_str t.os);
      J.mem (J.name "cpu") (opt_str t.cpu);
      J.mem (J.name "command") (J.list (List.map J.string t.command));
      J.mem (J.name "verify_policy") (J.string t.verify_policy);
      J.mem (J.name "payload_policy") (J.string t.payload_policy);
      J.mem (J.name "warmup") (J.number (float_of_int t.warmup));
      J.mem (J.name "repeats") (J.number (float_of_int t.repeats));
      J.mem (J.name "clock")
        (J.string
           "Unix.gettimeofday (wall clock; no monotonic clock library is \
            vendored in this repo)");
    ]

let to_string t =
  match
    Jsont_bytesrw.encode_string ~format:Jsont.Indent Jsont.json (json t)
  with
  | Ok s -> s
  | Error msg -> failwith msg
