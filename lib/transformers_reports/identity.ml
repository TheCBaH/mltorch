open Err.Syntax
open Transformers_metadata.Json_util
module J = Jsont.Json
module Source = Transformers_metadata.Source
module Producer = Transformers_metadata.Producer

let digest json =
  let+ bytes = text json in
  Source.hash bytes

let strings xs = J.list (List.map J.string xs)
let split_zero s = List.filter (( <> ) "") (String.split_on_char '\000' s)

(* Include the executable sources, build rules and selected input metadata.
   Documentation and generated evidence do not change executable identity. *)
let relevant path =
  List.exists
    (fun suffix -> Filename.check_suffix path suffix)
    [
      ".ml";
      ".mli";
      ".c";
      ".h";
      ".cpp";
      ".cc";
      ".sh";
      ".py";
      ".opam";
      ".yaml";
      ".yml";
      ".json";
      ".js";
      ".ts";
    ]
  || List.mem (Filename.basename path)
       [ "dune"; "dune-project"; "dune-workspace"; "Makefile"; ".gitmodules" ]

let file root path =
  let absolute = Filename.concat root path in
  let stat =
    try Some (Unix.lstat absolute)
    with Unix.Unix_error (Unix.ENOENT, _, _) -> None
  in
  match stat with
  | None -> Ok (path, J.string "missing")
  | Some stat -> (
      match stat.Unix.st_kind with
      | Unix.S_REG ->
          let* hash =
            Err.import ~pos:__POS__
              (fun s -> `Metadata_invalid s)
              (Pt2_fixture_unix.Cache.default_hasher absolute)
          in
          Ok
            ( path,
              obj
                [
                  ("sha256", J.string (Pt2_sha256.Digest.to_hex hash));
                  ("executable", J.bool (stat.st_perm land 0o111 <> 0));
                ] )
      | Unix.S_LNK ->
          let* content =
            if not (Sys.file_exists absolute) then Ok (J.null ())
            else
              let* hash =
                Err.import ~pos:__POS__
                  (fun s -> `Metadata_invalid s)
                  (Pt2_fixture_unix.Cache.default_hasher absolute)
              in
              Ok (J.string (Pt2_sha256.Digest.to_hex hash))
          in
          Ok
            ( path,
              obj
                [
                  ("symlink", J.string (Unix.readlink absolute));
                  ("target_sha256", content);
                ] )
      | _ -> invalid ("unexpected source file type: " ^ path))

let snapshot root paths =
  let* names =
    Producer.command root
      ([ "ls-files"; "-z"; "--cached"; "--others"; "--exclude-standard"; "--" ]
      @ paths)
  in
  let names = List.sort_uniq String.compare (split_zero names) in
  let* files = Err.List.map (file root) (List.filter relevant names) in
  let* diff =
    Producer.command root ([ "diff"; "--binary"; "HEAD"; "--" ] @ paths)
  in
  Ok
    (obj [ ("files", obj files); ("diff_sha256", J.string (Source.hash diff)) ])

let workspace root =
  let paths =
    [
      "bin";
      "lib";
      "js";
      "scripts";
      "data/transformers";
      "data/dune";
      "mltorch.opam";
      "dune-project";
      "dune-workspace";
      "Makefile";
      ".gitmodules";
    ]
  in
  let* commit = Producer.command root [ "rev-parse"; "HEAD" ] in
  let* source = snapshot root paths in
  let* links = Producer.command root [ "ls-files"; "--stage"; "-z" ] in
  let links =
    List.filter_map
      (fun row ->
        match String.split_on_char '\t' row with
        | [ header; path ] when String.starts_with ~prefix:"160000 " header ->
            Some (path, header)
        | _ -> None)
      (split_zero links)
  in
  let* links =
    Err.List.map
      (fun (path, pin) ->
        let dir = Filename.concat root path in
        if not (Sys.file_exists (Filename.concat dir ".git")) then
          Ok
            ( path,
              obj [ ("index", J.string pin); ("initialized", J.bool false) ] )
        else
          let* head = Producer.command dir [ "rev-parse"; "HEAD" ] in
          let* source =
            if String.starts_with ~prefix:"vendored/" path then
              snapshot dir [ "." ]
            else if path = "modules/pytorch" then
              snapshot dir
                [
                  "torch/_export/serde/schema.yaml";
                  "aten/src/ATen/native/native_functions.yaml";
                ]
            else Ok (J.null ())
          in
          let* diff = Producer.command dir [ "diff"; "--binary"; "HEAD" ] in
          Ok
            ( path,
              obj
                [
                  ("index", J.string pin);
                  ("head", J.string (String.trim head));
                  ("diff_sha256", J.string (Source.hash diff));
                  ("source", source);
                ] ))
      links
  in
  Ok
    (obj
       [
         ("commit", J.string (String.trim commit));
         ("source", source);
         ("submodules", obj links);
       ])

let process exe args =
  try
    let ic = Unix.open_process_args_in exe (Array.of_list (exe :: args)) in
    let out = In_channel.input_all ic in
    match Unix.close_process_in ic with
    | Unix.WEXITED 0 -> J.string (String.trim out)
    | _ -> J.string "unavailable"
  with Unix.Unix_error _ -> J.string "unavailable"

let environment () =
  let variables =
    [
      "OMP_NUM_THREADS";
      "MKL_NUM_THREADS";
      "OPENBLAS_NUM_THREADS";
      "VECLIB_MAXIMUM_THREADS";
      "OMP_DYNAMIC";
      "OMP_PROC_BIND";
      "OCAMLRUNPARAM";
    ]
  in
  let cpu =
    if Sys.file_exists "/proc/cpuinfo" then
      let bytes =
        In_channel.with_open_bin "/proc/cpuinfo" In_channel.input_all
      in
      let stable =
        String.split_on_char '\n' bytes
        |> List.filter (fun line ->
            List.exists
              (fun key -> String.starts_with ~prefix:key line)
              [
                "model name";
                "vendor_id";
                "flags";
                "Features";
                "CPU part";
                "CPU implementer";
                "Hardware";
              ])
        |> List.sort_uniq String.compare
      in
      J.string (Source.hash (String.concat "\n" stable))
    else J.null ()
  in
  obj
    [
      ("architecture", process "uname" [ "-m" ]);
      ("kernel", process "uname" [ "-sr" ]);
      ("cpuinfo_sha256", cpu);
      ("ocaml", J.string Sys.ocaml_version);
      ("c_compiler", process "cc" [ "--version" ]);
      ( "threading",
        obj
          (List.map
             (fun k ->
               ( k,
                 match Sys.getenv_opt k with
                 | None -> J.null ()
                 | Some s -> J.string s ))
             variables) );
      ("native_threads", J.int 1);
    ]

let capture ~consumer_root ~source ~cohort =
  let* workspace = workspace consumer_root in
  let* producer = Producer.load ~consumer_root ~root:source in
  let* cohort_bytes = Pt2_fixture_unix.Fetch.read cohort in
  let* cohort_json = parse cohort_bytes in
  let* publication = member "publication" cohort_json in
  let* release_tag = member "release_tag" cohort_json in
  let* release_producer_commit = member "release_producer_commit" cohort_json in
  let* repository = member "repository" cohort_json in
  let release =
    obj
      [
        ("tag", release_tag);
        ("producer_commit", release_producer_commit);
        ("repository", repository);
      ]
  in
  let* executable =
    file consumer_root "_build/default/bin/transformers_replay.exe"
  in
  Ok
    (obj
       [
         ("consumer", workspace);
         ("replay_executable", snd executable);
         ( "source",
           obj
             [
               ("gitlink", J.string producer.commit);
               ("inventory", producer.metadata);
             ] );
         ("cohort_sha256", J.string (Source.hash cohort_bytes));
         ("publication", publication);
         ("release", release);
         ("environment", environment ());
       ])
