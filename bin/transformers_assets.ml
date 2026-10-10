open Err.Syntax
module T = Transformers_metadata
module U = Pt2_fixture_unix

let run mode request_path source cache_dir cohort_path output_dir =
  let* request_json = T.Json_util.read request_path in
  let* requests = T.Assets.requests request_json in
  let* producer =
    T.Producer.load ~consumer_root:(Unix.getcwd ()) ~root:source
  in
  let* cache = U.Cache.create cache_dir in
  let* cohort = T.Json_util.read cohort_path in
  let transport = if mode = "fetch" then Some U.Transport.curl else None in
  let* generated =
    T.Assets.generate
      (U.Bundle.config ?transport cache)
      producer cohort requests
  in
  let* () =
    if mode = "check" then
      Err.List.iter
        (fun (name, json) ->
          let path = Filename.concat output_dir name in
          let* existing = T.Json_util.read path in
          T.Json_util.equal ~identity:path ~field:"derived adapter manifest"
            existing json)
        generated
    else begin
      U.Cache.mkdir_p output_dir;
      Err.List.iter
        (fun (name, j) -> T.Json_util.write (Filename.concat output_dir name) j)
        generated
    end
  in
  Fmt.pr
    "verified %d adapters against processor/config bytes and release \
     contracts@."
    (List.length generated);
  Ok ()

let () =
  let result =
    try
      match Array.to_list Sys.argv with
      | [
       _;
       (("generate" | "check" | "fetch") as mode);
       request;
       source;
       cache;
       cohort;
       output;
      ] ->
          run mode request source cache cohort output
      | _ ->
          T.Json_util.invalid
            "usage: transformers_assets (generate|check|fetch) \
             ADAPTER_SELECTION SOURCE CACHE COHORT OUTPUT_DIR"
    with
    | Sys_error e -> Err.fail (`File_io ("metadata", e))
    | Unix.Unix_error (e, op, path) ->
        Err.fail (`File_io (path, op ^ ": " ^ Unix.error_message e))
  in
  match result with
  | Ok () -> ()
  | Error e ->
      Fmt.epr "transformers_assets: %a@." T.Fault.pp_error (Err.Error.kind e);
      exit 2
