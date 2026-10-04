open Pt2_safetensors_test.Fixtures

let temp_dir () =
  let d = Filename.temp_file "pt2-st-" "" in
  Sys.remove d;
  Unix.mkdir d 0o755;
  d

let rec rm_rf p =
  match Unix.lstat p with
  | exception Unix.Unix_error _ -> ()
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (fun f -> rm_rf (Filename.concat p f)) (Sys.readdir p);
      Unix.rmdir p
  | _ -> Unix.unlink p

let write path contents =
  let rec mk d =
    if not (Sys.file_exists d) then (
      mk (Filename.dirname d);
      Unix.mkdir d 0o755)
  in
  mk (Filename.dirname path);
  Out_channel.with_open_bin path (fun oc -> output_string oc contents)

let real_sha256 =
  let tmp = Filename.temp_file "pt2-st-" ".bin" in
  write tmp checkpoint;
  let h = Result.get_ok (Hf_hub_unix.sha256sum tmp) in
  Sys.remove tmp;
  h

(* A fake Hub serving [checkpoint]; [requests] counts what reached it. *)
let run ?(map = map_json ~sha256:real_sha256 ()) () =
  let model = temp_dir () and cache = temp_dir () in
  Fun.protect
    ~finally:(fun () ->
      rm_rf model;
      rm_rf cache)
    (fun () ->
      write (Filename.concat model "models/model.json") program_json;
      write (Filename.concat model "models/safetensors.json") map;
      write
        (Filename.concat model "data/weights/model_weights_config.json")
        (weights ());
      let env = Hf_hub.Env.make ~cache_dir:cache ~endpoint:"https://hub" () in
      let requests = ref 0 in
      let http : Hf_hub_unix.http =
       fun req ->
        incr requests;
        match req with
        | Hf_hub.Request.Head _ ->
            Hf_hub.Response.Received
              {
                status = 200;
                headers =
                  [
                    ("X-Repo-Commit", String.make 40 'a');
                    ("X-Linked-Etag", "\"" ^ real_sha256 ^ "\"");
                    ("X-Linked-Size", string_of_int (String.length checkpoint));
                  ];
              }
        | Get { sink; _ } ->
            write
              (Hf_hub.Cache_layout.incomplete_path ~root:cache sink.repo
                 ~etag:sink.etag)
              checkpoint;
            Hf_hub.Response.Received { status = 200; headers = [] }
      in
      let result = Pt2_safetensors_unix.open_dir ~env ~http model in
      (match result with
      | Ok archive -> (
          match Pt2_archive.load_captured_tensor archive "w" with
          | Ok t -> Format.printf "ok %a@." Pt2_tensor.pp t
          | Error e ->
              Format.printf "load failed: %a@." Pt2_archive.pp_error
                (Err.Error.kind e))
      | Error e ->
          Format.printf "refused: %a@." Pt2_safetensors_unix.pp_error
            (Err.Error.kind e));
      Printf.printf "requests=%d\n" !requests)

let%expect_test "open_dir resolves the pinned checkpoint and maps it" =
  run ();
  [%expect {|
    ok float32[2; 3]
    requests=2 |}]

let%expect_test "the pin is checked: sha256 and size" =
  run ~map:(map_json ~sha256:(String.make 64 'c') ()) ();
  run
    ~map:(map_json ~sha256:real_sha256 ~size:(String.length checkpoint + 1) ())
    ();
  [%expect
    {|
    refused: checkpoint sha256 is 700178a2dbaf8e9e501e8612043e1cfcec66cc003908eefcc8df8cd77b167716, safetensors.json pins cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    requests=2
    refused: checkpoint is 239 bytes, safetensors.json pins 240
    requests=2 |}]

let%expect_test "a map that cannot run is refused before any request" =
  run ~map:(map_json ~sha256:real_sha256 ~unmapped:[ "b" ] ()) ();
  [%expect
    {|
    refused: the checkpoint lacks 1 captured tensor(s): b
    requests=0 |}]
