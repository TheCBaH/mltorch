(* Hub-hosted checkpoint sources: a fake Hub answering the requests hf-hub
   emits, a real temporary huggingface_hub-layout cache. *)
open Support
module Document = Pt2_checkpoint_map.Document

let commit = "0123456789abcdef0123456789abcdef01234567"
let other_commit = "fedcba9876543210fedcba9876543210fedcba98"
let body = "weights-bytes"
let digest data = Pt2_sha256.Digest.to_hex (Pt2_sha256.string data)

let source ?(url = None) ?(provenance = None) ?(data = body) () =
  let url =
    match url with
    | Some u -> u
    | None -> "https://huggingface.co/o/m/resolve/" ^ commit ^ "/w.safetensors"
  in
  let pin =
    match
      Err.payload
        (Document.pin_of_fields ~name:"w.safetensors" ~sha256:(digest data)
           ~size:(Int64.of_int (String.length data))
           ~url)
    with
    | Ok p -> p
    | Error _ -> failwith "pin"
  in
  let provenance =
    match provenance with
    | Some p -> p
    | None -> Document.Source.Upstream { repo_id = "o/m"; revision = commit }
  in
  { Document.Source.pin; provenance }

(* HEAD answers the metadata; the GET body is written by [with_hub]. *)
let http ~content calls : Hf_hub_unix.http = function
  | Hf_hub.Request.Head { url; _ } ->
      calls := ("HEAD " ^ url) :: !calls;
      Hf_hub.Response.Received
        {
          status = 200;
          headers =
            [
              ("X-Repo-Commit", commit);
              ("X-Linked-Etag", "\"" ^ digest content ^ "\"");
              ("X-Linked-Size", string_of_int (String.length content));
            ];
        }
  | Get { url; _ } ->
      calls := ("GET " ^ url) :: !calls;
      Hf_hub.Response.Received { status = 200; headers = [] }

let with_hub ?(offline = false) ~content f =
  with_dir (fun dir ->
      let env = Hf_hub.Env.make ~cache_dir:dir ~offline () in
      let calls = ref [] in
      let http env_ request =
        match request with
        | Hf_hub.Request.Get { sink; _ } as r ->
            let path =
              Hf_hub.Cache_layout.incomplete_path
                ~root:env_.Hf_hub.Env.cache_dir sink.repo ~etag:sink.etag
            in
            write path content;
            http ~content calls r
        | r -> http ~content calls r
      in
      let hub =
        U.Hub.make
          ~http:(fun r -> http env r)
          ~hasher:(fun p -> Ok (digest (read p)))
          env
      in
      f hub calls)

let show hub ~online s =
  match Err.payload (U.Hub.ensure ~online hub s) with
  | Ok None -> print_endline "not a hub file"
  | Ok (Some path) -> Fmt.pr "ok %s@." (read path)
  | Error e -> show_error e

let%expect_test "download once, then reuse from the cache with no request" =
  with_hub ~content:body (fun hub calls ->
      show hub ~online:true (source ());
      List.iter print_endline (List.rev !calls);
      calls := [];
      show hub ~online:true (source ());
      Fmt.pr "requests on reuse: %d@." (List.length !calls);
      show hub ~online:false (source ()));
  [%expect
    {|
    ok weights-bytes
    HEAD https://huggingface.co/o/m/resolve/0123456789abcdef0123456789abcdef01234567/w.safetensors
    GET https://huggingface.co/o/m/resolve/0123456789abcdef0123456789abcdef01234567/w.safetensors
    ok weights-bytes
    requests on reuse: 0
    ok weights-bytes |}]

let%expect_test "offline with an empty cache is refused, nothing requested" =
  with_hub ~content:body (fun hub calls ->
      show hub ~online:false (source ());
      Fmt.pr "requests: %d@." (List.length !calls));
  [%expect
    {|
    "w.safetensors" is not in the cache and no transport is configured
    requests: 0 |}]

let%expect_test "the pin decides, not the Hub" =
  (* The Hub serves other bytes than the pin names: its own sha256 check passes
     (the etag it sent is of what it sent) but the pin does not. *)
  with_hub ~content:"different-bytes" (fun hub _ ->
      show hub ~online:true (source ()));
  with_hub ~content:"weights-BYTES" (fun hub _ ->
      show hub ~online:true (source ()));
  [%expect
    {|
    source "w.safetensors": size is 15 bytes, expected 13
    source "w.safetensors": digest is 3c8836fced96074e334d3736b1acbfb24374ce1b7068389417b2f439afb0615d, expected 47c46a5fa409889c23e4d12bdc28a077a7ccc8a92c8dd2bfbe3d7d9c6c227e67 |}]

let%expect_test "which sources use the Hub" =
  with_hub ~content:body (fun hub calls ->
      let converted =
        Document.Source.Converted
          {
            Document.Derived.file = "pytorch_model.bin";
            repo_id = "o/m";
            revision = commit;
            sha256 = Pt2_sha256.string "x";
            tool = "t";
          }
      in
      show hub ~online:true (source ~provenance:(Some converted) ());
      show hub ~online:true
        (source ~url:(Some "https://example.org/w.safetensors") ());
      Fmt.pr "requests: %d@." (List.length !calls);
      show hub ~online:true
        (source
           ~url:
             (Some
                ("https://huggingface.co/o/m/resolve/" ^ other_commit
               ^ "/w.safetensors"))
           ());
      Fmt.pr "requests: %d@." (List.length !calls));
  [%expect
    {|
    not a hub file
    not a hub file
    requests: 0
    "w.safetensors": the pinned URL names the Hub but not this repository, revision and file
    requests: 0 |}]
