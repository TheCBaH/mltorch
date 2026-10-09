(* Usage: transformers_text_demo.exe COHORT.json CACHE_DIR TEXT_ASSETS.json
          VOCAB.txt (SENTENCE... | --lines FILE)

   A bounded raw-input example for the BERT-tiny text encoder: each ASCII
   sentence is tokenized with the pinned vocabulary (the file's digest is
   checked against the pin first), padded to the artifact's 16 positions, run
   through the verified graph, and the pooled embeddings are compared by cosine
   similarity against the first sentence.

   What is and is not established. The model component passes its published
   cases (make transformers.gate). The tokenizer follows BERT's documented
   algorithm on ASCII text and is checked against an independent
   implementation (scripts/transformers-wordpiece-check.py), not against the
   reference tokenizer, which is not available here; the published cases hold
   random token ids, so they cannot vouch for it. The end-to-end embedding has no
   producer reference. This is a demonstration, not a task-ready claim.

   With [--ids] the sentences are only tokenized and their ids printed, one
   line each, for that comparison. *)

open Err.Syntax
module Fixture = Pt2_fixture_unix.Fixture

let read_file path = In_channel.with_open_bin path In_channel.input_all

let pin_string json key =
  match Jsont_bytesrw.decode_string Jsont.json json with
  | Error _ -> None
  | Ok j -> (
      match j with
      | Jsont.Object (members, _) ->
          List.find_map
            (fun ((k, _), v) ->
              if String.equal k key then
                match v with Jsont.String (s, _) -> Some s | _ -> None
              else None)
            members
      | _ -> None)

let nested_pin json outer key =
  match Jsont_bytesrw.decode_string Jsont.json json with
  | Ok (Jsont.Object (members, _)) ->
      List.find_map
        (fun ((k, _), v) ->
          if String.equal k outer then
            match v with
            | Jsont.Object (inner, _) ->
                List.find_map
                  (fun ((k2, _), v2) ->
                    if String.equal k2 key then
                      match v2 with Jsont.String (s, _) -> Some s | _ -> None
                    else None)
                  inner
            | _ -> None
          else None)
        members
  | _ -> None

let pt2_i64 shape values =
  let b = Bytes.create (8 * List.length values) in
  List.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (Int64.of_int v)) values;
  {
    Pt2_tensor.dtype = Pt2_dtype.Int64;
    sizes = shape;
    strides = [ List.nth shape 1; 1 ];
    storage_offset = 0;
    data = Pt2_storage.of_string (Bytes.to_string b);
  }

let floats packed ~count =
  List.init count (fun c ->
      Tensor.read packed (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c))

let cosine a b =
  let dot = List.fold_left2 (fun s x y -> s +. (x *. y)) 0. a b in
  let norm l = sqrt (List.fold_left (fun s x -> s +. (x *. x)) 0. l) in
  dot /. (norm a *. norm b)

let () =
  let args = List.tl (Array.to_list Sys.argv) in
  let ids_only, args =
    match args with "--ids" :: rest -> (true, rest) | _ -> (false, args)
  in
  match args with
  | cohort_path :: cache_dir :: assets_path :: vocab_path :: sentences
    when sentences <> [] -> (
      (* [--lines FILE] reads one sentence per line instead of the arguments. *)
      let sentences =
        match sentences with
        | [ "--lines"; path ] -> (
            match List.rev (String.split_on_char '\n' (read_file path)) with
            | "" :: rest -> List.rev rest
            | all -> List.rev all)
        | s -> s
      in
      let assets = read_file assets_path in
      let vocab_text = read_file vocab_path in
      let pinned = nested_pin assets "vocab" "sha256" in
      let actual = Pt2_sha256.Digest.to_hex (Pt2_sha256.string vocab_text) in
      if pinned <> Some actual then (
        Fmt.epr "vocab.txt digest %s does not match the pin %s@." actual
          (Option.value pinned ~default:"(none)");
        exit 2);
      let lines =
        match List.rev (String.split_on_char '\n' vocab_text) with
        | "" :: rest -> List.rev rest
        | all -> List.rev all
      in
      let vocab = Wordpiece.vocab_of_lines lines in
      let max_length = 16 in
      let encoded =
        List.map
          (fun s ->
            match Wordpiece.encode vocab ~max_length s with
            | Ok e -> (s, e)
            | Error e ->
                Fmt.epr "%S: %a@." s Wordpiece.pp_error e;
                exit 2)
          sentences
      in
      if ids_only then
        List.iter
          (fun (_, (e : Wordpiece.encoded)) ->
            print_endline
              (String.concat " " (List.map string_of_int e.input_ids)))
          encoded
      else
        let artifact = Option.get (pin_string assets "artifact_id") in
        let setup =
          let* text = Pt2_fixture_unix.Fetch.read cohort_path in
          let* cohort = Pt2_fixture.Cohort.of_string text in
          let* cache = Pt2_fixture_unix.Cache.create cache_dir in
          let entry =
            List.find
              (fun (e : Pt2_fixture.Cohort.entry) ->
                String.equal e.artifact_id artifact)
              cohort.entries
          in
          Fixture.open_ (Pt2_fixture_unix.Bundle.config cache) cohort entry
        in
        match Err.payload setup with
        | Error e ->
            Fmt.epr "%a@." Fixture.pp_error e;
            exit 2
        | Ok fixture ->
            let embed (_, (e : Wordpiece.encoded)) =
              let inputs =
                [
                  ("input_ids", pt2_i64 [ 1; max_length ] e.input_ids);
                  ("attention_mask", pt2_i64 [ 1; max_length ] e.attention_mask);
                  ("token_type_ids", pt2_i64 [ 1; max_length ] e.token_type_ids);
                ]
              in
              match
                Err.payload (Native_interp.run_named fixture.archive ~inputs)
              with
              | Ok [ _hidden; pooled ] -> floats pooled ~count:128
              | Ok _ -> failwith "unexpected output count"
              | Error e ->
                  Fmt.epr "%a@." Native_interp.pp_error e;
                  exit 1
            in
            let embeddings = List.map embed encoded in
            List.iter2
              (fun (s, (e : Wordpiece.encoded)) emb ->
                Fmt.pr "%S@.  tokens: %s@.  pooled[0..3]: %s@." s
                  (String.concat " "
                     (List.filteri
                        (fun i _ -> List.nth e.attention_mask i = 1)
                        e.tokens))
                  (String.concat " "
                     (List.filteri (fun i _ -> i < 4) emb
                     |> List.map (Printf.sprintf "%.6f"))))
              encoded embeddings;
            let first = List.hd embeddings in
            List.iteri
              (fun i emb ->
                if i > 0 then
                  Fmt.pr "cosine(sentence 0, sentence %d) = %.6f@." i
                    (cosine first emb))
              embeddings)
  | _ ->
      prerr_endline
        "usage: transformers_text_demo [--ids] COHORT.json CACHE_DIR \
         TEXT_ASSETS.json VOCAB.txt (SENTENCE... | --lines FILE)";
      exit 2
