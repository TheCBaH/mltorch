(* Usage:
     transformers_clip_demo.exe ids TOKENIZER.json (SENTENCE... | --lines FILE)
     transformers_clip_demo.exe pixels IMAGE.ppm
     transformers_clip_demo.exe score COHORT.json CACHE_DIR CLIP_ASSETS.json
       TOKENIZER.json IMAGE.ppm SENTENCE...

   A bounded image-text scoring example for TinyCLIP: ASCII sentences and a binary
   PPM image become the forward artifact's inputs (the tokenizer.json's byte-pair
   encoding padded to 16, the image bicubic-resized, centre-cropped and
   normalized exactly as the pinned preprocessor config says), and the graph's
   logits_per_image is printed for each sentence. The tokenizer file's digest is
   checked against the pin first.

   What is established is in the design record: the ids and the pixel tensor are
   compared with the reference tokenizer and image processor
   (scripts/transformers-clip-crosscheck.py), and the scores with transformers'
   CLIPModel on the pinned weights. Scope: ASCII text, at most 14 tokens, a PPM
   image, native-direct with the default numerics. [ids] and [pixels] only
   prepare inputs, for that comparison. *)

open Err.Syntax
module Fixture = Pt2_fixture_unix.Fixture

let read_file path = In_channel.with_open_bin path In_channel.input_all

let lines_of text =
  match List.rev (String.split_on_char '\n' text) with
  | "" :: rest -> List.rev rest
  | all -> List.rev all

let sentences = function
  | [ "--lines"; path ] -> lines_of (read_file path)
  | s -> s

let json_member k = function
  | Jsont.Object (ms, _) ->
      List.find_map (fun ((n, _), v) -> if n = k then Some v else None) ms
  | _ -> None

let pins path =
  match Jsont_bytesrw.decode_string Jsont.json (read_file path) with
  | Ok j -> j
  | Error e -> failwith e

let floats_of = function
  | Some (Jsont.Array (l, _)) ->
      Array.of_list
        (List.map
           (function Jsont.Number (f, _) -> f | _ -> failwith "number")
           l)
  | _ -> failwith "array"

let string_of = function
  | Some (Jsont.String (s, _)) -> s
  | _ -> failwith "string"

let int_of = function
  | Some (Jsont.Number (f, _)) -> int_of_float f
  | _ -> failwith "number"

let load_tokenizer assets_json tokenizer_path =
  let text = read_file tokenizer_path in
  let pin =
    string_of
      (json_member "sha256" (Option.get (json_member "tokenizer" assets_json)))
  in
  let actual = Pt2_sha256.Digest.to_hex (Pt2_sha256.string text) in
  if not (String.equal pin actual) then (
    Fmt.epr "tokenizer.json digest %s does not match the pin %s@." actual pin;
    exit 2);
  match Clip_input.Bpe.of_json text with
  | Ok t -> t
  | Error e ->
      Fmt.epr "tokenizer.json: %s@." e;
      exit 2

let encode tok ~max_length s =
  match Clip_input.Bpe.encode tok ~max_length s with
  | Ok e -> e
  | Error e ->
      Fmt.epr "%S: %s@." s e;
      exit 2

let image_params assets =
  let image = Option.get (json_member "image" assets) in
  ( int_of (json_member "shortest_edge" image),
    floats_of (json_member "image_mean" image),
    floats_of (json_member "image_std" image) )

let load_ppm path =
  match Clip_input.Ppm.of_string (read_file path) with
  | Ok i -> i
  | Error e ->
      Fmt.epr "%s: %s@." path e;
      exit 2

let pt2 dtype ~sizes bytes =
  let strides =
    let rec go = function
      | [] -> []
      | _ :: rest as l -> List.fold_left ( * ) 1 rest :: go (List.tl l)
    in
    go sizes
  in
  {
    Pt2_tensor.dtype;
    sizes;
    strides;
    storage_offset = 0;
    data = Pt2_storage.of_string bytes;
  }

let i64_tensor shape values =
  let b = Bytes.create (8 * List.length values) in
  List.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (Int64.of_int v)) values;
  pt2 Pt2_dtype.Int64 ~sizes:shape (Bytes.to_string b)

let f32_tensor shape (values : float array) =
  let b = Bytes.create (4 * Array.length values) in
  Array.iteri
    (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float v))
    values;
  pt2 Pt2_dtype.Float32 ~sizes:shape (Bytes.to_string b)

let () =
  match List.tl (Array.to_list Sys.argv) with
  | "ids" :: tokenizer_path :: rest ->
      let tok =
        match Clip_input.Bpe.of_json (read_file tokenizer_path) with
        | Ok t -> t
        | Error e ->
            prerr_endline e;
            exit 2
      in
      List.iter
        (fun s ->
          let e = encode tok ~max_length:16 s in
          print_endline (String.concat " " (List.map string_of_int e.input_ids)))
        (sentences rest)
  | [ "pixels"; ppm ] ->
      let pixels =
        Clip_input.Prep.pixel_values (load_ppm ppm) ~size:224
          ~mean:[| 0.48145466; 0.4578275; 0.40821073 |]
          ~std:[| 0.26862954; 0.26130258; 0.27577711 |]
      in
      Array.iter (fun v -> Printf.printf "%.9g\n" v) pixels
  | "score" :: cohort_path :: cache_dir :: assets_path :: tokenizer_path :: ppm
    :: rest
    when rest <> [] -> (
      let assets = pins assets_path in
      let tok = load_tokenizer assets tokenizer_path in
      let size, mean, std = image_params assets in
      let pixels =
        Clip_input.Prep.pixel_values (load_ppm ppm) ~size ~mean ~std
      in
      let artifact = string_of (json_member "artifact_id" assets) in
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
          List.iter
            (fun s ->
              let e = encode tok ~max_length:16 s in
              let inputs =
                [
                  ("input_ids", i64_tensor [ 1; 16 ] e.input_ids);
                  ("attention_mask", i64_tensor [ 1; 16 ] e.attention_mask);
                  ("pixel_values", f32_tensor [ 1; 3; 224; 224 ] pixels);
                ]
              in
              match
                Err.payload (Native_interp.run_named fixture.archive ~inputs)
              with
              | Ok (_text :: _image :: logits :: _) ->
                  let v =
                    Tensor.read logits
                      (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0)
                  in
                  Fmt.pr "%-40S logits_per_image = %.6f@." s v
              | Ok _ -> failwith "unexpected output count"
              | Error e ->
                  Fmt.epr "%a@." Native_interp.pp_error e;
                  exit 1)
            (sentences rest))
  | _ ->
      prerr_endline
        "usage: transformers_clip_demo (ids TOKENIZER.json S... | pixels \
         IMAGE.ppm | score COHORT CACHE ASSETS TOKENIZER IMAGE.ppm S...)";
      exit 2
