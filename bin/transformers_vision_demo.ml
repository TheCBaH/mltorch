(* Usage:
     transformers_vision_demo.exe pixels IMAGE.ppm
     transformers_vision_demo.exe classify COHORT.json CACHE_DIR VISION_ASSETS.json
       CONFIG.json IMAGE.ppm

   A bounded image-classification example for MobileViT-xx-small: a binary PPM
   image is resized (shorter edge to 288, Pillow's bilinear arithmetic),
   centre-cropped to 256, scaled to [0, 1] and channel-flipped to BGR, as the
   pinned preprocessor config says, then run through the verified graph; the top
   five of the 1000 ImageNet classes are printed with the labels of the pinned
   config.json (whose digest is checked first).

   Established in the design record: the preprocessed tensor against the
   reference image processor, and the logits against transformers'
   MobileViTForImageClassification on the pinned weights
   (scripts/transformers-vision-crosscheck.py). Scope: a PPM image, native-direct
   with the default numerics. [pixels] only prepares the tensor. *)

open Err.Syntax
module Fixture = Pt2_fixture_unix.Fixture

let read_file path = In_channel.with_open_bin path In_channel.input_all

let json_member k = function
  | Jsont.Object (ms, _) ->
      List.find_map (fun ((n, _), v) -> if n = k then Some v else None) ms
  | _ -> None

let parse path =
  match Jsont_bytesrw.decode_string Jsont.json (read_file path) with
  | Ok j -> j
  | Error e -> failwith e

let preprocess ppm =
  match Clip_input.Ppm.of_string (read_file ppm) with
  | Error e ->
      Fmt.epr "%s: %s@." ppm e;
      exit 2
  | Ok img ->
      Clip_input.Prep.tensor img ~filter:Clip_input.Resample.Bilinear
        ~resize:288 ~crop:256 ~flip:true ~mean:[| 0.; 0.; 0. |]
        ~std:[| 1.; 1.; 1. |]

let () =
  match List.tl (Array.to_list Sys.argv) with
  | [ "pixels"; ppm ] ->
      Array.iter (fun v -> Printf.printf "%.9g\n" v) (preprocess ppm)
  | [ "classify"; cohort_path; cache_dir; assets_path; config_path; ppm ] -> (
      let assets = parse assets_path in
      let pin =
        match
          json_member "sha256" (Option.get (json_member "config" assets))
        with
        | Some (Jsont.String (s, _)) -> s
        | _ -> failwith "pin"
      in
      let config_text = read_file config_path in
      let actual = Pt2_sha256.Digest.to_hex (Pt2_sha256.string config_text) in
      if not (String.equal pin actual) then (
        Fmt.epr "config.json digest %s does not match the pin %s@." actual pin;
        exit 2);
      let labels =
        match Jsont_bytesrw.decode_string Jsont.json config_text with
        | Ok j -> (
            match json_member "id2label" j with
            | Some (Jsont.Object (ms, _)) ->
                List.filter_map
                  (fun ((k, _), v) ->
                    match (int_of_string_opt k, v) with
                    | Some i, Jsont.String (s, _) -> Some (i, s)
                    | _ -> None)
                  ms
            | _ -> [])
        | Error _ -> []
      in
      let pixels = preprocess ppm in
      let artifact =
        match json_member "artifact_id" assets with
        | Some (Jsont.String (s, _)) -> s
        | _ -> failwith "artifact_id"
      in
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
      | Ok fixture -> (
          let b = Bytes.create (4 * Array.length pixels) in
          Array.iteri
            (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float v))
            pixels;
          let input =
            {
              Pt2_tensor.dtype = Pt2_dtype.Float32;
              sizes = [ 1; 3; 256; 256 ];
              strides = [ 3 * 256 * 256; 256 * 256; 256; 1 ];
              storage_offset = 0;
              data = Pt2_storage.of_string (Bytes.to_string b);
            }
          in
          match
            Err.payload
              (Native_interp.run_named fixture.archive
                 ~inputs:[ ("pixel_values", input) ])
          with
          | Ok [ logits ] ->
              let scores =
                List.init 1000 (fun c ->
                    ( c,
                      Tensor.read logits
                        (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c) ))
              in
              let top =
                List.sort (fun (_, a) (_, b) -> compare b a) scores
                |> List.filteri (fun i _ -> i < 5)
              in
              List.iter
                (fun (c, v) ->
                  Fmt.pr "%4d %-34s %.6f@." c
                    (Option.value ~default:"?" (List.assoc_opt c labels))
                    v)
                top;
              if Sys.getenv_opt "VISION_DUMP_LOGITS" <> None then
                Fmt.pr "logits: %s@."
                  (String.concat " "
                     (List.map (fun (_, v) -> Printf.sprintf "%.9g" v) scores))
          | Ok _ -> failwith "unexpected output count"
          | Error e ->
              Fmt.epr "%a@." Native_interp.pp_error e;
              exit 1))
  | _ ->
      prerr_endline
        "usage: transformers_vision_demo (pixels IMAGE.ppm | classify COHORT \
         CACHE ASSETS CONFIG.json IMAGE.ppm)";
      exit 2
