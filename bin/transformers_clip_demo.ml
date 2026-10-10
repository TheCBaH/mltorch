(* Bounded TinyCLIP ASCII/PPM scoring. The score route verifies the tokenizer,
   config and processor bytes and recipe against the selected release contract.
   Producer task reports retain current input/output comparisons and gaps. *)
open Err.Syntax
open Transformers_metadata.Json_util
module T = Transformers_tasks
module J = Jsont.Json

let defaults =
  obj
    [
      ("shortest_edge", J.int 224);
      ("crop_size", J.int 224);
      ("resample", J.string "bicubic");
      ( "image_mean",
        J.list (List.map J.number [ 0.48145466; 0.4578275; 0.40821073 ]) );
      ( "image_std",
        J.list (List.map J.number [ 0.26862954; 0.26130258; 0.27577711 ]) );
    ]

let sentences = function
  | [ "--lines"; file ] ->
      let+ bytes = Pt2_fixture_unix.Fetch.read file in
      T.Adapter.lines bytes
  | [] -> invalid "at least one sentence required"
  | all -> Ok all

let () =
  T.Demo.main "transformers_clip_demo" (fun () ->
      match List.tl (Array.to_list Sys.argv) with
      | "ids" :: tokenizer_path :: rest ->
          let* tokenizer = Pt2_fixture_unix.Fetch.read tokenizer_path in
          let* sentences = sentences rest in
          Err.List.iter
            (fun text ->
              let* inputs = T.Adapter.clip tokenizer ~length:16 text in
              let+ ids = T.Lifecycle.named "input_ids" inputs in
              print_endline
                (String.concat " "
                   (List.init 16 (fun i ->
                        Int64.to_string (Pt2_fixture.Logical.get_int64 ids i)))))
            sentences
      | [ "pixels"; ppm ] ->
          let* bytes = Pt2_fixture_unix.Fetch.read ppm in
          let+ pixels = T.Adapter.image defaults bytes in
          for i = 0 to Int64.to_int (Pt2_fixture.Logical.numel pixels) - 1 do
            Printf.printf "%.9g\n" (Pt2_fixture.Logical.get_float pixels i)
          done
      | "score" :: cohort :: cache :: assets_path :: tokenizer_path :: ppm
        :: rest ->
          let* texts = sentences rest in
          let* assets, _, fixture, contract =
            T.Demo.load Transformers_metadata.Assets.Tinyclip cohort cache
              assets_path
          in
          let* tokenizer = Pt2_fixture_unix.Fetch.read tokenizer_path in
          let* pin = member "tokenizer" assets in
          let* () = T.Demo.bytes_pin pin tokenizer in
          let* bytes = Pt2_fixture_unix.Fetch.read ppm in
          let* params = member "image" assets in
          let* pixels = T.Adapter.image params bytes in
          let* length = T.Adapter.extent "max_length" assets in
          Err.List.iter
            (fun text ->
              let* inputs = T.Adapter.clip tokenizer ~length text in
              let* inputs =
                T.Acceptance.order
                  (List.map
                     (fun (s : Pt2_fixture.Contract.Tensor_spec.t) ->
                       (s.name, ()))
                     contract.inputs)
                  (inputs @ [ ("pixel_values", pixels) ])
              in
              let* outputs = T.Input.run fixture.archive contract inputs in
              let+ logits = T.Lifecycle.named "logits_per_image" outputs in
              Fmt.pr "%-40S logits_per_image = %.6f@." text
                (Pt2_fixture.Logical.get_float logits 0))
            texts
      | _ ->
          invalid
            "usage: transformers_clip_demo (ids TOKENIZER S... | pixels \
             IMAGE.ppm | score COHORT CACHE ASSETS TOKENIZER IMAGE.ppm S...)")
