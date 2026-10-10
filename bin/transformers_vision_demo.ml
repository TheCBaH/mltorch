(* MobileViT PPM example: actual config/processor bytes and the supported BGR
   recipe are verified before model execution. Top-five agreement is separate
   from full-tensor acceptance; the historical scene-logit failure stays open. *)
open Err.Syntax
open Transformers_metadata.Json_util
module T = Transformers_tasks

let defaults =
  obj
    [
      ("shortest_edge", Jsont.Json.int 288);
      ("crop_size", Jsont.Json.int 256);
      ("resample", Jsont.Json.string "bilinear");
      ("flip_channels", Jsont.Json.bool true);
    ]

let () =
  T.Demo.main "transformers_vision_demo" (fun () ->
      match List.tl (Array.to_list Sys.argv) with
      | [ "pixels"; ppm ] ->
          let* bytes = Pt2_fixture_unix.Fetch.read ppm in
          let+ pixels = T.Adapter.image defaults bytes in
          for i = 0 to Int64.to_int (Pt2_fixture.Logical.numel pixels) - 1 do
            Printf.printf "%.9g\n" (Pt2_fixture.Logical.get_float pixels i)
          done
      | [ "classify"; cohort; cache; assets_path; config_path; ppm ] ->
          let* assets, _, fixture, contract =
            T.Demo.load Transformers_metadata.Assets.Mobilevit cohort cache
              assets_path
          in
          let* config = Pt2_fixture_unix.Fetch.read config_path in
          let* pin = member "config" assets in
          let* () = T.Demo.bytes_pin pin config in
          let* config = parse config in
          let* labels = member "id2label" config in
          let* bytes = Pt2_fixture_unix.Fetch.read ppm in
          let* params = member "image" assets in
          let* pixels = T.Adapter.image params bytes in
          let* outputs =
            T.Input.run fixture.archive contract [ ("pixel_values", pixels) ]
          in
          let* logits = T.Lifecycle.named "logits" outputs in
          let count = Int64.to_int (Pt2_fixture.Logical.numel logits) in
          let scores =
            List.init count (fun i ->
                (i, Pt2_fixture.Logical.get_float logits i))
          in
          let top =
            List.sort
              (fun (i, a) (j, b) ->
                let c = Float.compare b a in
                if c = 0 then Int.compare i j else c)
              scores
            |> List.filteri (fun i _ -> i < 5)
          in
          let* () =
            Err.List.iter
              (fun (i, v) ->
                let+ label = field (string_of_int i) labels in
                Fmt.pr "%4d %-34s %.6f@." i label v)
              top
          in
          if Sys.getenv_opt "VISION_DUMP_LOGITS" <> None then
            Fmt.pr "logits: %s@."
              (String.concat " "
                 (List.map (fun (_, v) -> Printf.sprintf "%.9g" v) scores));
          Ok ()
      | _ ->
          invalid
            "usage: transformers_vision_demo (pixels IMAGE.ppm | classify \
             COHORT CACHE ASSETS CONFIG IMAGE.ppm)")
