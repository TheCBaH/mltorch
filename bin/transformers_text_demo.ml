(* Bounded ASCII BERT example. Pinned task reports establish current input/model
   comparisons; Unicode and offset coverage remain explicit acceptance items. *)
open Err.Syntax
open Transformers_metadata.Json_util
module T = Transformers_tasks

let () =
  T.Demo.main "transformers_text_demo" (fun () ->
      let args = List.tl (Array.to_list Sys.argv) in
      let ids_only, args =
        match args with "--ids" :: rest -> (true, rest) | _ -> (false, args)
      in
      match args with
      | cohort :: cache :: assets_path :: vocab_path :: sentences
        when sentences <> [] ->
          let* sentences =
            match sentences with
            | [ "--lines"; file ] ->
                let+ bytes = Pt2_fixture_unix.Fetch.read file in
                T.Adapter.lines bytes
            | all -> Ok all
          in
          let* () =
            T.Spec.require (sentences <> []) "at least one sentence required"
          in
          let* assets, _, fixture, contract =
            T.Demo.load Transformers_metadata.Assets.Bert cohort cache
              assets_path
          in
          let* vocab = Pt2_fixture_unix.Fetch.read vocab_path in
          let* pin = member "vocab" assets in
          let* () = T.Demo.bytes_pin pin vocab in
          let* length = T.Adapter.extent "max_length" assets in
          Err.List.iter
            (fun text ->
              let* inputs = T.Adapter.bert vocab ~length text in
              if ids_only then
                let+ ids = T.Lifecycle.named "input_ids" inputs in
                print_endline
                  (String.concat " "
                     (List.init length (fun i ->
                          Int64.to_string (Pt2_fixture.Logical.get_int64 ids i))))
              else
                let* inputs =
                  T.Acceptance.order
                    (List.map
                       (fun (s : Pt2_fixture.Contract.Tensor_spec.t) ->
                         (s.name, ()))
                       contract.inputs)
                    inputs
                in
                let* outputs = T.Input.run fixture.archive contract inputs in
                let+ pooled = T.Lifecycle.named "pooler_output" outputs in
                Fmt.pr "%S pooled[0..3]: %s@." text
                  (String.concat " "
                     (List.init 4 (fun i ->
                          Printf.sprintf "%.6f"
                            (Pt2_fixture.Logical.get_float pooled i)))))
            sentences
      | _ ->
          invalid
            "usage: transformers_text_demo [--ids] COHORT CACHE ASSETS VOCAB \
             (SENTENCE... | --lines FILE)")
