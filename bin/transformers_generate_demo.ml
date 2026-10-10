(* Four validated token IDs, one prefill and at most one history-four decode.
   Greedy token agreement is separate from numerical acceptance: historical
   reference checks had 20 and 495 over-tolerance logits. Pinned task checks
   retain every output and cache comparison; this example makes no parity claim. *)
open Err.Syntax
open Transformers_metadata.Json_util
module T = Transformers_tasks
module M = Transformers_metadata
module U = Pt2_fixture_unix
module F = Pt2_fixture

let prefill_id =
  "smollm2-135m/text-decoder/reference/prefill/fp32/dynamo/static/ckpt-12fd25f77366"

let decode_id =
  "smollm2-135m/text-decoder/reference/decode/fp32/dynamo/static-h4/ckpt-12fd25f77366"

let () =
  T.Demo.main "transformers_generate_demo" (fun () ->
      match Array.to_list Sys.argv with
      | [ _; cohort_path; cache_path; ids ] ->
          let* prompt =
            Err.List.map
              (fun s ->
                Err.of_option
                  (`Metadata_invalid ("invalid prompt ID: " ^ s))
                  (Int64.of_string_opt s))
              (String.split_on_char ',' ids)
          in
          let* () =
            T.Spec.require
              (List.length prompt = 4)
              "prefill requires exactly four token IDs"
          in
          let* cohort, config = T.Demo.config cohort_path cache_path in
          let* prefill = T.Demo.fixture config cohort prefill_id in
          let* decode = T.Demo.fixture config cohort decode_id in
          let contract f =
            U.Bundle.read_member f.U.Fixture.bundle "contract.json" >>= parse
          in
          let* p_json = contract prefill in
          let* d_json = contract decode in
          let* p, d, history = M.Chaining.check p_json d_json in
          let* logits =
            Err.of_option (`Metadata_missing "logits contract")
              (List.find_opt
                 (fun (s : F.Contract.Tensor_spec.t) -> s.name = "logits")
                 p.outputs)
          in
          let* vocab =
            match List.rev logits.shape with
            | n :: _ when n > 0L -> Ok n
            | _ -> invalid "logits vocabulary shape"
          in
          let* () =
            T.Spec.require
              (List.for_all (fun n -> n >= 0L && n < vocab) prompt)
              "prompt IDs outside vocabulary"
          in
          let* producer =
            M.Producer.load ~consumer_root:"."
              ~root:"modules/devcontainer.transformers"
          in
          let* config_bytes =
            M.Producer.checked_text producer.root producer.commit
              "configs/reference/smollm2-135m.json"
          in
          let* config_json = parse config_bytes in
          let* config_sha =
            path [ "weights"; "config_sha256" ] p_json >>= string
          in
          let* () =
            same ~identity:"generation" ~field:"config bytes"
              (M.Source.hash config_bytes)
              config_sha
          in
          let* eos_json = member "eos_token_id" config_json in
          let* eos =
            match eos_json with
            | Jsont.Array (xs, _) -> Err.List.map integer xs
            | value ->
                let+ n = integer value in
                [ n ]
          in
          let* state = T.Lifecycle.fresh ~eos ~maximum:2 in
          let* ids = T.Input.i64 [ 1L; 4L ] prompt in
          let* mask = T.Input.i64 [ 1L; 4L ] [ 1L; 1L; 1L; 1L ] in
          let* outputs =
            T.Input.run prefill.archive p
              [ ("input_ids", ids); ("attention_mask", mask) ]
          in
          let* state, token, stop = T.Lifecycle.advance state outputs in
          let* state =
            match stop with
            | T.Lifecycle.Eos | T.Lifecycle.Limit -> Ok state
            | T.Lifecycle.Continue ->
                let* inputs = T.Lifecycle.next ~history outputs mask token in
                let* outputs = T.Input.run decode.archive d inputs in
                let+ state, _, _ = T.Lifecycle.advance state outputs in
                state
          in
          Fmt.pr
            "generated: %s@.Stopped at EOS or the two-token exported limit; \
             history 5 is uncovered.@."
            (String.concat ","
               (List.map Int64.to_string (prompt @ state.generated)));
          Ok ()
      | _ ->
          invalid "usage: transformers_generate_demo COHORT CACHE ID,ID,ID,ID")
