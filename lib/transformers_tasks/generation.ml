open Err.Syntax
open Transformers_metadata.Json_util
module F = Pt2_fixture
module J = Jsont.Json

let stop_name = function
  | Lifecycle.Continue -> J.null ()
  | Lifecycle.Eos -> J.string "eos"
  | Lifecycle.Limit -> J.string "max_new_tokens"

let run config cohort producer bundle =
  let* () = Adapter.recipe producer bundle.Reference.Bundle.manifest in
  let* recipe_id = field "recipe_id" bundle.manifest in
  let* () =
    Spec.require
      (recipe_id = "smollm2-static-h4-v1")
      "generation recipe unsupported"
  in
  let* p, d =
    match bundle.references with
    | [ p; d ] -> Ok (p, d)
    | _ -> invalid "generation requires prefill and decode references"
  in
  let* pc, dc, history =
    Transformers_metadata.Chaining.check p.contract d.contract
  in
  let* prefill = Demo.execution_fixture config cohort p in
  let* decode = Demo.execution_fixture config cohort d in
  let* eos =
    path [ "recipe"; "eos_token_ids" ] bundle.manifest
    >>= array >>= Err.List.map integer
  in
  let* maximum =
    path [ "recipe"; "max_new_tokens" ] bundle.manifest >>= integer
  in
  let* () =
    Spec.require (maximum = 2L)
      "generation fixture must be bounded to two tokens"
  in
  let* cases = member "cases" bundle.manifest >>= array in
  let active = ref None and seen = ref [] in
  let* reports =
    Err.List.map
      (fun row ->
        let* id = field "id" row in
        let result () =
          let* transition = member "transition" row in
          let* sequence = member "sequence" transition >>= integer in
          let* reset = member "reset" transition >>= bool in
          let* previous = member "previous_case" transition in
          let* expected_inputs = Adapter.role bundle row "inputs" in
          let* artifact, _ = Acceptance.reference bundle row in
          let* state, inputs, fixture, contract =
            if reset then (
              let* () =
                Spec.require
                  (not (List.mem sequence !seen))
                  "duplicate prompt/reset sequence"
              in
              let* () =
                equal ~identity:id ~field:"reset previous case" previous
                  (J.null ())
              in
              let* () =
                match !active with
                | Some (_, _, _, _, _, Lifecycle.Continue) ->
                    invalid "reset before previous sequence stopped"
                | _ -> Ok ()
              in
              seen := sequence :: !seen;
              let+ state = Lifecycle.fresh ~eos ~maximum:2 in
              (state, expected_inputs, prefill, pc))
            else
              match !active with
              | Some (seq, last, state, outputs, mask, Lifecycle.Continue)
                when seq = sequence ->
                  let* () =
                    equal ~identity:id ~field:"previous case" previous
                      (J.string last)
                  in
                  let* token =
                    Lifecycle.named "logits" outputs >>= Lifecycle.greedy
                  in
                  let+ inputs = Lifecycle.next ~history outputs mask token in
                  (state, inputs, decode, dc)
              | _ ->
                  invalid
                    "decode requires an active previous step in the same \
                     sequence"
          in
          let* () =
            same ~identity:id ~field:"component" artifact
              contract.F.Contract.artifact_id
          in
          let* inputs = Acceptance.order expected_inputs inputs in
          let* _ = Input.validate contract inputs in
          let* input_checks =
            Diagnostic.compare ~atol:contract.atol ~rtol:contract.rtol
              expected_inputs inputs
          in
          let* mask = Lifecycle.named "attention_mask" inputs in
          let* ids = Lifecycle.named "input_ids" inputs in
          let* count =
            match ids.shape with
            | [ 1L; n ] when n > 0L && n <= 1_000_000L -> Ok n
            | _ -> invalid "generation requires bounded batch-one token IDs"
          in
          let* input_history = member "history_in" transition >>= integer in
          let* () =
            Spec.require
              (mask.shape = [ 1L; Int64.add input_history count ])
              "mask/position history"
          in
          let* positions =
            Input.i64 ids.shape
              (List.init (Int64.to_int count) (fun i ->
                   Int64.add input_history (Int64.of_int i)))
          in
          let* preprocessing = Adapter.role bundle row "preprocessing" in
          let* expected_positions =
            Lifecycle.named "position_ids" preprocessing
          in
          let* position_checks =
            Diagnostic.compare ~atol:0. ~rtol:0.
              [ ("position_ids", expected_positions) ]
              [ ("position_ids", positions) ]
          in
          let before =
            List.map
              (fun (name, v) -> (name, Pt2_sha256.bigstring v.F.Logical.data))
              inputs
          in
          let normalizations = ref [] in
          let on_empty_caches r = normalizations := Input.normalizations r in
          let* outputs =
            Input.run ~on_empty_caches fixture.archive contract inputs
          in
          let* () =
            Err.List.iter
              (fun ((name, hash), (_, v)) ->
                F.Check.digest (F.Fault.Member name)
                  (Pt2_sha256.bigstring v.F.Logical.data)
                  hash)
              (List.combine before inputs)
          in
          let* expected_outputs = Adapter.role bundle row "outputs" in
          let* output_checks =
            Diagnostic.compare ~atol:contract.atol ~rtol:contract.rtol
              expected_outputs outputs
          in
          let* state, token, stop = Lifecycle.advance state outputs in
          let* output_history = member "history_out" transition >>= integer in
          let* () =
            Err.List.iter
              (fun (name, v) ->
                if String.starts_with ~prefix:"present_" name then
                  match F.History.history_of_shape v.F.Logical.shape with
                  | Some n when Int64.of_int n = output_history -> Ok ()
                  | _ -> invalid "cache output history mismatch"
                else Ok ())
              outputs
          in
          let edges =
            List.filter_map
              (fun (name, _) ->
                if String.starts_with ~prefix:"present_" name then
                  Some
                    ( name,
                      J.string
                        ("past_" ^ String.sub name 8 (String.length name - 8))
                    )
                else None)
              outputs
          in
          let* claimed_edges = member "cache_edges" transition in
          let* () =
            equal ~identity:id ~field:"cache bindings" claimed_edges (obj edges)
          in
          let* claimed_stop = member "stop" transition in
          let* () =
            equal ~identity:id ~field:"stop policy" claimed_stop
              (stop_name stop)
          in
          let* host = Adapter.role bundle row "host" in
          let* generated_token = Input.i64 [ 1L; 1L ] [ token ] in
          let* generated_tokens =
            Input.i64
              [ 1L; Int64.of_int (List.length state.generated) ]
              state.generated
          in
          let* host_checks =
            Diagnostic.compare ~atol:0. ~rtol:0. host
              [
                ("generated_token", generated_token);
                ("generated_tokens", generated_tokens);
              ]
          in
          active := Some (sequence, id, state, outputs, mask, stop);
          let* report =
            Acceptance.report ~normalizations:!normalizations
              ~pins:(Demo.execution_pins fixture)
              artifact id ~atol:contract.atol ~rtol:contract.rtol
              (input_checks @ position_checks @ output_checks @ host_checks)
          in
          Ok
            (obj
               [
                 ("id", J.string id);
                 ("status", J.string "compared");
                 ("adapter", report);
                 ("model", J.null ());
                 ("transition", transition);
               ])
        in
        match Err.payload (result ()) with
        | Ok report -> Ok report
        | Error e ->
            active := None;
            Ok
              (obj
                 [
                   ("id", J.string id);
                   ("status", J.string "refused");
                   ("error", J.string (Fmt.str "%a" Fault.pp e));
                 ]))
      cases
  in
  let* () =
    match !active with
    | Some (_, _, _, _, _, Lifecycle.Continue) ->
        invalid "incomplete generation sequence"
    | _ -> Ok ()
  in
  let* fixture_id = member "fixture_id" bundle.manifest in
  Ok
    (obj
       [
         ("schema_version", J.int 1);
         ("fixture_id", fixture_id);
         ("recipe_id", J.string recipe_id);
         ("status", J.string "bounded chain compared");
         ("consumer_acceptance", J.string "not promoted");
         ( "coverage",
           J.string
             "producer prompt IDs; raw-text BPE adapter deferred; no \
              history-five decode" );
         ("cases", J.list reports);
       ])
