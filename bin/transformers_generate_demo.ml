(* Usage: transformers_generate_demo.exe COHORT.json CACHE_DIR ID,ID,ID,ID

   A bounded greedy generation over two released artifacts of one model: the
   SmolLM2 prefill (4 tokens) and the static-history decode snapshot at history
   4. The prompt is four token ids (tokenization is outside this tool: it takes
   ids). The prefill's last-position logits give the first new token; its
   present K/V become the decode artifact's past, which is fed that token, and
   the decode step's logits give the second. That is all the released graphs can
   establish: there is no decode artifact at history 5, so the loop stops after
   two tokens, and [Pt2_fixture.History.chain] checks beforehand that the
   prefill's outputs and the decode's inputs meet.

   Verified against transformers' own greedy generation by
   scripts/transformers-generate-crosscheck.py (same pinned weights, torch
   2.12.0+cpu): same tokens, and the logits within the producer's tolerances. A
   claim for this model, these two artifacts and native-direct only. *)

open Err.Syntax
module Fixture = Pt2_fixture_unix.Fixture
module Replay = Pt2_fixture_replay

let prefill_id =
  "smollm2-135m/text-decoder/reference/prefill/fp32/dynamo/static/ckpt-12fd25f77366"

let decode_id =
  "smollm2-135m/text-decoder/reference/decode/fp32/dynamo/static-h4/ckpt-12fd25f77366"

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

let contiguous sizes =
  let rec go = function
    | [] -> []
    | _ :: rest as l -> List.fold_left ( * ) 1 (List.tl l) :: go rest
  in
  go sizes

(* A float32 Native tensor handed back to the engine as a named input. *)
let pt2_of_packed packed ~shape =
  match Replay.to_logical packed ~rank:(List.length shape) with
  | Error (`Native_layout why) -> failwith why
  | Ok (l : Pt2_fixture.Logical.t) ->
      {
        Pt2_tensor.dtype = Pt2_dtype.Float32;
        sizes = shape;
        strides = contiguous shape;
        storage_offset = 0;
        data = l.data;
      }

let f32_at (l : Pt2_fixture.Logical.t) i =
  Int32.float_of_bits (Pt2_storage.get_int32_le l.data (4 * i))

let argmax_at (l : Pt2_fixture.Logical.t) ~row ~width =
  let best = ref 0 and best_v = ref neg_infinity in
  for v = 0 to width - 1 do
    let x = f32_at l ((row * width) + v) in
    if x > !best_v then (
      best := v;
      best_v := x)
  done;
  (!best, !best_v)

let open_fixture config cohort id =
  let entry =
    List.find
      (fun (e : Pt2_fixture.Cohort.entry) -> String.equal e.artifact_id id)
      cohort.Pt2_fixture.Cohort.entries
  in
  Fixture.open_ config cohort entry

let () =
  match Array.to_list Sys.argv with
  | [ _; cohort_path; cache_dir; ids ] -> (
      let prompt = List.map int_of_string (String.split_on_char ',' ids) in
      if List.length prompt <> 4 then (
        prerr_endline "the prefill artifact takes exactly 4 token ids";
        exit 2);
      let setup =
        let* text = Pt2_fixture_unix.Fetch.read cohort_path in
        let* cohort = Pt2_fixture.Cohort.of_string text in
        let* cache = Pt2_fixture_unix.Cache.create cache_dir in
        let config = Pt2_fixture_unix.Bundle.config cache in
        let* prefill = open_fixture config cohort prefill_id in
        let* decode = open_fixture config cohort decode_id in
        Err.return (prefill, decode)
      in
      match Err.payload setup with
      | Error e ->
          Fmt.epr "%a@." Fixture.pp_error e;
          exit 2
      | Ok (prefill, decode) ->
          let contract (f : Fixture.t) =
            let text =
              match
                Err.payload
                  (Pt2_fixture_unix.Bundle.read_member f.bundle "contract.json")
              with
              | Ok t -> t
              | Error _ -> failwith "contract"
            in
            match Err.payload (Pt2_fixture.Contract.of_string text) with
            | Ok c -> (text, c)
            | Error _ -> failwith "contract decode"
          in
          let p_text, p_contract = contract prefill in
          let d_text, d_contract = contract decode in
          let history =
            match Pt2_fixture.History.of_contract_string d_text with
            | Ok (Some h) -> h
            | _ -> failwith "decode artifact is not a static-history snapshot"
          in
          ignore p_text;
          let shape_of (specs : Pt2_fixture.Contract.Tensor_spec.t list) =
            List.map
              (fun (s : Pt2_fixture.Contract.Tensor_spec.t) ->
                (s.name, s.shape))
              specs
          in
          (match
             Pt2_fixture.History.chain
               ~prefill:(shape_of p_contract.outputs)
               ~decode:history
               ~decode_inputs:(shape_of d_contract.inputs)
           with
          | Ok () -> ()
          | Error f ->
              Fmt.epr "the artifacts do not chain: %a@."
                Pt2_fixture.History.pp_fault f;
              exit 2);
          Fmt.pr "history chain: prefill (4) feeds decode (history %d)@."
            history.history;
          let run (f : Fixture.t) inputs =
            match
              Err.payload
                (Native_interp.run_named ~empty_caches:ignore f.archive ~inputs)
            with
            | Ok outs -> outs
            | Error e ->
                Fmt.epr "%a@." Native_interp.pp_error e;
                exit 1
          in
          let named (c : Pt2_fixture.Contract.t) outs =
            List.combine
              (List.map
                 (fun (s : Pt2_fixture.Contract.Tensor_spec.t) -> s.name)
                 c.outputs)
              outs
          in
          let vocab = 49152 in
          let t0 = Unix.gettimeofday () in
          let outs =
            named p_contract
              (run prefill
                 [
                   ("input_ids", pt2_i64 [ 1; 4 ] prompt);
                   ("attention_mask", pt2_i64 [ 1; 4 ] [ 1; 1; 1; 1 ]);
                 ])
          in
          let logits =
            match Replay.to_logical (List.assoc "logits" outs) ~rank:3 with
            | Ok l -> l
            | Error _ -> failwith "logits"
          in
          let t5, v5 = argmax_at logits ~row:3 ~width:vocab in
          Fmt.pr "prefill: first new token %d (logit %.6f) [%.0fs]@." t5 v5
            (Unix.gettimeofday () -. t0);
          let past =
            List.concat_map
              (fun (name, packed) ->
                match String.split_on_char '_' name with
                | [ "present"; i; kind ] ->
                    [
                      ( Printf.sprintf "past_%s_%s" i kind,
                        pt2_of_packed packed ~shape:[ 1; 3; 4; 64 ] );
                    ]
                | _ -> [])
              outs
          in
          let t1 = Unix.gettimeofday () in
          let outs' =
            named d_contract
              (run decode
                 ([
                    ("input_ids", pt2_i64 [ 1; 1 ] [ t5 ]);
                    ("attention_mask", pt2_i64 [ 1; 5 ] [ 1; 1; 1; 1; 1 ]);
                  ]
                 @ past))
          in
          let logits' =
            match Replay.to_logical (List.assoc "logits" outs') ~rank:3 with
            | Ok l -> l
            | Error _ -> failwith "logits"
          in
          let t6, v6 = argmax_at logits' ~row:0 ~width:vocab in
          Fmt.pr "decode:  second new token %d (logit %.6f) [%.0fs]@." t6 v6
            (Unix.gettimeofday () -. t1);
          Fmt.pr "generated: %s@."
            (String.concat "," (List.map string_of_int (prompt @ [ t5; t6 ])));
          (* The full last-position logits, for the cross-check. *)
          let dump name l ~row =
            Fmt.pr "logits %s: %s@." name
              (String.concat " "
                 (List.init vocab (fun v ->
                      Printf.sprintf "%.9g" (f32_at l ((row * vocab) + v)))))
          in
          if Sys.getenv_opt "GENERATE_DUMP_LOGITS" <> None then (
            dump "prefill" logits ~row:3;
            dump "decode" logits' ~row:0))
  | _ ->
      prerr_endline
        "usage: transformers_generate_demo COHORT.json CACHE_DIR ID,ID,ID,ID";
      exit 2
