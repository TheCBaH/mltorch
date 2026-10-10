(* Host model identity checks precede the pure tensor/history check. *)
open Err.Syntax
open Json_util

let contracts prefill decode =
  Err.List.iter
    (fun key ->
      let* a = member key prefill in
      let* b = member key decode in
      equal ~identity:"generation components" ~field:key a b)
    [ "model_id"; "weights" ]

let check prefill decode =
  let* () = contracts prefill decode in
  let* p = text prefill in
  let* d = text decode in
  let* p = Pt2_fixture.Contract.of_string p in
  let* c = Pt2_fixture.Contract.of_string d in
  let* history =
    Err.import
      (fun e -> `Metadata_invalid e)
      (Pt2_fixture.History.of_contract_string d)
  in
  let* history =
    Err.of_option (`Metadata_invalid "decode requires static history") history
  in
  let+ () =
    Err.import
      (fun e -> `Metadata_invalid (Fmt.str "%a" Pt2_fixture.History.pp_fault e))
      (Pt2_fixture.History.chain_tensors ~prefill:p.outputs ~decode:history
         ~decode_inputs:c.inputs)
  in
  (p, c, history)
