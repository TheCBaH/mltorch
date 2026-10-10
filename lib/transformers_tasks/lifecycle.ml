(* A fresh value per prompt. The only decode step consumes the complete,
   unmodified cache returned by prefill; history five has no selected graph. *)
open Err.Syntax
open Transformers_metadata.Json_util
module L = Pt2_fixture.Logical
module D = Pt2_checkpoint_map.Dtype

type stop = Continue | Eos | Limit
type t = { generated : int64 list; eos : int64 list; maximum : int }

let fresh ~eos ~maximum =
  let+ () =
    Spec.require
      (maximum >= 1 && maximum <= 2 && List.for_all (( <= ) 0L) eos)
      "bounded generation policy"
  in
  { generated = []; eos; maximum }

let named name values =
  Err.of_option (`Missing_tensor name) (List.assoc_opt name values)

let greedy logits =
  let* _ = Input.tensor logits in
  match (logits.L.dtype, logits.shape) with
  | D.F32, [ 1L; rows; width ]
    when rows > 0L && width > 0L && width <= 1_000_000L ->
      let start = Int64.to_int (Int64.mul (Int64.pred rows) width) in
      let count = Int64.to_int width in
      let best = ref 0 and value = ref neg_infinity and finite = ref true in
      for i = 0 to count - 1 do
        let x = L.get_float logits (start + i) in
        if not (Float.is_finite x) then finite := false;
        if x > !value then (
          best := i;
          value := x)
      done;
      let+ () = Spec.require !finite "nonfinite generation logits" in
      Int64.of_int !best
  | _ -> invalid "generation requires batch-one float32 logits"

let advance state outputs =
  let* () =
    Spec.require
      (List.length state.generated < state.maximum)
      "generation already stopped at limit"
  in
  let* () =
    Spec.require
      (not (List.exists (fun n -> List.mem n state.eos) state.generated))
      "generation already stopped at EOS"
  in
  let* logits = named "logits" outputs in
  let+ token = greedy logits in
  let generated = state.generated @ [ token ] in
  let stop =
    if List.mem token state.eos then Eos
    else if List.length generated = state.maximum then Limit
    else Continue
  in
  ({ state with generated }, token, stop)

let next ~history outputs mask token =
  let* _ = Input.tensor mask in
  let* () =
    Spec.require
      (mask.L.dtype = D.I64
      && mask.shape = [ 1L; Int64.of_int history.Pt2_fixture.History.history ])
      "prefill mask/history mismatch"
  in
  let length = history.history in
  let* () =
    Spec.require
      (List.for_all
         (fun i -> L.get_int64 mask i = 1L)
         (List.init length Fun.id))
      "padded generation prompt unsupported"
  in
  let* () =
    Err.import
      (fun e -> `Metadata_invalid (Fmt.str "%a" Pt2_fixture.History.pp_fault e))
      (Pt2_fixture.History.check_feed history ~requested:length)
  in
  let* past =
    Err.List.map
      (fun name ->
        let* () =
          Spec.require
            (String.starts_with ~prefix:"past_" name)
            "unsupported cache name"
        in
        let present = "present_" ^ String.sub name 5 (String.length name - 5) in
        let+ value = named present outputs in
        (name, value))
      history.state_inputs
  in
  let* ids = Input.i64 [ 1L; 1L ] [ token ] in
  let+ mask =
    Input.i64
      [ 1L; Int64.of_int history.attention_length ]
      (List.init history.attention_length (fun _ -> 1L))
  in
  [ ("input_ids", ids); ("attention_mask", mask) ] @ past
