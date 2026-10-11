(* The vocabulary of a certification: its stages, how a stage fails, and the
   small helpers every stage shares. *)

open Inference_compare_schema
module W = Wire
module Codecs = Inference_compare_codecs
module H = Codecs.Payload_hasher

type stage =
  | Contract
  | Reference_trajectory
  | Tested_trajectory
  | Identity_continuity
  | Full_graph_confirmation
  | Required_end_to_end
  | Golden

let stage_name = function
  | Contract -> "contract"
  | Reference_trajectory -> "reference_trajectory"
  | Tested_trajectory -> "tested_trajectory"
  | Identity_continuity -> "identity_continuity"
  | Full_graph_confirmation -> "full_graph_confirmation"
  | Required_end_to_end -> "required_end_to_end_checks"
  | Golden -> "golden_materialization"

type kind = Disagreement | Failed | Incomplete | Invalid

type failure = {
  stage : stage;
  kind : kind;
  detail : string;
  run : string option;  (** The run directory holding the evidence, if any. *)
}

let pp_failure ppf f =
  Fmt.pf ppf "%s: %s%a" (stage_name f.stage) f.detail
    (Fmt.option (fun ppf d -> Fmt.pf ppf " (evidence: %s)" d))
    f.run

let exit_code f =
  match f.kind with
  | Invalid -> 2
  | Disagreement -> 1
  | Failed -> 3
  | Incomplete -> 4

type certified = { directory : string; record : Golden_record.t }

type hooks = { before : stage -> unit }
(** A seam for fault injection: called just before each stage starts. *)

let no_hooks = { before = (fun _ -> ()) }
let ( let* ) = Result.bind
let fail ?run stage kind detail = Error { stage; kind; detail; run }
let invalid stage detail = fail stage Invalid detail

let io stage r =
  Result.map_error
    (fun e ->
      {
        stage;
        kind = Failed;
        detail = Format.asprintf "%a" File_io.pp_error e;
        run = None;
      })
    r

let load_error stage e =
  {
    stage;
    kind = Invalid;
    detail = Format.asprintf "%a" Session.pp_error e;
    run = None;
  }

let hashers = Session.hashers

let digest_text algorithm text =
  match H.digest_string hashers algorithm text with
  | Ok d -> Ok d
  | Error _ -> Error "digest backend unavailable"

let hex algorithm text =
  Result.map Digest.Payload.hex (digest_text algorithm text)

let make_digest mk algorithm text stage =
  match
    let* h = hex algorithm text in
    Result.map_error (fun _ -> "non-canonical digest") (mk algorithm ~hex:h)
  with
  | Ok d -> Ok d
  | Error m -> fail stage Failed m

let canonical_text enc v = W.canonical_string (enc v)
