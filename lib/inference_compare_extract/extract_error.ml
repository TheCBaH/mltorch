open Inference_compare_schema

type detail =
  | Digest of Inference_compare_codecs.Payload_hasher.error
  | Invalid_body of Inference_compare_plan.Plan_error.t
  | No_detail
  | Position of int

type t = {
  code : Failure.code;
  side : Side.t option;
  unit : Id.Unit.t option;
  node : Node_ref.t option;
  tensor : Tensor_ref.t option;
  detail : detail;
}

let make ?side ?unit ?node ?tensor ?(detail = No_detail) code =
  { code; side; unit; node; tensor; detail }

let pp ppf e =
  Fmt.string ppf
    (match Failure.encode_code e.code with
    | Jsont.String (s, _) -> s
    | _ -> "?");
  Option.iter (Fmt.pf ppf " side=%a" Side.pp) e.side;
  Option.iter (Fmt.pf ppf " unit=%a" Id.Unit.pp) e.unit;
  Option.iter (Fmt.pf ppf " node=%a" Node_ref.pp) e.node;
  Option.iter (Fmt.pf ppf " tensor=%a" Tensor_ref.pp) e.tensor;
  match e.detail with
  | No_detail -> ()
  | Position i -> Fmt.pf ppf " position=%d" i
  | Digest d ->
      Fmt.pf ppf " (%a)" Inference_compare_codecs.Payload_hasher.pp_error d
  | Invalid_body p -> Fmt.pf ppf " (%a)" Inference_compare_plan.Plan_error.pp p

let fail e = Err.fail ~pp_error:pp e
