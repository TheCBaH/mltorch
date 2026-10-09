module J = Jsont.Json

type status = Failed | Passed | Refused

type case = {
  error : string option;
  id : string;
  inputs_digest_ok : bool;
  outputs : Compare.t list;
  outputs_digest_ok : bool;
}

type t = {
  artifact_id : string;
  atol : float;
  backend : string;
  cases : case list;
  consumer : string;
  normalizations : string list;
  pins : (string * string) list;
  refusal : string option;
  scope : string option;
  rtol : float;
  status : status;
}

let schema_version = 3

let case_passed c =
  c.error = None && c.inputs_digest_ok && c.outputs_digest_ok && c.outputs <> []
  && List.for_all Compare.passed c.outputs

let status_of_cases cases =
  if cases <> [] && List.for_all case_passed cases then Passed else Failed

let status_name = function
  | Failed -> "failed"
  | Passed -> "passed"
  | Refused -> "refused"

let verdict_name : Compare.verdict -> string = function
  | Dtype_differs _ -> "dtype_differs"
  | Pass -> "pass"
  | Shape_differs _ -> "shape_differs"
  | Unsupported_dtype _ -> "unsupported_dtype"
  | Values_differ -> "values_differ"

let mem k v = J.mem (J.name k) v
let obj members = J.object' (List.map (fun (k, v) -> mem k v) members)
let str s = J.string s
let i64 n = J.string (Int64.to_string n)
let shape s = J.list (List.map i64 s)

(* JSON has no NaN or infinity: such a number is reported as its name. *)
let number f =
  if Float.is_finite f then J.number f else J.string (string_of_float f)

let diff (d : Compare.Diff.t) =
  obj
    [
      ("index", shape d.index);
      ("actual", str d.actual);
      ("expected", str d.expected);
    ]

let verdict_detail : Compare.verdict -> (string * Jsont.json) list = function
  | Dtype_differs { actual; expected } ->
      [
        ("actual_dtype", str (Pt2_checkpoint_map.Dtype.to_code actual));
        ("expected_dtype", str (Pt2_checkpoint_map.Dtype.to_code expected));
      ]
  | Shape_differs { actual; expected } ->
      [ ("actual_shape", shape actual); ("expected_shape", shape expected) ]
  | Unsupported_dtype d ->
      [ ("dtype", str (Pt2_checkpoint_map.Dtype.to_code d)) ]
  | Pass | Values_differ -> []

let output (c : Compare.t) =
  obj
    ([
       ("name", str c.name);
       ("verdict", str (verdict_name c.verdict));
       ("elements", i64 c.elements);
       ("mismatches", i64 c.mismatches);
       ("max_abs_error", number c.max_abs_error);
       ("max_rel_error", number c.max_rel_error);
       ("first_mismatches", J.list (List.map diff c.first));
     ]
    @ verdict_detail c.verdict)

let case (c : case) =
  obj
    [
      ("id", str c.id);
      ("passed", J.bool (case_passed c));
      ("inputs_digest_verified", J.bool c.inputs_digest_ok);
      ("outputs_digest_verified", J.bool c.outputs_digest_ok);
      ("error", match c.error with Some e -> str e | None -> J.null ());
      ("outputs", J.list (List.map output c.outputs));
    ]

let to_string t =
  let json =
    obj
      [
        ("schema_version", J.int schema_version);
        ("artifact_id", str t.artifact_id);
        ("status", str (status_name t.status));
        ("refusal", match t.refusal with Some r -> str r | None -> J.null ());
        ("scope", match t.scope with Some r -> str r | None -> J.null ());
        ("backend", str t.backend);
        ("consumer", str t.consumer);
        ("normalizations", J.list (List.map str t.normalizations));
        ("tolerances", obj [ ("atol", number t.atol); ("rtol", number t.rtol) ]);
        ("pins", obj (List.map (fun (k, v) -> (k, str v)) t.pins));
        ("cases", J.list (List.map case t.cases));
      ]
  in
  match Jsont_bytesrw.encode_string ~format:Jsont.Indent Jsont.json json with
  | Ok s -> s ^ "\n"
  | Error e -> failwith ("report encoding: " ^ e)
