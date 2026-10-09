open Err.Syntax

module Case = struct
  type t = {
    id : string;
    inputs : string list;
    inputs_sha256 : Pt2_sha256.Digest.t;
    outputs : string list;
    outputs_sha256 : Pt2_sha256.Digest.t;
  }
end

type t = {
  artifact_id : string;
  atol : float;
  cases : Case.t list;
  rtol : float;
}

module Wire = struct
  type case = {
    id : string;
    inputs : string list;
    inputs_sha256 : string;
    outputs : string list;
    outputs_sha256 : string;
  }

  type tolerances = { atol : float; rtol : float }

  type document = {
    artifact_id : string;
    cases : case list;
    tolerances : tolerances;
  }

  let case_jsont =
    Jsont.Object.map ~kind:"case"
      (fun id inputs inputs_sha256 outputs outputs_sha256 ->
        { id; inputs; inputs_sha256; outputs; outputs_sha256 })
    |> Jsont.Object.mem "id" Jsont.string
    |> Jsont.Object.mem "inputs" (Jsont.list Jsont.string)
    |> Jsont.Object.mem "inputs_sha256" Jsont.string
    |> Jsont.Object.mem "outputs" (Jsont.list Jsont.string)
    |> Jsont.Object.mem "outputs_sha256" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let tolerances_jsont =
    Jsont.Object.map ~kind:"tolerances" (fun atol rtol -> { atol; rtol })
    |> Jsont.Object.mem "atol" Jsont.number
    |> Jsont.Object.mem "rtol" Jsont.number
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"cases" (fun artifact_id cases tolerances ->
        { artifact_id; cases; tolerances })
    |> Jsont.Object.mem "artifact_id" Jsont.string
    |> Jsont.Object.mem "cases" (Jsont.list case_jsont)
    |> Jsont.Object.mem "tolerances" tolerances_jsont
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish
end

let digest_of s =
  match Pt2_sha256.Digest.of_hex s with
  | Some d -> Err.return d
  | None -> Err.fail (`Bad_digest s)

let case_of (w : Wire.case) =
  let* inputs_sha256 = digest_of w.inputs_sha256 in
  let+ outputs_sha256 = digest_of w.outputs_sha256 in
  {
    Case.id = w.id;
    inputs = w.inputs;
    inputs_sha256;
    outputs = w.outputs;
    outputs_sha256;
  }

let of_string text =
  let* w =
    Jsont_bytesrw.decode_string Wire.jsont text
    |> Err.import ~pos:__POS__ (fun e -> `Cases_decode e)
  in
  let+ cases = Err.List.map case_of w.cases in
  {
    artifact_id = w.artifact_id;
    atol = w.tolerances.atol;
    cases;
    rtol = w.tolerances.rtol;
  }

let field field ~actual ~expected =
  if String.equal actual expected then Err.return ()
  else
    Err.fail
      (`Field_clash
         { Fault.Field_clash.layer = Fault.Cases; field; actual; expected })

let check (contract : Contract.t) (t : t) =
  let* () =
    field Fault.Artifact_id ~actual:t.artifact_id ~expected:contract.artifact_id
  in
  let* () =
    if t.cases <> [] && List.length t.cases = contract.verified_cases then
      Err.return ()
    else
      field Fault.Cases
        ~actual:(string_of_int (List.length t.cases))
        ~expected:(string_of_int contract.verified_cases)
  in
  let* () =
    if Float.equal t.atol contract.atol && Float.equal t.rtol contract.rtol then
      Err.return ()
    else
      field Fault.Tolerances
        ~actual:(Printf.sprintf "atol=%g rtol=%g" t.atol t.rtol)
        ~expected:(Printf.sprintf "atol=%g rtol=%g" contract.atol contract.rtol)
  in
  let names l = List.map (fun (s : Contract.Tensor_spec.t) -> s.name) l in
  Err.List.iter
    (fun (index, (c : Case.t)) ->
      let* () =
        field Fault.Case_ids ~actual:c.id
          ~expected:(Printf.sprintf "case-%02d" index)
      in
      let* () =
        if c.inputs = names contract.inputs then Err.return ()
        else
          Err.fail
            (`Case_names
               {
                 Fault.Case_names.case = c.id;
                 role = Fault.Inputs;
                 actual = c.inputs;
                 expected = names contract.inputs;
               })
      in
      if c.outputs = names contract.outputs then Err.return ()
      else
        Err.fail
          (`Case_names
             {
               Fault.Case_names.case = c.id;
               role = Fault.Outputs;
               actual = c.outputs;
               expected = names contract.outputs;
             }))
    (List.mapi (fun i c -> (i, c)) t.cases)
