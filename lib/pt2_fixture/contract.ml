open Err.Syntax
module Dtype = Pt2_checkpoint_map.Dtype

module Tensor_spec = struct
  type t = { dtype : Dtype.t; name : string; shape : int64 list }
end

type t = {
  artifact_id : string;
  atol : float;
  dynamic : bool;
  graph_sha256 : Pt2_sha256.Digest.t;
  inputs : Tensor_spec.t list;
  outputs : Tensor_spec.t list;
  rtol : float;
  verified_cases : int;
}

module Wire = struct
  type tensor = { dtype : string; name : string; shape : int64 list }
  type call = { args : string list; kwargs : string list; result : string }
  type tolerances = { atol : float; rtol : float }

  type document = {
    artifact_id : string;
    call : call;
    dynamic_constraints : Jsont.json;
    graph_sha256 : string;
    inputs : tensor list;
    mutations : Jsont.json list;
    outputs : tensor list;
    tolerances : tolerances;
    verified_cases : int;
  }

  let tensor_jsont =
    Jsont.Object.map ~kind:"contract tensor" (fun dtype name shape ->
        { dtype; name; shape })
    |> Jsont.Object.mem "dtype" Jsont.string
    |> Jsont.Object.mem "name" Jsont.string
    |> Jsont.Object.mem "shape" (Jsont.list Jsont.int64)
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let call_jsont =
    Jsont.Object.map ~kind:"contract call" (fun args kwargs result ->
        { args; kwargs; result })
    |> Jsont.Object.mem "args" (Jsont.list Jsont.string)
    |> Jsont.Object.mem "kwargs" (Jsont.list Jsont.string)
    |> Jsont.Object.mem "outputs" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let tolerances_jsont =
    Jsont.Object.map ~kind:"tolerances" (fun atol rtol -> { atol; rtol })
    |> Jsont.Object.mem "atol" Jsont.number
    |> Jsont.Object.mem "rtol" Jsont.number
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"contract"
      (fun
        artifact_id
        call
        dynamic_constraints
        graph_sha256
        inputs
        mutations
        outputs
        tolerances
        verified_cases
      ->
        {
          artifact_id;
          call;
          dynamic_constraints;
          graph_sha256;
          inputs;
          mutations;
          outputs;
          tolerances;
          verified_cases;
        })
    |> Jsont.Object.mem "artifact_id" Jsont.string
    |> Jsont.Object.mem "call" call_jsont
    |> Jsont.Object.mem "dynamic_constraints" Jsont.json
    |> Jsont.Object.mem "graph_sha256" Jsont.string
    |> Jsont.Object.mem "inputs" (Jsont.list tensor_jsont)
    |> Jsont.Object.mem "mutations" (Jsont.list Jsont.json)
    |> Jsont.Object.mem "outputs" (Jsont.list tensor_jsont)
    |> Jsont.Object.mem "tolerances" tolerances_jsont
    |> Jsont.Object.mem "verified_cases" Jsont.int
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish
end

let spec_of (w : Wire.tensor) =
  match Dtype.of_torch_name w.dtype with
  | Some dtype ->
      Err.return { Tensor_spec.dtype; name = w.name; shape = w.shape }
  | None -> Err.fail (`Unknown_dtype w.dtype)

let unique what names =
  let rec go seen = function
    | [] -> Err.return ()
    | n :: rest ->
        if List.mem n seen then
          Err.fail
            (`Contract_shape (Printf.sprintf "duplicate %s name %S" what n))
        else go (n :: seen) rest
  in
  go [] names

let of_string text =
  let* w =
    Jsont_bytesrw.decode_string Wire.jsont text
    |> Err.import ~pos:__POS__ (fun e -> `Contract_decode e)
  in
  let* () =
    match w.mutations with
    | [] -> Err.return ()
    | ms -> Err.fail (`Contract_mutations (List.map (fun _ -> "mutation") ms))
  in
  let* () =
    if w.call.args <> [] then Err.fail (`Contract_shape "positional arguments")
    else if not (String.equal w.call.result "tensor_tuple") then
      Err.fail (`Contract_shape ("result is " ^ w.call.result))
    else Err.return ()
  in
  let* graph_sha256 =
    match Pt2_sha256.Digest.of_hex w.graph_sha256 with
    | Some d -> Err.return d
    | None -> Err.fail (`Bad_digest w.graph_sha256)
  in
  let* inputs = Err.List.map spec_of w.inputs in
  let* outputs = Err.List.map spec_of w.outputs in
  let names l = List.map (fun (s : Tensor_spec.t) -> s.name) l in
  let* () = unique "input" (names inputs) in
  let* () = unique "output" (names outputs) in
  let* () =
    if names inputs = w.call.kwargs then Err.return ()
    else Err.fail (`Contract_shape "keyword order differs from the input list")
  in
  let dynamic =
    match w.dynamic_constraints with Jsont.Object ([], _) -> false | _ -> true
  in
  Err.return
    {
      artifact_id = w.artifact_id;
      atol = w.tolerances.atol;
      dynamic;
      graph_sha256;
      inputs;
      outputs;
      rtol = w.tolerances.rtol;
      verified_cases = w.verified_cases;
    }
