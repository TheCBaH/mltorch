open Err.Syntax

module Capture = struct
  type t = {
    dtype : Dtype.t;
    kind : Fault.capture_kind;
    live : bool;
    referenced : bool;
    scalar : bool;
    shape : int64 list;
    target : string;
    value_sha256 : Pt2_sha256.Digest.t;
  }
end

type t = {
  artifact_id : string;
  captures : Capture.t list;
  graph_sha256 : Pt2_sha256.Digest.t;
}

module Wire = struct
  type source = { value_sha256 : string }

  let source_jsont =
    Jsont.Object.map ~kind:"capture source" (fun value_sha256 ->
        { value_sha256 })
    |> Jsont.Object.mem "value_sha256" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  type capture = {
    dtype : string;
    kind : string;
    live : bool;
    referenced : bool;
    scalar : bool;
    shape : int64 list;
    source : source;
    target : string;
  }

  let capture_jsont =
    Jsont.Object.map ~kind:"capture"
      (fun dtype kind live referenced scalar shape source target ->
        { dtype; kind; live; referenced; scalar; shape; source; target })
    |> Jsont.Object.mem "dtype" Jsont.string
    |> Jsont.Object.mem "kind" Jsont.string
    |> Jsont.Object.mem "live" Jsont.bool
    |> Jsont.Object.mem "referenced" Jsont.bool
    |> Jsont.Object.mem "scalar" Jsont.bool
    |> Jsont.Object.mem "shape" (Jsont.list Jsont.int64)
    |> Jsont.Object.mem "source" source_jsont
    |> Jsont.Object.mem "target" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  type document = {
    artifact_id : string;
    captures : capture list;
    graph_sha256 : string;
  }

  let document_jsont =
    Jsont.Object.map ~kind:"captures" (fun artifact_id captures graph_sha256 ->
        { artifact_id; captures; graph_sha256 })
    |> Jsont.Object.mem "artifact_id" Jsont.string
    |> Jsont.Object.mem "captures" (Jsont.list capture_jsont)
    |> Jsont.Object.mem "graph_sha256" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish
end

let digest_of s =
  match Pt2_sha256.Digest.of_hex s with
  | Some d -> Err.return d
  | None -> Err.fail (`Bad_digest s)

let kind_of = function
  | "BUFFER" -> Some Fault.Buffer
  | "CONSTANT_TENSOR" -> Some Fault.Constant_tensor
  | "PARAMETER" -> Some Fault.Parameter
  | _ -> None

let capture_of (w : Wire.capture) =
  let* dtype =
    match Dtype.of_torch_name w.dtype with
    | Some d -> Err.return d
    | None -> Err.fail (`Unknown_dtype w.dtype)
  in
  let* kind =
    match kind_of w.kind with
    | Some k -> Err.return k
    | None -> Err.fail (`Captures_decode ("unknown capture kind " ^ w.kind))
  in
  let+ value_sha256 = digest_of w.source.value_sha256 in
  {
    Capture.dtype;
    kind;
    live = w.live;
    referenced = w.referenced;
    scalar = w.scalar;
    shape = w.shape;
    target = w.target;
    value_sha256;
  }

let of_string ?(limits = Limits.default) text =
  let* () =
    if String.length text > limits.Limits.max_document_bytes then
      Err.fail
        (`Over_limit
           {
             Fault.Over_limit.what = Limits.Document_bytes;
             limit = Int64.of_int limits.max_document_bytes;
             actual = Int64.of_int (String.length text);
           })
    else Err.return ()
  in
  let* w =
    Jsont_bytesrw.decode_string Wire.document_jsont text
    |> Err.import ~pos:__POS__ (fun e -> `Captures_decode e)
  in
  let n = List.length w.captures in
  let* () =
    if n > limits.max_captures then
      Err.fail
        (`Over_limit
           {
             Fault.Over_limit.what = Limits.Captures;
             limit = Int64.of_int limits.max_captures;
             actual = Int64.of_int n;
           })
    else Err.return ()
  in
  let* graph_sha256 = digest_of w.graph_sha256 in
  let* captures = Err.List.map capture_of w.captures in
  let seen = Hashtbl.create 64 in
  let* () =
    Err.List.iter
      (fun (c : Capture.t) ->
        if Hashtbl.mem seen c.target then
          Err.fail (`Duplicate (Fault.Capture, c.target))
        else begin
          Hashtbl.add seen c.target ();
          Err.return ()
        end)
      captures
  in
  Err.return { artifact_id = w.artifact_id; captures; graph_sha256 }
