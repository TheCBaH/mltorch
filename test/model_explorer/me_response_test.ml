(* [Me_response]: the worker's answer and the seam it crosses on
   (.ai/model_explorer_design.md).

   What a round trip cannot show, and what this pins instead: that the byte
   count is DERIVED from the document rather than supplied beside it, that
   payload presence follows from the constructor, and that the writer limit is
   a branch something can actually reach. *)

module MR = Me_request
module Rsp = Me_response
module L = Me_limits.Limits

let limits = L.untrusted

(* [untrusted] rather than [small], so the profile encodes as [{}] and this
   file's goldens stay about the RESPONSE. That the profile crosses at all, and
   what it looks like, belongs to the wire-limits suite. *)
let wire =
  Err.or_raise ~pp_error:Me_limits.pp_error
    (Me_limits.Wire_limits.of_limits ~ceiling:L.untrusted L.untrusted)

let uuid = "0f8fad5b-d9cb-469f-a165-70867728950e"

let id =
  Err.or_raise ~pp_error:MR.Request.pp_error
    (MR.Request_id.of_string (uuid ^ "-3"))

let key =
  Err.or_raise ~pp_error:MR.Request.pp_error
    (MR.Detail_key.create ~limits ~parent_graph:"g/native/001"
       ~value:(Graph_ir.Tensor_id.of_int 4))

let diagnostic =
  Me_limits.Diagnostic.create ~limits Me_limits.Diagnostic.Code.Over_limit
    "graph nodes = 99 is over the ceiling"

let session json = Rsp.Final.Session { Rsp.Session.id; limits = wire; json }
let delta json = Rsp.Final.Delta { Rsp.Delta.id; key; json }

let failed =
  Rsp.Final.Failed { Rsp.Failed.id; key = Some key; error = diagnostic }

let protocol =
  Rsp.Final.Protocol_failure { Rsp.Protocol_failure.error = diagnostic }

let progress =
  { Rsp.Progress.id; phase = Rsp.Phase.Project; done_ = 7L; total = Some 9L }

let pp_wire ppf r =
  Core.Pretty.err_result
    ~ok:(fun ppf (w : Rsp.Wire.t) ->
      Fmt.pf ppf "%s@\npayload=%a" w.Rsp.Wire.meta
        (Core.Pretty.option_or ~none:"none" (fun ppf s ->
             Fmt.pf ppf "%d bytes" (String.length s)))
        w.Rsp.Wire.payload)
    ~error:Rsp.Wire.pp_error ppf r

(* --- the five messages --- *)

let%expect_test "a session carries its document and a count derived from it" =
  (* There is no [bytes] field on [Session.t] at all: it is measured here, from
     the string that is being sent, so "the count disagrees with the payload" is
     not a state anyone can build. *)
  Format.printf "%a@." pp_wire
    (Rsp.Wire.of_final (session {|{"schemaVersion":1}|}));
  [%expect
    {|
    {"kind":"session","id":"0f8fad5b-d9cb-469f-a165-70867728950e-3","limits":{},"bytes":19}
    payload=19 bytes |}]

let pp_shape ppf r =
  Core.Pretty.err_result
    ~ok:(fun ppf (w : Rsp.Wire.t) ->
      Fmt.pf ppf "payload=%a"
        (Core.Pretty.option_or ~none:"none" (fun ppf s ->
             Fmt.pf ppf "%d bytes" (String.length s)))
        w.Rsp.Wire.payload)
    ~error:Rsp.Wire.pp_error ppf r

let%expect_test "which messages have a payload, and which cannot" =
  (* Payload presence follows from the CONSTRUCTOR. It is not a field the shell
     fills in, so a failure carrying a document and a session without one are
     both unwritable -- and a progress message cannot have one at all, because
     [of_progress] takes a type that has no document to give it. *)
  List.iter
    (fun (label, w) -> Format.printf "%-17s %a@." label pp_shape w)
    [
      ("session", Rsp.Wire.of_final (session "{}"));
      ("delta", Rsp.Wire.of_final (delta "{}"));
      ("failed", Rsp.Wire.of_final failed);
      ("protocol failure", Rsp.Wire.of_final protocol);
      ("progress", Rsp.Wire.of_progress progress);
    ];
  [%expect
    {|
    session           payload=2 bytes
    delta             payload=2 bytes
    failed            payload=none
    protocol failure  payload=none
    progress          payload=none |}]

(* --- the metadata is the only thing that crosses --- *)

let round_trip w =
  Result.bind (Rsp.Meta.decode w.Rsp.Wire.meta) (fun m ->
      Result.map
        (fun again -> String.equal w.Rsp.Wire.meta again)
        (Jsont_bytesrw.encode_string Rsp.Meta.jsont m))

let%expect_test "every kind survives the seam" =
  List.iter
    (fun (label, w) ->
      Format.printf "%-17s %a@." label
        (Core.Pretty.result ~ok:Fmt.bool ~error:(fun ppf e ->
             Fmt.pf ppf "REJECTED %s" e))
        (round_trip (Err.or_raise ~pp_error:Rsp.Wire.pp_error w)))
    [
      ("progress", Rsp.Wire.of_progress progress);
      ("session", Rsp.Wire.of_final (session "{}"));
      ("delta", Rsp.Wire.of_final (delta "{}"));
      ("failed", Rsp.Wire.of_final failed);
      ("protocol failure", Rsp.Wire.of_final protocol);
    ];
  [%expect
    {|
    progress          true
    session           true
    delta             true
    failed            true
    protocol failure  true |}]

let first_line s =
  match String.index_opt s '\n' with None -> s | Some i -> String.sub s 0 i

let%expect_test "a kind that does not carry its own field" =
  (* The five constructors share one object, so every member is optional on the
     wire and the discriminant is what says which are required. A shell that
     sent a session with no byte count is a defective shell, and this is where
     it is caught. *)
  List.iter
    (fun text ->
      Format.printf "%a@."
        (Core.Pretty.result ~ok:(Fmt.any "accepted") ~error:(fun ppf e ->
             Fmt.pf ppf "REJECTED %s" (first_line e)))
        (Rsp.Meta.decode text))
    [
      {|{"kind":"session","id":"0f8fad5b-d9cb-469f-a165-70867728950e-3","limits":{}}|};
      {|{"kind":"progress","id":"0f8fad5b-d9cb-469f-a165-70867728950e-3"}|};
      {|{"kind":"delta","id":"0f8fad5b-d9cb-469f-a165-70867728950e-3","bytes":2}|};
      {|{"kind":"nonsense"}|};
    ];
  [%expect
    {|
    REJECTED response kind "session" does not carry its own bytes
    REJECTED response kind "progress" does not carry its own done
    REJECTED response kind "delta" does not carry its own key
    REJECTED unknown response kind "nonsense" |}]

(* --- the writer limit --- *)

let%expect_test "the ceiling is a branch, and something can reach it" =
  (* An AGGREGATE bound applied by the writer, not an inference from
     individually-bounded fields: JSON escaping expands and three bounded fields
     still sum. A function whose implementation can fail says so regardless of
     which inputs reach it today, which is why this is result-valued -- and why
     the ceiling is injectable, since a bound nobody can drive is a bound taken
     on trust. *)
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:(Fmt.any "encoded") ~error:Rsp.Wire.pp_error)
    (Rsp.Wire.of_final_bounded ~max_meta_bytes:16 (session "{}"));
  [%expect {| response metadata is over the ceiling |}]

let%expect_test "the terminal response is inside the ceiling it must meet" =
  (* Asserted over THESE bytes -- the ones that path actually posts -- because
     the fallback does not run the encoder at all. "It cannot itself fail" is a
     property of a value here, not an argument about a function. *)
  let w = Rsp.Wire.constant_protocol_failure in
  Printf.printf "%d <= %d, payload=%b\n"
    (String.length w.Rsp.Wire.meta)
    Me_limits.Hard.max_response_meta_bytes
    (Option.is_some w.Rsp.Wire.payload);
  print_endline w.Rsp.Wire.meta;
  [%expect
    {|
    126 <= 1048576, payload=false
    {"kind":"protocol_failure","error":{"code":"internal","message":"the worker could not answer this request","truncated":false}} |}]

(* --- the narrowing --- *)

let%expect_test "a handler's outcomes are three, and the shell's are four" =
  (* [Handle_result.t] is narrower than [Final.t] by exactly one constructor,
     and that constructor is the point: a protocol failure means the request was
     never understood, so a function that received a VALIDATED request cannot
     honestly produce one. [of_handle_result] is the only way across, and it is
     total -- there is no argument it could give the fourth constructor. *)
  List.iter
    (fun r ->
      Format.printf "%s@."
        (match Rsp.Final.of_handle_result r with
        | Rsp.Final.Session _ -> "session"
        | Rsp.Final.Delta _ -> "delta"
        | Rsp.Final.Failed _ -> "failed"
        | Rsp.Final.Protocol_failure _ -> "protocol_failure"))
    [
      Rsp.Handle_result.Session { Rsp.Session.id; limits = wire; json = "{}" };
      Rsp.Handle_result.Delta { Rsp.Delta.id; key; json = "{}" };
      Rsp.Handle_result.Failed { Rsp.Failed.id; key = None; error = diagnostic };
    ];
  [%expect {|
    session
    delta
    failed |}]
