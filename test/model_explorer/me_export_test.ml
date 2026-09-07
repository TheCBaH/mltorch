(* [Me_export]: model bytes in, session out — and the worker entry point
   (.ai/model_explorer_design.md).

   The session builder lives in the LIBRARY, not in the CLI, because the browser
   compiles this library rather than a second implementation of it. This suite
   is what makes that claim testable without a browser: it drives [handle], the
   worker's own entry point, over bytes it builds itself. *)

module MR = Me_request
module Rsp = Me_response
module L = Me_limits.Limits

let limits = L.untrusted

let wire =
  Err.or_raise ~pp_error:Me_limits.pp_error
    (Me_limits.Wire_limits.of_limits ~ceiling:L.untrusted L.untrusted)

let uuid = "0f8fad5b-d9cb-469f-a165-70867728950e"

let id =
  Err.or_raise ~pp_error:MR.Request.pp_error
    (MR.Request_id.of_string (uuid ^ "-1"))

(* A whole ExportedProgram, small enough to read. [target] is the one knob:
   an operator the lowerer knows and one it does not are the two rows §5.4 calls
   conditional, and both have to reach a SESSION rather than an error. *)
let model ?(target = "torch.ops.aten.relu.default") () =
  Printf.sprintf
    {|{"graph_module":{"graph":{
        "inputs":[{"as_tensor":{"name":"x"}}],
        "outputs":[{"as_tensor":{"name":"y"}}],
        "nodes":[{"target":"%s",
                  "inputs":[{"name":"self","arg":{"as_tensor":{"name":"x"}},"kind":1}],
                  "outputs":[{"as_tensor":{"name":"y"}}],
                  "metadata":{}}],
        "tensor_values":{"x":%s,"y":%s},
        "sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},
      "signature":{"input_specs":[{"user_input":{"arg":{"as_tensor":{"name":"x"}}}}],
                   "output_specs":[{"user_output":{"arg":{"as_tensor":{"name":"y"}}}}]},
      "module_call_graph":[]},
      "opset_version":{"aten":15},"range_constraints":{},
      "schema_version":{"major":8,"minor":5}}|}
    target
    {|{"dtype":7,"sizes":[{"as_int":1},{"as_int":4}],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":4},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    {|{"dtype":7,"sizes":[{"as_int":1},{"as_int":4}],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":4},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}

(* --- detection --- *)

let%expect_test "content decides, not a declared extension" =
  List.iter
    (fun (label, bytes) ->
      Format.printf "%-14s %a@." label
        (Core.Pretty.err_result
           ~ok:(fun ppf -> function
             | Me_export.Model_json -> Fmt.string ppf "model.json"
             | Me_export.Pt2_archive -> Fmt.string ppf "pt2")
           ~error:Me_export.pp_error)
        (Me_export.detect ~bytes))
    [
      ("json", "{\"a\":1}");
      ("zip", "PK\003\004rest of an archive");
      ("neither", "not a model");
      ("empty", "");
    ];
  [%expect
    {|
    json           model.json
    zip            pt2
    neither        not a zip archive and not a JSON object
    empty          not a zip archive and not a JSON object |}]

(* --- the worker entry point --- *)

let source ?(format = `Model_json) ?(name = "tiny") bytes =
  Err.or_raise ~pp_error:MR.Request.pp_error
    (MR.Source.create ~limits ~origin:MR.Source.Origin.Local ~name
       ~bytes:(Int64.of_int (String.length bytes))
       ~format)

let options =
  Err.or_raise ~pp_error:MR.Request.pp_error
    (MR.Options.create
       ~stages:[ Me_session.Capability.Canonical ]
       ~fold:false ~verify_symbolic:None ~namespace:MR.Options.Structural)

let run ?format ?(wire = wire) ?(options = options) ?(bytes = model ()) () =
  let seen = ref [] in
  let emit (p : Rsp.Progress.t) =
    seen := Rsp.Phase.to_string p.Rsp.Progress.phase :: !seen
  in
  let request =
    Err.or_raise ~pp_error:MR.Request.pp_error
      (MR.Request.build_session ~id ~source:(source ?format bytes) ~options
         ~limits:wire)
  in
  (* [let], not a tuple: OCaml evaluates tuple components right to left, so
     [(handle ..., List.rev !seen)] reads the accumulator BEFORE the call that
     fills it -- and reports no phases at all, which is what this test first
     showed. *)
  let result = Me_export.handle ~emit request ~bytes in
  (result, List.rev !seen)

(* The METADATA, because that is the only thing that crosses; the document
   itself is summarised, since a golden holding it would be re-promoted on every
   unrelated projection tweak. *)
let show (result, phases) =
  Printf.printf "phases: %s\n" (String.concat " " phases);
  match Rsp.Wire.of_final (Rsp.Final.of_handle_result result) with
  | Error e -> Format.printf "%a@." Rsp.Wire.pp_error (Err.Error.kind e)
  | Ok w ->
      Format.printf "%a@."
        (Core.Pretty.err_result
           ~ok:(fun ppf (m : Rsp.Meta.t) ->
             Fmt.pf ppf "%s payload=%a" (Rsp.Meta.kind m)
               (Core.Pretty.option_or ~none:"none" (fun ppf s ->
                    Fmt.pf ppf "%d bytes" (String.length s)))
               w.Rsp.Wire.payload)
           ~error:Rsp.Meta.pp_error)
        (Rsp.Meta.decode w.Rsp.Wire.meta);
      Format.printf "%s@." w.Rsp.Wire.meta

let%expect_test "a model the lowerer handles" =
  (* This one reaches the four-axis branch too: a 1x4 relu is inside the
     dialect, which is what makes the payload larger than the graphs alone
     account for and what covers [Me_native4d] end to end. *)
  show (run ());
  [%expect
    {|
    phases: decode encode
    session payload=6873 bytes
    {"kind":"session","id":"0f8fad5b-d9cb-469f-a165-70867728950e-1","limits":{},"bytes":6873} |}]

let%expect_test "a model the lowerer does NOT handle is still a session" =
  (* The one row that makes this a capability protocol rather than an error
     protocol: the exported program decoded, which is the whole of what
     [Graph_stage Source] claims. *)
  let result, _ =
    run ~bytes:(model ~target:"torch.ops.aten.bogus.default" ()) ()
  in
  (match result with
  | Rsp.Handle_result.Session s ->
      Format.printf "%a@."
        (Core.Pretty.result
           ~ok:(fun ppf (d : Me_session.Session.t) ->
             Fmt.list ~sep:(Fmt.any "@\n")
               (fun ppf (c : Me_session.Capability.t) ->
                 Fmt.pf ppf "%-28s %s"
                   (Me_session.Capability.key_name c.Me_session.Capability.key)
                   (match c.Me_session.Capability.status with
                   | Me_session.Capability.Available _ -> "available"
                   | Me_session.Capability.Not_requested -> "not_requested"
                   | Me_session.Capability.Unavailable { reason; _ } ->
                       Core.Pretty.to_string Me_classify.pp_verdict
                         (Me_classify.Unavailable reason)))
               ppf d.Me_session.Session.capabilities)
           ~error:Fmt.string)
        (Jsont_bytesrw.decode_string Me_session.Session.jsont s.Rsp.Session.json)
  | _ -> print_endline "not a session");
  [%expect
    {|
    stage:source                 available
    stage:initial_native         unavailable unsupported_operator
    stage:canonical              unavailable prerequisite_unavailable
    stage:native4d               not_requested
    stage:stage_program          not_requested
    stage:kernel                 not_requested
    stage:fusion                 not_requested
    feature:flow                 unavailable prerequisite_unavailable
    feature:verification         unavailable prerequisite_unavailable
    feature:pass_audits          unavailable prerequisite_unavailable
    feature:fold                 unavailable prerequisite_unavailable
    feature:expression_detail    available
    feature:loop_ir              unavailable not_implemented
    feature:codegen              unavailable not_implemented |}]

let%expect_test "which stages a tiny in-dialect model reaches" =
  let result, _ = run () in
  (match result with
  | Rsp.Handle_result.Session s ->
      Format.printf "%a@."
        (Core.Pretty.result
           ~ok:(fun ppf (d : Me_session.Session.t) ->
             Fmt.pf ppf "%s"
               (String.concat " "
                  (List.filter_map
                     (fun (c : Me_session.Capability.t) ->
                       match c.Me_session.Capability.status with
                       | Me_session.Capability.Available _ ->
                           Some
                             (Me_session.Capability.key_name
                                c.Me_session.Capability.key)
                       | _ -> None)
                     d.Me_session.Session.capabilities)))
           ~error:Fmt.string)
        (Jsont_bytesrw.decode_string Me_session.Session.jsont s.Rsp.Session.json)
  | _ -> print_endline "not a session");
  [%expect
    {| stage:source stage:initial_native stage:canonical feature:flow feature:expression_detail |}]

let%expect_test
    "requesting Kernel alone gets Kernel, not its unrequested prerequisite" =
  (* [Kernel] forces [Stage_program] to be COMPUTED -- there is no kernel
     without a stage program to adapt -- but computed is not requested: the
     capability row for [Stage_program] must still read [not_requested], and
     [Native4d] (not on the path to [Kernel] at all) must too. If either
     ever showed [available] here it would mean this test's own model quietly
     grew a second reason to reach that row, not that the option was
     ignored -- the request-options test below is the one that would catch
     that directly. *)
  let kernel_options =
    Err.or_raise ~pp_error:MR.Request.pp_error
      (MR.Options.create
         ~stages:[ Me_session.Capability.Kernel ]
         ~fold:false ~verify_symbolic:None ~namespace:MR.Options.Structural)
  in
  let result, _ = run ~options:kernel_options () in
  (match result with
  | Rsp.Handle_result.Session s ->
      Format.printf "%a@."
        (Core.Pretty.result
           ~ok:(fun ppf (d : Me_session.Session.t) ->
             Fmt.list ~sep:(Fmt.any "@\n")
               (fun ppf (c : Me_session.Capability.t) ->
                 Fmt.pf ppf "%-28s %s"
                   (Me_session.Capability.key_name c.Me_session.Capability.key)
                   (match c.Me_session.Capability.status with
                   | Me_session.Capability.Available _ -> "available"
                   | Me_session.Capability.Not_requested -> "not_requested"
                   | Me_session.Capability.Unavailable { reason; _ } ->
                       Core.Pretty.to_string Me_classify.pp_verdict
                         (Me_classify.Unavailable reason)))
               ppf d.Me_session.Session.capabilities)
           ~error:Fmt.string)
        (Jsont_bytesrw.decode_string Me_session.Session.jsont s.Rsp.Session.json)
  | _ -> print_endline "not a session");
  [%expect
    {|
    stage:source                 available
    stage:initial_native         available
    stage:canonical              available
    stage:native4d               not_requested
    stage:stage_program          not_requested
    stage:kernel                 available
    stage:fusion                 not_requested
    feature:flow                 available
    feature:verification         not_requested
    feature:pass_audits          not_requested
    feature:fold                 unavailable requires_payloads
    feature:expression_detail    available
    feature:loop_ir              unavailable not_implemented
    feature:codegen              unavailable not_implemented |}]

(* --- the detail path --- *)

let%expect_test "a detail request answers with a delta, not a session" =
  (* The worker holds no session between requests, so this re-lowers -- which is
     what "self-contained" means and why there is nothing that could be stale.
     It runs a SMALLER pipeline than a session request: a detail needs the
     kernel and nothing else. *)
  let bytes = model () in
  let key =
    Err.or_raise ~pp_error:MR.Request.pp_error
      (MR.Detail_key.create ~limits
         ~parent_graph:(Me_ids.graph Me_ids.Layer.Kernel 0)
         ~value:(Graph_ir.Tensor_id.of_int 1))
  in
  let request =
    Err.or_raise ~pp_error:MR.Request.pp_error
      (MR.Request.build_detail ~id ~source:(source bytes) ~options ~limits:wire
         ~key)
  in
  let result = Me_export.handle ~emit:(fun _ -> ()) request ~bytes in
  show (result, []);
  [%expect
    {|
    phases:
    delta payload=6752 bytes
    {"kind":"delta","id":"0f8fad5b-d9cb-469f-a165-70867728950e-1","key":{"kind":"value","parentGraph":"g/kernel/000","value":1},"bytes":6752} |}]

let%expect_test "an operator detail derives its canonical outputs" =
  let bytes = model () in
  let key =
    Err.or_raise ~pp_error:MR.Request.pp_error
      (MR.Detail_key.create_operator ~limits ~parent_graph:"g/native/001"
         ~node:(Graph_ir.Node_id.of_int 0))
  in
  let request =
    Err.or_raise ~pp_error:MR.Request.pp_error
      (MR.Request.build_detail ~id ~source:(source bytes) ~options ~limits:wire
         ~key)
  in
  show (Me_export.handle ~emit:(fun _ -> ()) request ~bytes, []);
  [%expect
    {|
    phases:
    delta payload=7565 bytes
    {"kind":"delta","id":"0f8fad5b-d9cb-469f-a165-70867728950e-1","key":{"kind":"operator","parentGraph":"g/native/001","node":0},"bytes":7565} |}]

let%expect_test "a detail request for a value the model does not produce" =
  (* A valid request about something ABSENT, which is a different fact from a
     malformed key and carries its own code. *)
  let bytes = model () in
  let key =
    Err.or_raise ~pp_error:MR.Request.pp_error
      (MR.Detail_key.create ~limits
         ~parent_graph:(Me_ids.graph Me_ids.Layer.Kernel 0)
         ~value:(Graph_ir.Tensor_id.of_int 9999))
  in
  let request =
    Err.or_raise ~pp_error:MR.Request.pp_error
      (MR.Request.build_detail ~id ~source:(source bytes) ~options ~limits:wire
         ~key)
  in
  show (Me_export.handle ~emit:(fun _ -> ()) request ~bytes, []);
  [%expect
    {|
    phases:
    failed payload=none
    {"kind":"failed","id":"0f8fad5b-d9cb-469f-a165-70867728950e-1","key":{"kind":"value","parentGraph":"g/kernel/000","value":9999},"error":{"code":"unsupported_detail_key","message":"the key names no value this model produces","truncated":false}} |}]

let%expect_test "an encoded delta over max_detail_bytes is refused" =
  (* Same producer-side proof as the session case, for the other encoder:
     [max_detail_nodes] stays generous (that ceiling is [Me_detail.apply]'s
     own, checked value-level, and is not what this test is about), only
     [max_detail_bytes] is tightened below the ~1229-byte delta this same
     model and value produce elsewhere in this suite. *)
  let bytes = model () in
  let key =
    Err.or_raise ~pp_error:MR.Request.pp_error
      (MR.Detail_key.create ~limits
         ~parent_graph:(Me_ids.graph Me_ids.Layer.Kernel 0)
         ~value:(Graph_ir.Tensor_id.of_int 1))
  in
  let tiny_detail =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (L.create ~max_detail_bytes:256 L.untrusted)
  in
  let tiny_wire =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Wire_limits.of_limits ~ceiling:L.untrusted tiny_detail)
  in
  let request =
    Err.or_raise ~pp_error:MR.Request.pp_error
      (MR.Request.build_detail ~id ~source:(source bytes) ~options
         ~limits:tiny_wire ~key)
  in
  show (Me_export.handle ~emit:(fun _ -> ()) request ~bytes, []);
  [%expect
    {|
    phases:
    failed payload=none
    {"kind":"failed","id":"0f8fad5b-d9cb-469f-a165-70867728950e-1","key":{"kind":"value","parentGraph":"g/kernel/000","value":1},"error":{"code":"over_limit","message":"the encoded document is over the ceiling","truncated":false}} |}]

let%expect_test "bytes that are not a model at all" =
  show (run ~bytes:"not a model" ());
  [%expect
    {|
    phases: decode
    failed payload=none
    {"kind":"failed","id":"0f8fad5b-d9cb-469f-a165-70867728950e-1","error":{"code":"invalid_source","message":"not a zip archive and not a JSON object","truncated":false}} |}]

let%expect_test "a declared format the bytes are not" =
  (* DECLARED is a hint; the content decides. A disagreement is a fact about the
     source, so it is [Invalid_source] rather than an internal error. *)
  show (run ~format:`Pt2 ());
  [%expect
    {|
    phases: decode
    failed payload=none
    {"kind":"failed","id":"0f8fad5b-d9cb-469f-a165-70867728950e-1","error":{"code":"invalid_source","message":"the bytes are not the format the request declared","truncated":false}} |}]

let%expect_test "the JSON ceiling applies to a JSON model, not the pt2 one" =
  (* [max_pt2_bytes] stays at [untrusted]'s (huge) default; only
     [max_json_bytes] is tightened below [model ()]'s length. If the size
     check fell back to comparing every format against [max_pt2_bytes] --
     which it did before this test existed -- this model would be accepted. *)
  let tiny_json =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (L.create ~max_json_bytes:64L L.untrusted)
  in
  let tiny_wire =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Wire_limits.of_limits ~ceiling:L.untrusted tiny_json)
  in
  show (run ~wire:tiny_wire ());
  [%expect
    {|
    phases: decode
    failed payload=none
    {"kind":"failed","id":"0f8fad5b-d9cb-469f-a165-70867728950e-1","error":{"code":"over_limit","message":"1133 bytes is over the ceiling","truncated":false}} |}]

let%expect_test
    "an encoded session over max_session_bytes is refused, not truncated" =
  (* Producer-side enforcement: the model itself is well inside every input
     ceiling (it lowers and encodes to ~8786 bytes elsewhere in this suite),
     so only [max_session_bytes] being tightened below that makes this fail --
     proving the writer bound is what caught it, not one of the input checks
     above it. *)
  let tiny_session =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (L.create ~max_session_bytes:256 L.untrusted)
  in
  let tiny_wire =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Wire_limits.of_limits ~ceiling:L.untrusted tiny_session)
  in
  show (run ~wire:tiny_wire ());
  [%expect
    {|
    phases: decode encode
    failed payload=none
    {"kind":"failed","id":"0f8fad5b-d9cb-469f-a165-70867728950e-1","error":{"code":"over_limit","message":"the encoded document is over the ceiling","truncated":false}} |}]
