(* Section 7: the whole worker path, request in and response metadata out.

   This is the section that makes the "one implementation" claim checkable. The
   browser runs [Me_export.handle] compiled by js_of_ocaml; the CLI runs the
   same function compiled by ocamlopt. Everything between the request bytes and
   the response metadata — the profile codec, the staged request decoder, the
   projection, the session codec, the response envelope — is pure OCaml in one
   library, so the two backends must agree BYTE FOR BYTE, and this is where
   that is asserted rather than assumed.

   In-memory throughout, because that is what a worker has: the request arrives
   as JSON bytes and the model as a transferred buffer, and neither is a path.

   The document is SUMMARISED. A megabyte of session JSON in the diff would
   drown every other section and be re-promoted without being read; the summary
   still moves on any divergence, since the metadata carries a byte count
   measured from the document itself. *)

let read_file path =
  try In_channel.with_open_bin path In_channel.input_all
  with Sys_error msg -> failwith ("probe: cannot read " ^ path ^ ": " ^ msg)

let ok what pp r =
  Err.or_raise ~pp_error:(fun ppf e -> Fmt.pf ppf "probe: %s: %a" what pp e) r

(* Plain [result] from Jsont, so [failwith]. And it must RAISE rather than
   print: both backends run this source, so a failure identical on each would
   diff clean and exit 0. *)
let jsont what = function
  | Ok v -> v
  | Error e -> failwith (Printf.sprintf "probe: %s: %s" what e)

(* A FIXED request, spelled out as the bytes a worker would receive. Building it
   through the constructors would test the constructors; decoding it tests the
   staged decoder, which is the half a browser actually exercises. *)
let request_json ~name ~bytes =
  Printf.sprintf {|{"id":"0f8fad5b-d9cb-469f-a165-70867728950e-1",|}
  ^ Printf.sprintf {|"limits":{"max_label_bytes":"128"},|}
  ^ Printf.sprintf
      {|"options":{"stages":["canonical"],"fold":false,"namespace":"structural"},|}
  ^ Printf.sprintf
      {|"source":{"name":"%s","bytes":"%d","format":"model_json"}}|} name bytes

let run path =
  print_endline "=== worker ===";
  let bytes = read_file path in
  let request =
    jsont "request decode"
      (Jsont_bytesrw.decode_string Me_request.Request.jsont
         (request_json ~name:"resnet18" ~bytes:(String.length bytes)))
  in
  Printf.printf "epoch %s\n" (Me_request.Request.epoch request);
  Printf.printf "profile-overrides %s\n"
    (jsont "profile encode"
       (Jsont_bytesrw.encode_string Me_limits.Wire_limits.jsont
          (Me_request.Request.limits request)));
  let phases = ref [] in
  let emit (p : Me_response.Progress.t) =
    phases :=
      Me_response.Phase.to_string p.Me_response.Progress.phase :: !phases
  in
  let result = Me_export.handle ~emit request ~bytes in
  Printf.printf "phases %s\n" (String.concat "," (List.rev !phases));
  let wire =
    ok "wire" Me_response.Wire.pp_error
      (Me_response.Wire.of_final (Me_response.Final.of_handle_result result))
  in
  let meta =
    jsont "meta decode" (Me_response.Meta.decode wire.Me_response.Wire.meta)
  in
  Printf.printf "kind %s\n" (Me_response.Meta.kind meta);
  Printf.printf "meta-bytes %d\n" (String.length wire.Me_response.Wire.meta);
  Printf.printf "payload-bytes %d\n"
    (Option.fold ~none:0 ~some:String.length wire.Me_response.Wire.payload);
  (* The document's own shape, decoded back through the session codec: a byte
     count alone would move for a whitespace reason, and this moves for a
     PROJECTION reason. *)
  let session =
    jsont "session decode"
      (Jsont_bytesrw.decode_string Me_session.Session.jsont
         (Option.get wire.Me_response.Wire.payload))
  in
  let s = session.Me_session.Session.model in
  Printf.printf "model %s %Ld bytes, %d op targets\n"
    s.Me_session.Model_summary.name s.Me_session.Model_summary.source_bytes
    s.Me_session.Model_summary.op_targets;
  List.iter
    (fun (c : Model_explorer.GraphCollection.t) ->
      Printf.printf "collection %s\n" c.Model_explorer.GraphCollection.label;
      List.iter
        (fun (g : Model_explorer.Graph.t) ->
          Printf.printf "  graph %s %d nodes\n" g.Model_explorer.Graph.id
            (List.length g.Model_explorer.Graph.nodes))
        c.Model_explorer.GraphCollection.graphs)
    session.Me_session.Session.graph_collections;
  List.iter
    (fun (c : Me_session.Capability.t) ->
      Printf.printf "  %-28s %s\n"
        (Me_session.Capability.key_name c.Me_session.Capability.key)
        (match c.Me_session.Capability.status with
        | Me_session.Capability.Available _ -> "available"
        | Me_session.Capability.Not_requested -> "not_requested"
        | Me_session.Capability.Unavailable { reason; _ } ->
            Core.Pretty.to_string Me_classify.pp_verdict
              (Me_classify.Unavailable reason)))
    session.Me_session.Session.capabilities;
  (* The one thing only a failure path can show, and the reason it is here: a
     model the worker cannot READ still produces a well-formed response, with a
     code from the closed vocabulary rather than an exception escaping into the
     shell. *)
  let failure =
    Me_export.handle ~emit:(fun _ -> ()) request ~bytes:"not a model"
  in
  let wire =
    ok "wire" Me_response.Wire.pp_error
      (Me_response.Wire.of_final (Me_response.Final.of_handle_result failure))
  in
  print_endline wire.Me_response.Wire.meta
