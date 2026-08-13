open Js_of_ocaml
module Buffer = Webapp_jsoo_buffer

let get o n = Js.Unsafe.get o n
let string o n = Js.to_string (Js.Unsafe.coerce (get o n))

let response ?request ?render ?token ?error ok =
  let fields = [ ("ok", Js.Unsafe.inject (Js.bool ok)) ] in
  let add name value fields = (name, Js.Unsafe.inject value) :: fields in
  let fields =
    match request with None -> fields | Some v -> add "request" v fields
  in
  let fields =
    match render with
    | None -> fields
    | Some v -> add "render" (Js.string v) fields
  in
  let fields =
    match token with
    | None -> fields
    | Some v -> add "token" (Js.string v) fields
  in
  let fields =
    match error with
    | None -> fields
    | Some v -> add "error" (Js.string v) fields
  in
  Js.Unsafe.obj (Array.of_list fields)

let build_session raw =
  try
    let id =
      Err.or_raise
        ~pp_error:(fun _ _ -> ())
        (Me_request.Request_id.of_string (string raw "id"))
    in
    let source_raw = get raw "source" in
    let bytes = Int64.of_string (string source_raw "bytes") in
    let origin =
      if Js.to_string (Js.Unsafe.coerce (get source_raw "kind")) = "catalog"
      then
        let catalog = get source_raw "catalog" in
        Me_request.Source.Origin.Catalog
          {
            url = string catalog "url";
            ref_ = string catalog "ref";
            verified_sha256 = string catalog "verifiedSha256";
          }
      else Me_request.Source.Origin.Local
    in
    let source =
      Err.or_raise
        ~pp_error:(fun _ _ -> ())
        (Me_request.Source.create ~limits:Me_limits.Limits.untrusted ~origin
           ~name:(string source_raw "name") ~bytes ~format:`Model_json)
    in
    let options =
      Err.or_raise
        ~pp_error:(fun _ _ -> ())
        (Me_request.Options.create ~stages:Me_session.Capability.all_stages
           ~fold:false ~verify_symbolic:None
           ~namespace:Me_request.Options.Structural)
    in
    let limits =
      Err.or_raise
        ~pp_error:(fun _ _ -> ())
        (Me_limits.Wire_limits.of_limits ~ceiling:Me_limits.Limits.untrusted
           Me_limits.Limits.untrusted)
    in
    let request =
      Err.or_raise
        ~pp_error:(fun _ _ -> ())
        (Me_request.Request.build_session ~id ~source ~options ~limits)
    in
    match Jsont_bytesrw.encode_string Me_request.Request.jsont request with
    | Ok json -> response true ~request:(Buffer.fresh_array_buffer json)
    | Error message -> response false ~error:message
  with exn -> response false ~error:(Printexc.to_string exn)

let staged = ref None

let prepare expected_id meta_text document =
  try
    let expected =
      Err.or_raise
        ~pp_error:(fun _ _ -> ())
        (Me_request.Request_id.of_string expected_id)
    in
    let meta =
      Err.or_raise ~pp_error:Me_response.Meta.pp_error
        (Me_response.Meta.decode meta_text)
    in
    let limits =
      match meta with
      | Me_response.Meta.Session s
        when Me_request.Request_id.equal expected s.id
             && s.bytes = String.length document ->
          Me_limits.Wire_limits.limits s.limits
      | _ -> raise Exit
    in
    let session =
      match Jsont_bytesrw.decode_string Me_session.Session.jsont document with
      | Ok s -> s
      | Error _ -> raise Exit
    in
    Err.or_raise ~pp_error:Me_session.Session.pp_error
      (Me_session.Session.validate ~limits session);
    match Jsont_bytesrw.encode_string Me_session.Session.jsont session with
    | Ok render ->
        staged := Some expected_id;
        response true ~token:expected_id ~render
    | Error message -> response false ~error:message
  with exn -> response false ~error:(Printexc.to_string exn)

let commit token =
  if !staged = Some token then (
    staged := None;
    response true)
  else response false ~error:"invalid session token"

let abort token =
  if !staged = Some token then staged := None;
  response true

let reset () =
  staged := None;
  response true

let () =
  let hard =
    Js.Unsafe.obj
      [|
        ( "maxRequestJsonBytes",
          Js.Unsafe.inject Me_limits.Hard.max_request_json_bytes );
        ( "maxResponseMetaBytes",
          Js.Unsafe.inject Me_limits.Hard.max_response_meta_bytes );
        ( "maxResponseDocumentBytes",
          Js.Unsafe.inject Me_limits.Hard.max_response_document_bytes );
        ( "maxJsonBytes",
          Js.Unsafe.inject
            (Js.string (Int64.to_string Me_limits.Hard.max_json_bytes)) );
        ("maxRestoreSteps", Js.Unsafe.inject Me_limits.Hard.max_restore_steps);
      |]
  in
  let request =
    Js.Unsafe.obj
      [| ("buildSession", Js.Unsafe.inject (Js.wrap_callback build_session)) |]
  in
  let session =
    Js.Unsafe.obj
      [|
        ( "prepare",
          Js.Unsafe.inject
            (Js.wrap_callback (fun id meta doc ->
                 prepare (Js.to_string id) (Js.to_string meta)
                   (Js.to_string doc))) );
        ( "commit",
          Js.Unsafe.inject
            (Js.wrap_callback (fun token -> commit (Js.to_string token))) );
        ( "abort",
          Js.Unsafe.inject
            (Js.wrap_callback (fun token -> abort (Js.to_string token))) );
        ("reset", Js.Unsafe.inject (Js.wrap_callback reset));
      |]
  in
  Js.export "mltorch"
    (Js.Unsafe.obj
       [|
         ("hard", Js.Unsafe.inject hard);
         ("request", Js.Unsafe.inject request);
         ("session", Js.Unsafe.inject session);
       |])
