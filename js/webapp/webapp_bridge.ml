open Js_of_ocaml
module Buffer = Webapp_jsoo_buffer
module C = Me_session.Capability

(* Reading an untrusted JavaScript value is not a coercion.

   [Js.Unsafe.get] hands back [undefined] for an absent field, [Js.to_string]
   renders a number or a null as whatever the runtime makes of it, and
   [typeof null] is ["object"] -- so before this module every malformed request
   arrived either as a silently coerced value or as a [Failure] printed through
   [Printexc.to_string], and neither names the field at fault. Every field below
   is checked with [Js.typeof] first and every rejection is DATA, in one closed
   row with one printer.

   The precedent is next door: [Webapp_jsoo_buffer.array_buffer_length] checks
   with [Js.instanceof] and returns [`Not_an_array_buffer] rather than trusting
   its argument. *)
module Js_read = struct
  type error =
    [ `Not_an_object of string
    | `Not_a_string of string
    | `Not_a_bool of string
    | `Not_an_array of string
    | `Not_a_string_element of string
    | `Not_a_decimal of string
    | `Not_an_integer of string
    | `Unknown_stage of string
    | `Unknown_effort of string
    | `Unknown_source_kind of string ]

  (* Closed, not [[< error ]]: this domain is private to the bridge and every
     reader below returns exactly it, so there is no row to keep open. *)
  let pp_error fmt (e : error) =
    match e with
    | `Not_an_object field -> Format.fprintf fmt "%s must be an object" field
    | `Not_a_string field -> Format.fprintf fmt "%s must be a string" field
    | `Not_a_bool field -> Format.fprintf fmt "%s must be a boolean" field
    | `Not_an_array field -> Format.fprintf fmt "%s must be an array" field
    | `Not_a_string_element field ->
        Format.fprintf fmt "every %s entry must be a string" field
    | `Not_a_decimal field ->
        Format.fprintf fmt "%s must be a decimal string" field
    | `Not_an_integer field -> Format.fprintf fmt "%s must be an integer" field
    | `Unknown_stage s -> Format.fprintf fmt "unknown output stage %S" s
    | `Unknown_effort s -> Format.fprintf fmt "unknown verification effort %S" s
    | `Unknown_source_kind s -> Format.fprintf fmt "unknown source kind %S" s

  let fail e = Err.fail ~pp_error e

  (* Concrete [any] rather than [Js.Unsafe.get]'s free result type, so every
     reader below shares one argument type and no call site can silently pick
     a convenient one. *)
  let field o name : Js.Unsafe.any = Js.Unsafe.get o name
  let typeof v = Js.to_string (Js.typeof v)
  let absent v = String.equal (typeof v) "undefined"

  (* [Js.Unsafe.coerce] is [_ t -> _ t] and so cannot reach [Js.opt]; both are
     [%identity], so the cast to it is declared here rather than worked around.
     [Js.Opt.test] is [x !== null] and therefore PASSES [undefined], which is
     why every caller checks {!absent} first. *)
  external as_opt : Js.Unsafe.any -> 'a Js.Opt.t = "%identity"

  let is_null v = not (Js.Opt.test (as_opt v))

  (* The null check is not optional. [typeof null] is ["object"], so without it
     a null [options] would read as an object whose every field is absent, and
     a caller that meant to send nothing would silently get the DEFAULT
     request rather than a rejection. *)
  let is_object v = String.equal (typeof v) "object" && not (is_null v)

  let is_array v =
    Js.to_bool
      (Js.Unsafe.meth_call
         (Js.Unsafe.get Js.Unsafe.global "Array")
         "isArray"
         [| Js.Unsafe.inject v |])

  (* [label] is the dotted path shown to the reader, which is not always the
     key: ["url"] alone would not say which object was malformed. *)
  let string_field ?label o name =
    let label = match label with Some l -> l | None -> name in
    let v = field o name in
    if String.equal (typeof v) "string" then
      Err.return (Js.to_string (Js.Unsafe.coerce v))
    else fail (`Not_a_string label)

  let bool_field o name ~default =
    let v = field o name in
    if absent v then Err.return default
    else if String.equal (typeof v) "boolean" then
      Err.return (Js.to_bool (Js.Unsafe.coerce v))
    else fail (`Not_a_bool name)

  let string_list o name =
    let v = field o name in
    if not (is_array v) then fail (`Not_an_array name)
    else
      let rec go acc = function
        | [] -> Err.return (List.rev acc)
        | item :: rest ->
            if String.equal (typeof item) "string" then
              go (Js.to_string (Js.Unsafe.coerce item) :: acc) rest
            else fail (`Not_a_string_element name)
      in
      go [] (Array.to_list (Js.to_array (Js.Unsafe.coerce v)))

  let int_field ?label o name =
    let label = match label with Some l -> l | None -> name in
    let v = field o name in
    if String.equal (typeof v) "number" then
      let f = Js.float_of_number (Js.Unsafe.coerce v) in
      if
        Float.is_finite f
        && Float.floor f = f
        && f >= float_of_int min_int
        && f <= float_of_int max_int
      then Err.return (int_of_float f)
      else fail (`Not_an_integer label)
    else fail (`Not_an_integer label)
end

open Err.Syntax

(* --- the request options ------------------------------------------------- *)

let stage_of_name name =
  List.find_opt (fun s -> String.equal (C.stage_name s) name) C.all_stages

let read_stages options_raw =
  if Js_read.absent (Js_read.field options_raw "stages") then
    (* The browser's own default is every stage, which is what this bridge
       requested unconditionally before options existed. Keeping it here means
       a caller that sends no [stages] gets exactly today's request. *)
    Err.return C.all_stages
  else
    let* names = Js_read.string_list options_raw "stages" in
    let rec go acc = function
      | [] -> Err.return (List.rev acc)
      | name :: rest -> (
          match stage_of_name name with
          | Some stage -> go (stage :: acc) rest
          | None -> Js_read.fail (`Unknown_stage name))
    in
    go [] names

let read_effort options_raw =
  let v = Js_read.field options_raw "verifySymbolic" in
  if Js_read.absent v || Js_read.is_null v then Err.return None
  else if String.equal (Js_read.typeof v) "string" then
    let name = Js.to_string (Js.Unsafe.coerce v) in
    match Map_verify.Effort.of_string name with
    | Ok effort -> Err.return (Some effort)
    | Error _ -> Js_read.fail (`Unknown_effort name)
  else Js_read.fail (`Not_a_string "verifySymbolic")

(* An absent [options] cannot fall through to the field readers: reading a
   property off [undefined] throws in JavaScript rather than yielding
   [undefined] again. *)
let read_options raw =
  let v = Js_read.field raw "options" in
  if Js_read.absent v then Err.return (C.all_stages, false, None)
  else if not (Js_read.is_object v) then Js_read.fail (`Not_an_object "options")
  else
    let* stages = read_stages v in
    let* fold = Js_read.bool_field v "fold" ~default:false in
    let+ effort = read_effort v in
    (stages, fold, effort)

(* Every non-["catalog"] value used to be read as [Local], so a typo in the
   kind produced a request with no provenance instead of a rejection. *)
let read_origin source_raw =
  let* kind = Js_read.string_field source_raw "kind" ~label:"source.kind" in
  match kind with
  | "local" -> Err.return Me_request.Source.Origin.Local
  | "catalog" ->
      let catalog = Js_read.field source_raw "catalog" in
      if not (Js_read.is_object catalog) then
        Js_read.fail (`Not_an_object "source.catalog")
      else
        let* url =
          Js_read.string_field catalog "url" ~label:"source.catalog.url"
        in
        let* ref_ =
          Js_read.string_field catalog "ref" ~label:"source.catalog.ref"
        in
        let+ verified_sha256 =
          Js_read.string_field catalog "verifiedSha256"
            ~label:"source.catalog.verifiedSha256"
        in
        Me_request.Source.Origin.Catalog { url; ref_; verified_sha256 }
  | other -> Js_read.fail (`Unknown_source_kind other)

let read_bytes source_raw =
  let* text = Js_read.string_field source_raw "bytes" ~label:"source.bytes" in
  match Int64.of_string_opt text with
  | Some bytes -> Err.return bytes
  | None -> Js_read.fail (`Not_a_decimal "source.bytes")

(* --- responses ----------------------------------------------------------- *)

let response ?request ?options ?render ?token ?error ok =
  let fields = [ ("ok", Js.Unsafe.inject (Js.bool ok)) ] in
  let add name value fields = (name, Js.Unsafe.inject value) :: fields in
  let fields =
    match request with None -> fields | Some v -> add "request" v fields
  in
  let fields =
    match options with None -> fields | Some v -> ("options", v) :: fields
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

(* The NORMALISED options, read back off the private record after
   [Options.create]. Normalisation -- duplicates removed, [all_stages] order
   imposed -- happens in OCaml, so the page cannot derive this value and must
   be told it: the effective-options readout and the reproducible URL both
   record what was actually requested rather than how a control spelled it. *)
let options_echo (options : Me_request.Options.t) =
  let stages =
    Array.of_list
      (List.map
         (fun s -> Js.Unsafe.inject (Js.string (C.stage_name s)))
         options.Me_request.Options.stages)
  in
  Js.Unsafe.obj
    [|
      ("stages", Js.Unsafe.inject (Js.array stages));
      ("fold", Js.Unsafe.inject (Js.bool options.Me_request.Options.fold));
      ( "verifySymbolic",
        match options.Me_request.Options.verify_symbolic with
        | None -> Js.Unsafe.pure_js_expr "null"
        | Some effort ->
            Js.Unsafe.inject (Js.string (Map_verify.Effort.to_string effort)) );
    |]

(* A boundary that emits OUTWARD -- this response reaches the browser -- prints
   [Err.Exn.pp_kind], never [Printexc.to_string]: the default rendering of
   [Err.Exn.E] carries the detection stack, which is provenance the page has no
   use for and should not be shown. *)
let error_text = function
  | Err.Exn.E packed -> Format.asprintf "%a" Err.Exn.pp_kind packed
  | exn -> Printexc.to_string exn

let build_session raw =
  try
    let id_text =
      Err.or_raise ~pp_error:Js_read.pp_error (Js_read.string_field raw "id")
    in
    let id =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Request_id.of_string id_text)
    in
    let source_raw = Js_read.field raw "source" in
    let origin =
      Err.or_raise ~pp_error:Js_read.pp_error (read_origin source_raw)
    in
    let bytes =
      Err.or_raise ~pp_error:Js_read.pp_error (read_bytes source_raw)
    in
    let name =
      Err.or_raise ~pp_error:Js_read.pp_error
        (Js_read.string_field source_raw "name" ~label:"source.name")
    in
    let source =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Source.create ~limits:Me_limits.Limits.untrusted ~origin
           ~name ~bytes ~format:`Model_json)
    in
    let stages, fold, verify_symbolic =
      Err.or_raise ~pp_error:Js_read.pp_error (read_options raw)
    in
    let options =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Options.create ~stages ~fold ~verify_symbolic
           ~namespace:Me_request.Options.Structural)
    in
    let limits =
      Err.or_raise ~pp_error:Me_limits.pp_error
        (Me_limits.Wire_limits.of_limits ~ceiling:Me_limits.Limits.untrusted
           Me_limits.Limits.untrusted)
    in
    let request =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Request.build_session ~id ~source ~options ~limits)
    in
    match Jsont_bytesrw.encode_string Me_request.Request.jsont request with
    | Ok json ->
        response true
          ~request:(Buffer.fresh_array_buffer json)
          ~options:(options_echo options)
    | Error message -> response false ~error:message
  with exn -> response false ~error:(error_text exn)

(* A detail uses the same source and normalised options as a session request;
   only its closed parent key differs.  Keeping construction here means the
   page never serialises an unchecked rendered node id into the worker wire. *)
let build_detail raw =
  try
    let id_text =
      Err.or_raise ~pp_error:Js_read.pp_error (Js_read.string_field raw "id")
    in
    let id =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Request_id.of_string id_text)
    in
    let source_raw = Js_read.field raw "source" in
    let origin =
      Err.or_raise ~pp_error:Js_read.pp_error (read_origin source_raw)
    in
    let bytes =
      Err.or_raise ~pp_error:Js_read.pp_error (read_bytes source_raw)
    in
    let name =
      Err.or_raise ~pp_error:Js_read.pp_error
        (Js_read.string_field source_raw "name" ~label:"source.name")
    in
    let source =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Source.create ~limits:Me_limits.Limits.untrusted ~origin
           ~name ~bytes ~format:`Model_json)
    in
    let stages, fold, verify_symbolic =
      Err.or_raise ~pp_error:Js_read.pp_error (read_options raw)
    in
    let options =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Options.create ~stages ~fold ~verify_symbolic
           ~namespace:Me_request.Options.Structural)
    in
    let key_raw = Js_read.field raw "key" in
    if not (Js_read.is_object key_raw) then
      Err.or_raise ~pp_error:Js_read.pp_error
        (Js_read.fail (`Not_an_object "key"));
    let parent_graph =
      Err.or_raise ~pp_error:Js_read.pp_error
        (Js_read.string_field key_raw "parentGraph" ~label:"key.parentGraph")
    in
    let node =
      Err.or_raise ~pp_error:Js_read.pp_error
        (Js_read.int_field key_raw "node" ~label:"key.node")
    in
    let limits =
      Err.or_raise ~pp_error:Me_limits.pp_error
        (Me_limits.Wire_limits.of_limits ~ceiling:Me_limits.Limits.untrusted
           Me_limits.Limits.untrusted)
    in
    let key =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Detail_key.create_operator
           ~limits:Me_limits.Limits.untrusted ~parent_graph
           ~node:(Graph_ir.Node_id.of_int node))
    in
    let request =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Request.build_detail ~id ~source ~options ~limits ~key)
    in
    match Jsont_bytesrw.encode_string Me_request.Request.jsont request with
    | Ok json -> response true ~request:(Buffer.fresh_array_buffer json)
    | Error message -> response false ~error:message
  with exn -> response false ~error:(error_text exn)

let staged = ref None

let prepare expected_id meta_text document =
  try
    let expected =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
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
  with exn -> response false ~error:(error_text exn)

(* The delta payload never repeats its request key.  This boundary rebuilds
   that typed key from the pending action and gives it to [Me_detail.apply], so
   a valid delta for one operator cannot be installed on another. *)
let apply_detail expected_id raw_key session_text meta_text document =
  try
    let expected =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Request_id.of_string expected_id)
    in
    let parent_graph =
      Err.or_raise ~pp_error:Js_read.pp_error
        (Js_read.string_field raw_key "parentGraph" ~label:"key.parentGraph")
    in
    let node =
      Err.or_raise ~pp_error:Js_read.pp_error
        (Js_read.int_field raw_key "node" ~label:"key.node")
    in
    let key =
      Err.or_raise ~pp_error:Me_request.Request.pp_error
        (Me_request.Detail_key.create_operator
           ~limits:Me_limits.Limits.untrusted ~parent_graph
           ~node:(Graph_ir.Node_id.of_int node))
    in
    let meta =
      Err.or_raise ~pp_error:Me_response.Meta.pp_error
        (Me_response.Meta.decode meta_text)
    in
    (match meta with
    | Me_response.Meta.Delta d
      when Me_request.Request_id.equal expected d.id
           && Me_request.Detail_key.equal key d.key
           && d.bytes = String.length document ->
        ()
    | _ -> raise Exit);
    let session =
      match
        Jsont_bytesrw.decode_string Me_session.Session.jsont session_text
      with
      | Ok value -> value
      | Error _ -> raise Exit
    in
    let delta =
      match Jsont_bytesrw.decode_string Me_detail.Delta.jsont document with
      | Ok value -> value
      | Error _ -> raise Exit
    in
    let merged =
      Err.or_raise ~pp_error:Me_detail.pp_error
        (Me_detail.apply ~key ~limits:Me_limits.Limits.untrusted session delta)
    in
    match Jsont_bytesrw.encode_string Me_session.Session.jsont merged with
    | Ok render -> response true ~render
    | Error message -> response false ~error:message
  with exn -> response false ~error:(error_text exn)

(* Presentation-only: reassigns constant namespaces (see
   [Me_group_constants]) and hands back the transformed document, so the
   page can install it through the ordinary renderer lifecycle with no new
   worker request. Takes the already-prepared document text directly rather
   than a token -- unlike [prepare]/[commit]/[abort] this never touches
   [staged], because it installs nothing itself.

   Re-validated against [Me_limits.Limits.untrusted] rather than skipped:
   [Me_group_constants.apply] only ever copies an existing, already-valid
   namespace or substitutes the short literal ["parameters"], so it cannot
   newly violate a per-component or total-depth ceiling -- but a cheap check
   is what actually distinguishes "safe to show" from "the transform did not
   raise", and every other boundary in this bridge validates before it
   answers rather than trusting its own reasoning about why it must be fine. *)
let group_constants document =
  try
    let session =
      match Jsont_bytesrw.decode_string Me_session.Session.jsont document with
      | Ok s -> s
      | Error message -> failwith message
    in
    let grouped =
      {
        session with
        Me_session.Session.graph_collections =
          List.map
            (fun (c : Model_explorer.GraphCollection.t) ->
              {
                c with
                Model_explorer.GraphCollection.graphs =
                  List.map
                    (Me_group_constants.apply Me_group_constants.Grouped)
                    c.Model_explorer.GraphCollection.graphs;
              })
            session.Me_session.Session.graph_collections;
      }
    in
    Err.or_raise ~pp_error:Me_session.Session.pp_error
      (Me_session.Session.validate ~limits:Me_limits.Limits.untrusted grouped);
    match Jsont_bytesrw.encode_string Me_session.Session.jsont grouped with
    | Ok render -> response true ~render
    | Error message -> response false ~error:message
  with exn -> response false ~error:(error_text exn)

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
        (* The page reads its quarantine ceiling from here rather than carrying
           a literal of its own: [response_live_bytes] budgets the browser's
           retained elements against this exact number, so a JavaScript copy
           that drifted would silently invalidate the enforced peak. The
           renderer may tighten it, never widen it. *)
        ( "maxQuarantinedElements",
          Js.Unsafe.inject Me_limits.Hard.max_quarantined_elements );
      |]
  in
  let request =
    Js.Unsafe.obj
      [|
        ("buildSession", Js.Unsafe.inject (Js.wrap_callback build_session));
        ("buildDetail", Js.Unsafe.inject (Js.wrap_callback build_detail));
      |]
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
        ( "groupConstants",
          Js.Unsafe.inject
            (Js.wrap_callback (fun document ->
                 group_constants (Js.to_string document))) );
        ( "applyDetail",
          Js.Unsafe.inject
            (Js.wrap_callback (fun id key session meta document ->
                 apply_detail (Js.to_string id) key (Js.to_string session)
                   (Js.to_string meta) (Js.to_string document))) );
      |]
  in
  Js.export "mltorch"
    (Js.Unsafe.obj
       [|
         ("hard", Js.Unsafe.inject hard);
         ("request", Js.Unsafe.inject request);
         ("session", Js.Unsafe.inject session);
       |])
