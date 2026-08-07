(* [Me_request]: the worker request (.ai/model_explorer_design.md).

   Four properties, none of them visible in a round trip that happens to
   succeed:

   - request identity is a GRAMMAR, so the strings the builder can issue and the
     strings the worker accepts are the same set — a length ceiling admits
     sequences no producer bounded by 2^32 can emit;
   - the sequence comparison never passes through [int], which wraps under node;
   - a stage list is NORMALISED, so two spellings of one request are one value;
   - the constructors REVALIDATE under the request's own profile, so a component
     valid alone can be invalid in the request that carries it. *)

module MR = Me_request
module L = Me_limits.Limits

let limits = L.untrusted

let wire =
  Core.or_raise Me_limits.pp_error
    (Me_limits.Wire_limits.of_limits ~ceiling:L.untrusted L.untrusted)

let wire_small =
  Core.or_raise Me_limits.pp_error
    (Me_limits.Wire_limits.of_limits ~ceiling:L.untrusted L.small)

let uuid = "0f8fad5b-d9cb-469f-a165-70867728950e"

let pp_id ppf r =
  Core.Pretty.core_result
    ~ok:(fun ppf i -> Fmt.pf ppf "ok epoch=%s" (MR.Request_id.epoch i))
    ~error:MR.Request.pp_error ppf r

(* --- request identity --- *)

let%expect_test "the grammar, exactly" =
  List.iter
    (fun s -> Format.printf "%-52s %a@." s pp_id (MR.Request_id.of_string s))
    [
      uuid ^ "-0";
      uuid ^ "-1";
      uuid ^ "-4294967295";
      (* one past the ceiling: ten digits, so a length rule admits it *)
      uuid ^ "-4294967296";
      (* the value a length-only rule was silently accepting *)
      uuid ^ "-9999999999";
      (* eleven digits *)
      uuid ^ "-42949672950";
      (* a leading zero is a second spelling of one sequence *)
      uuid ^ "-01";
      (* uppercase is a second spelling of one epoch *)
      String.uppercase_ascii uuid ^ "-1";
      (* no sequence at all *)
      uuid;
      uuid ^ "-";
      (* a UUID whose hyphens are misplaced *)
      "0f8fad5bd9cb-469f-a165-70867728950e0-1";
    ];
  [%expect
    {|
    0f8fad5b-d9cb-469f-a165-70867728950e-0               ok epoch=0f8fad5b-d9cb-469f-a165-70867728950e
    0f8fad5b-d9cb-469f-a165-70867728950e-1               ok epoch=0f8fad5b-d9cb-469f-a165-70867728950e
    0f8fad5b-d9cb-469f-a165-70867728950e-4294967295      ok epoch=0f8fad5b-d9cb-469f-a165-70867728950e
    0f8fad5b-d9cb-469f-a165-70867728950e-4294967296      malformed request id
    0f8fad5b-d9cb-469f-a165-70867728950e-9999999999      malformed request id
    0f8fad5b-d9cb-469f-a165-70867728950e-42949672950     malformed request id
    0f8fad5b-d9cb-469f-a165-70867728950e-01              malformed request id
    0F8FAD5B-D9CB-469F-A165-70867728950E-1               malformed request id
    0f8fad5b-d9cb-469f-a165-70867728950e                 malformed request id
    0f8fad5b-d9cb-469f-a165-70867728950e-                malformed request id
    0f8fad5bd9cb-469f-a165-70867728950e0-1               malformed request id |}]

let%expect_test "the widest accepted id is within Hard's byte constant" =
  (* [Hard.max_request_id_bytes] is a CONSEQUENCE of the grammar, not a second
     rule. If they ever disagree, the byte constant is the one that is wrong. *)
  let widest = uuid ^ "-4294967295" in
  Printf.printf "widest=%d hard=%d epoch=%d/%d\n" (String.length widest)
    Me_limits.Hard.max_request_id_bytes (String.length uuid)
    Me_limits.Hard.max_epoch_bytes;
  [%expect {| widest=47 hard=47 epoch=36/36 |}]

(* --- options --- *)

module C = Me_session.Capability

let opts ?(stages = [ C.Canonical ]) ?(fold = false) () =
  MR.Options.create ~stages ~fold ~verify_symbolic:None
    ~namespace:MR.Options.Structural

let pp_opts ppf r =
  Core.Pretty.core_result
    ~ok:(fun ppf (o : MR.Options.t) ->
      Fmt.pf ppf "[%s]"
        (String.concat " " (List.map C.stage_name o.MR.Options.stages)))
    ~error:MR.Request.pp_error ppf r

let%expect_test "two spellings of one request are one value" =
  (* Normalised against [all_stages], which removes duplicates and imposes the
     constructor order in one pass -- so the list is bounded by that type's
     cardinality by construction, with no separate ceiling. *)
  List.iter
    (fun stages -> Format.printf "%a@." pp_opts (opts ~stages ()))
    [
      [ C.Canonical; C.Source ];
      [ C.Source; C.Canonical ];
      [ C.Canonical; C.Canonical; C.Source; C.Canonical ];
      [];
    ];
  [%expect
    {|
    [source canonical]
    [source canonical]
    [source canonical]
    a request must ask for at least one stage |}]

(* --- the source --- *)

let digest = String.make 64 'a'

let src ?(origin = MR.Source.Origin.Local) ?(name = "resnet18") ?(bytes = 1024L)
    ?(limits = limits) () =
  MR.Source.create ~limits ~origin ~name ~bytes ~format:`Model_json

let pp_src ppf r =
  Core.Pretty.core_result
    ~ok:(fun ppf (s : MR.Source.t) ->
      Fmt.pf ppf "ok digest=%a"
        (Core.Pretty.option_or ~none:"none" Fmt.string)
        (MR.Source.verified_sha256 s))
    ~error:MR.Request.pp_error ppf r

let%expect_test "a digest is only constructible where it is meaningful" =
  (* The digest lives INSIDE [Catalog], so a local source claiming a verified
     digest -- a false verification claim entering the deterministic session --
     is not a value anyone can build. *)
  Format.printf "local   %a@." pp_src (src ());
  Format.printf "catalog %a@." pp_src
    (src
       ~origin:
         (MR.Source.Origin.Catalog
            {
              MR.Source.Origin.Catalog.url = "https://example.test/m.pt2";
              ref_ = "v1";
              verified_sha256 = digest;
            })
       ());
  [%expect
    {|
    local   ok digest=none
    catalog ok digest=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa |}]

let%expect_test "every rejection row" =
  let catalog ?(url = "u") ?(ref_ = "r") ?(sha = digest) () =
    MR.Source.Origin.Catalog
      { MR.Source.Origin.Catalog.url; ref_; verified_sha256 = sha }
  in
  let long n = String.make n 'x' in
  List.iter
    (fun (label, r) -> Format.printf "%-14s %a@." label pp_src r)
    [
      ("negative", src ~bytes:(-1L) ());
      ("name", src ~name:(long (limits.L.max_label_bytes + 1)) ());
      ( "url",
        src ~origin:(catalog ~url:(long (limits.L.max_url_bytes + 1)) ()) () );
      ( "ref",
        src ~origin:(catalog ~ref_:(long (limits.L.max_url_bytes + 1)) ()) () );
      ("short digest", src ~origin:(catalog ~sha:"abc" ()) ());
      ("upper digest", src ~origin:(catalog ~sha:(String.make 64 'A') ()) ());
    ];
  [%expect
    {|
    negative       source byte count is negative
    name           source name is too long
    url            source url is too long
    ref            source ref is too long
    short digest   source digest is not 64 lowercase hex bytes
    upper digest   source digest is not 64 lowercase hex bytes |}]

(* --- the request itself --- *)

let id =
  Core.or_raise MR.Request.pp_error (MR.Request_id.of_string (uuid ^ "-7"))

let options = Core.or_raise MR.Request.pp_error (opts ())

let pp_req ppf r =
  Core.Pretty.core_result
    ~ok:(fun ppf (t : MR.Request.t) ->
      Fmt.pf ppf "ok epoch=%s key=%a" (MR.Request.epoch t)
        (Core.Pretty.option_or ~none:"none" (fun ppf k ->
             Fmt.string ppf (MR.Detail_key.id k)))
        (MR.Request.key t))
    ~error:MR.Request.pp_error ppf r

let%expect_test "a session request, and a detail request" =
  let source = Core.or_raise MR.Request.pp_error (src ()) in
  Format.printf "session %a@." pp_req
    (MR.Request.build_session ~id ~source ~options ~limits:wire);
  let key =
    Core.or_raise MR.Request.pp_error
      (MR.Detail_key.create ~limits ~parent_graph:"g/native/001"
         ~value:(Graph_ir.Tensor_id.of_int 12))
  in
  Format.printf "detail  %a@." pp_req
    (MR.Request.build_detail ~id ~source ~options ~limits:wire ~key);
  Printf.printf "parent node %s\n" (MR.Detail_key.parent_node key);
  [%expect
    {|
    session ok epoch=0f8fad5b-d9cb-469f-a165-70867728950e key=none
    detail  ok epoch=0f8fad5b-d9cb-469f-a165-70867728950e key=expr/g/native/001/t12/t12
    parent node t12 |}]

let%expect_test "a component valid ALONE can be invalid in its request" =
  (* This is what revalidation buys, and it is the row a "were the components
     built under this profile" check could not express: a [Source.t] keeps no
     witness of the profile it was checked against, so only re-running the
     checks under the REQUEST's profile can answer. *)
  let name = String.make (L.small.L.max_label_bytes + 1) 'x' in
  let source = Core.or_raise MR.Request.pp_error (src ~name ()) in
  Format.printf "under untrusted %a@." pp_req
    (MR.Request.build_session ~id ~source ~options ~limits:wire);
  Format.printf "under small     %a@." pp_req
    (MR.Request.build_session ~id ~source ~options ~limits:wire_small);
  [%expect
    {|
    under untrusted ok epoch=0f8fad5b-d9cb-469f-a165-70867728950e key=none
    under small     source name is too long |}]

let%expect_test "and so can a detail key" =
  let source = Core.or_raise MR.Request.pp_error (src ()) in
  let parent_graph = String.make (L.small.L.max_id_bytes + 1) 'g' in
  let key =
    Core.or_raise MR.Request.pp_error
      (MR.Detail_key.create ~limits ~parent_graph
         ~value:(Graph_ir.Tensor_id.of_int 0))
  in
  Format.printf "under small %a@." pp_req
    (MR.Request.build_detail ~id ~source ~options ~limits:wire_small ~key);
  [%expect {| under small detail key parent graph is too long |}]

let%expect_test "the derived id is checked, not only its input" =
  (* A [parent_graph] inside the ceiling can still produce a rendered id past
     it, because the id is longer than the component it is built from. *)
  let parent_graph = String.make limits.L.max_id_bytes 'g' in
  Format.printf "%a@."
    (Core.Pretty.core_result ~ok:(Fmt.any "ok") ~error:MR.Request.pp_error)
    (MR.Detail_key.create ~limits ~parent_graph
       ~value:(Graph_ir.Tensor_id.of_int 0));
  [%expect {| derived detail id is too long |}]

(* --- the wire --- *)

let decode = Jsont_bytesrw.decode_string MR.Request.jsont
let encode = Jsont_bytesrw.encode_string MR.Request.jsont

(* Jsont appends the whole location context to an error. The first line is the
   message the check produced; the rest is where in the document it was, which
   a golden would re-promote on every unrelated fixture edit. *)
let first_line s =
  match String.index_opt s '\n' with None -> s | Some i -> String.sub s 0 i

let pp_wire ppf r =
  Core.Pretty.result ~ok:Fmt.string
    ~error:(fun ppf e -> Fmt.pf ppf "REJECTED %s" (first_line e))
    ppf r

let%expect_test "what a session request looks like on the wire" =
  let source = Core.or_raise MR.Request.pp_error (src ()) in
  let req =
    Core.or_raise MR.Request.pp_error
      (MR.Request.build_session ~id ~source ~options ~limits:wire)
  in
  Format.printf "%a@." pp_wire (encode req);
  [%expect
    {| {"id":"0f8fad5b-d9cb-469f-a165-70867728950e-7","limits":{},"options":{"stages":["canonical"],"fold":false,"namespace":"structural"},"source":{"name":"resnet18","bytes":"1024","format":"model_json"}} |}]

(* Encode, decode, encode: equal bytes says the decoder lands in the encoder's
   own domain, which is what finishing through [build_session] buys. *)
let round_trip req =
  Result.bind (encode req) (fun text ->
      Result.bind (decode text) (fun back ->
          Result.map (fun again -> String.equal text again) (encode back)))

let%expect_test "the decoder lands in the encoder's domain" =
  let source = Core.or_raise MR.Request.pp_error (src ()) in
  let key =
    Core.or_raise MR.Request.pp_error
      (MR.Detail_key.create ~limits ~parent_graph:"g/native/001"
         ~value:(Graph_ir.Tensor_id.of_int 12))
  in
  List.iter
    (fun (label, req) ->
      Format.printf "%-8s %a@." label
        (Core.Pretty.result ~ok:Fmt.bool ~error:Fmt.string)
        (round_trip (Core.or_raise MR.Request.pp_error req)))
    [
      ("session", MR.Request.build_session ~id ~source ~options ~limits:wire);
      ("detail", MR.Request.build_detail ~id ~source ~options ~limits:wire ~key);
    ];
  [%expect {|
    session  true
    detail   true |}]

let%expect_test "the staged decoder rejects in its own order" =
  (* [Source.create ~limits] needs the DECODED limits, so the members cannot be
     validated independently. Each row below is refused by the step that owns
     it, and the profile row is refused before the source it would govern. *)
  let body ?(id = uuid ^ "-7") ?(limits = "{}")
      ?(source = {|{"name":"m","bytes":"1","format":"model_json"}|}) () =
    Printf.sprintf
      {|{"id":"%s","limits":%s,"options":{"stages":["canonical"],"fold":false,"namespace":"structural"},"source":%s}|}
      id limits source
  in
  List.iter
    (fun (label, text) ->
      Format.printf "%-32s %a@." label pp_wire
        (Result.map (fun _ -> "accepted") (decode text)))
    [
      ("ok", body ());
      ("bad id", body ~id:(uuid ^ "-9999999999") ());
      ("bad profile", body ~limits:{|{"max_views":"999999"}|} ());
      ( "bad source",
        body ~source:{|{"name":"m","bytes":"-1","format":"model_json"}|} () );
      ( "source under a tightened profile",
        body ~limits:{|{"max_label_bytes":"4"}|}
          ~source:
            {|{"name":"far too long a name","bytes":"1","format":"model_json"}|}
          () );
    ];
  [%expect
    {|
    ok                               accepted
    bad id                           REJECTED malformed request id
    bad profile                      REJECTED invalid limit max_views = 999999
    bad source                       REJECTED source byte count is negative
    source under a tightened profile REJECTED source name is too long |}]
