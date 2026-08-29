(* Bounded diagnostics that cross the worker/page/session boundaries.
   Split from me_limits.ml. Depends on me_limits_profile.ml
   for [Limits.t]/[Limits.trusted]. *)

open Me_limits_profile

(* --- diagnostics --- *)

module Diagnostic = struct
  module Code = struct
    type t =
      | Buffer_mismatch
      | Inconsistent_mount
      | Internal
      | Invalid_limits
      | Invalid_source
      | Key_disagrees_with_ids
      | Malformed_request
      | Malformed_response
      | Not_an_array_buffer
      | Not_implemented
      | Outside_dialect_domain
      | Over_limit
      | Prerequisite_unavailable
      | Request_in_flight
      | Requires_payloads
      | Settlement_mismatch
      | Stale_epoch
      | Unsupported_detail_key
      | Unsupported_graph_shape
      | Unsupported_input
      | Unsupported_operator

    let to_string = function
      | Buffer_mismatch -> "buffer_mismatch"
      | Inconsistent_mount -> "inconsistent_mount"
      | Internal -> "internal"
      | Invalid_limits -> "invalid_limits"
      | Invalid_source -> "invalid_source"
      | Key_disagrees_with_ids -> "key_disagrees_with_ids"
      | Malformed_request -> "malformed_request"
      | Malformed_response -> "malformed_response"
      | Not_an_array_buffer -> "not_an_array_buffer"
      | Not_implemented -> "not_implemented"
      | Outside_dialect_domain -> "outside_dialect_domain"
      | Over_limit -> "over_limit"
      | Prerequisite_unavailable -> "prerequisite_unavailable"
      | Request_in_flight -> "request_in_flight"
      | Requires_payloads -> "requires_payloads"
      | Settlement_mismatch -> "settlement_mismatch"
      | Stale_epoch -> "stale_epoch"
      | Unsupported_detail_key -> "unsupported_detail_key"
      | Unsupported_graph_shape -> "unsupported_graph_shape"
      | Unsupported_input -> "unsupported_input"
      | Unsupported_operator -> "unsupported_operator"

    (* This successor relation defines the public vocabulary timeline used by
       [all]; the declaration and printer above remain alphabetical. *)
    let next = function
      | Over_limit -> Some Malformed_request
      | Malformed_request -> Some Invalid_limits
      | Invalid_limits -> Some Invalid_source
      | Invalid_source -> Some Malformed_response
      | Malformed_response -> Some Request_in_flight
      | Request_in_flight -> Some Inconsistent_mount
      | Inconsistent_mount -> Some Settlement_mismatch
      | Settlement_mismatch -> Some Buffer_mismatch
      | Buffer_mismatch -> Some Not_an_array_buffer
      | Not_an_array_buffer -> Some Stale_epoch
      | Stale_epoch -> Some Unsupported_detail_key
      | Unsupported_detail_key -> Some Key_disagrees_with_ids
      | Key_disagrees_with_ids -> Some Unsupported_operator
      | Unsupported_operator -> Some Unsupported_input
      | Unsupported_input -> Some Unsupported_graph_shape
      | Unsupported_graph_shape -> Some Outside_dialect_domain
      | Outside_dialect_domain -> Some Requires_payloads
      | Requires_payloads -> Some Prerequisite_unavailable
      | Prerequisite_unavailable -> Some Not_implemented
      | Not_implemented -> Some Internal
      | Internal -> None

    let all =
      let rec walk acc c =
        match next c with
        | None -> List.rev (c :: acc)
        | Some n -> walk (c :: acc) n
      in
      walk [] Over_limit

    let assoc = List.map (fun c -> (to_string c, c)) all

    (* Total in its failure, so an unknown tag from a future producer decodes
       as a named error rather than being silently dropped. *)
    let of_string s = List.assoc_opt s assoc
    let jsont = Jsont.enum ~kind:"diagnostic_code" assoc
  end

  type t = {
    code : Code.t;
    message : string;
    graph : string option;
    truncated : bool;
  }

  (* Sanitisation is REPLACEMENT, not only truncation. [of_exn] admits
     arbitrary OCaml bytes, which need not be valid UTF-8 at all, and
     [Jsont_bytesrw]'s encoder does NOT validate them — it passes them through
     and emits a document that is not JSON. So nothing in OCaml catches it, and
     the failure surfaces in the browser as a malformed response: the very
     condition the diagnostic existed to report. Invalid sequences and control
     bytes therefore become U+FFFD BEFORE the length rule applies, and the
     length rule then cuts on a scalar-value boundary. *)

  let is_control u =
    let c = Uchar.to_int u in
    c < 0x20 || c = 0x7F

  let utf_8_width u =
    let c = Uchar.to_int u in
    if c < 0x80 then 1
    else if c < 0x800 then 2
    else if c < 0x10000 then 3
    else 4

  let sanitise ~max s =
    let buf = Buffer.create (min (String.length s) max) in
    let truncated = ref false in
    let i = ref 0 in
    let n = String.length s in
    (try
       while !i < n do
         let d = String.get_utf_8_uchar s !i in
         let u =
           if Uchar.utf_decode_is_valid d then
             let u = Uchar.utf_decode_uchar d in
             if is_control u then Uchar.rep else u
           else Uchar.rep
         in
         (* Replacement can WIDEN — one stray byte becomes three — so the
            ceiling is tested against what is about to be written, not against
            what was read. *)
         if Buffer.length buf + utf_8_width u > max then begin
           truncated := true;
           raise Exit
         end;
         Buffer.add_utf_8_uchar buf u;
         i := !i + Uchar.utf_decode_length d
       done
     with Exit -> ());
    (Buffer.contents buf, !truncated)

  (* An identifier is either exactly right or absent. Truncating one yields a
     different identifier, which either resolves to the wrong graph or fails to
     resolve while looking authoritative; sanitising one has the same defect for
     the same reason. So this is a predicate and its failure is a DROP. *)
  let acceptable_id ~max s =
    String.length s <= max
    &&
    let ok = ref true and i = ref 0 in
    let n = String.length s in
    while !ok && !i < n do
      let d = String.get_utf_8_uchar s !i in
      if
        (not (Uchar.utf_decode_is_valid d))
        || is_control (Uchar.utf_decode_uchar d)
      then ok := false
      else i := !i + Uchar.utf_decode_length d
    done;
    !ok

  let build ~limits ?graph code message =
    let message, message_truncated =
      sanitise ~max:limits.Limits.max_diagnostic_bytes message
    in
    let graph, graph_dropped =
      match graph with
      | None -> (None, false)
      | Some g when acceptable_id ~max:limits.Limits.max_id_bytes g ->
          (Some g, false)
      | Some _ -> (None, true)
    in
    (* The drop sets [truncated] too, so a missing graph is visible rather than
       indistinguishable from one that was never supplied. *)
    { code; message; graph; truncated = message_truncated || graph_dropped }

  let create ~limits ?graph code message = build ~limits ?graph code message

  (* [Err.Exn.E] is matched rather than left to [Printexc.to_string], which
     WOULD render it: [Err] registers a printer, so the default path emits the
     payload followed by the detection stack and the semantic trace. That text
     is a developer diagnostic, and this value is a wire response the browser
     receives -- shipping frame-by-frame provenance of the server's own source
     to a client is not something a message like this should do, and the
     sanitiser below would only truncate it, not remove it.

     So render the payload alone here. The provenance is not lost, it simply
     stays on this side of the boundary: the exception is still whole for any
     top-level handler that logs it. *)
  let of_exn ~limits e =
    let message =
      match e with
      | Err.Exn.E packed -> Format.asprintf "%a" Err.Exn.pp_kind packed
      | e -> Printexc.to_string e
    in
    build ~limits Code.Internal message

  let jsont =
    Jsont.Object.map ~kind:"diagnostic" (fun code message graph truncated ->
        (* Through the same constructor as the producer, under HARD limits: a
           [Jsont.t] carries no profile, and the profile-level bound is applied
           by whoever holds one. Whatever the wire claimed about [truncated] is
           kept beside what re-sanitising found, since a producer that already
           cut something knows that and this decoder cannot. *)
        let d = build ~limits:Limits.trusted ?graph code message in
        { d with truncated = d.truncated || truncated })
    |> Jsont.Object.mem "code" Code.jsont ~enc:(fun d -> d.code)
    |> Jsont.Object.mem "message" Jsont.string ~enc:(fun d -> d.message)
    |> Jsont.Object.opt_mem "graph" Jsont.string ~enc:(fun d -> d.graph)
    |> Jsont.Object.mem "truncated" Jsont.bool ~dec_absent:false ~enc:(fun d ->
        d.truncated)
    |> Jsont.Object.finish

  let pp fmt d =
    Fmt.pf fmt "%s: %s%a%s" (Code.to_string d.code) d.message
      (Core.Pretty.option_or ~none:"" (fun fmt g -> Fmt.pf fmt " [%s]" g))
      d.graph
      (if d.truncated then " (truncated)" else "")
end
