(* What the worker sends back. See the .mli. *)

module Hard = Me_limits.Hard

module Phase = struct
  type t = Decode | Lower | Project | Encode

  let to_string = function
    | Decode -> "decode"
    | Lower -> "lower"
    | Project -> "project"
    | Encode -> "encode"

  (* The same successor chain the other closed vocabularies here use: a list
     written beside the type compiles while everything iterating it quietly
     stops seeing the new member. *)
  let next = function
    | Decode -> Some Lower
    | Lower -> Some Project
    | Project -> Some Encode
    | Encode -> None

  let all =
    let rec walk acc c =
      match next c with
      | None -> List.rev (c :: acc)
      | Some n -> walk (c :: acc) n
    in
    walk [] Decode

  let of_string s = List.assoc_opt s (List.map (fun p -> (to_string p, p)) all)

  let jsont =
    Jsont.enum ~kind:"phase" (List.map (fun p -> (to_string p, p)) all)
end

module Progress = struct
  type t = {
    id : Me_request.Request_id.t;
    phase : Phase.t;
    done_ : int64;
    total : int64 option;
  }
end

module Session = struct
  type t = {
    id : Me_request.Request_id.t;
    limits : Me_limits.Wire_limits.t;
    json : string;
  }
end

module Delta = struct
  type t = {
    id : Me_request.Request_id.t;
    key : Me_request.Detail_key.t;
    json : string;
  }
end

module Failed = struct
  type t = {
    id : Me_request.Request_id.t;
    key : Me_request.Detail_key.t option;
    error : Me_limits.Diagnostic.t;
  }
end

module Protocol_failure = struct
  type t = { error : Me_limits.Diagnostic.t }
end

module Handle_result = struct
  type t = Session of Session.t | Delta of Delta.t | Failed of Failed.t
end

module Final = struct
  type t =
    | Session of Session.t
    | Delta of Delta.t
    | Failed of Failed.t
    | Protocol_failure of Protocol_failure.t

  let of_handle_result = function
    | Handle_result.Session s -> Session s
    | Handle_result.Delta d -> Delta d
    | Handle_result.Failed f -> Failed f
end

module Meta = struct
  module Session = struct
    type t = {
      id : Me_request.Request_id.t;
      limits : Me_limits.Wire_limits.t;
      bytes : int;
    }
  end

  module Delta = struct
    type t = {
      id : Me_request.Request_id.t;
      key : Me_request.Detail_key.t;
      bytes : int;
    }
  end

  type t =
    | Progress of Progress.t
    | Session of Session.t
    | Delta of Delta.t
    | Failed of Failed.t
    | Protocol_failure of Protocol_failure.t

  let kind = function
    | Progress _ -> "progress"
    | Session _ -> "session"
    | Delta _ -> "delta"
    | Failed _ -> "failed"
    | Protocol_failure _ -> "protocol_failure"

  (* A "kind" discriminant over the five constructors, every optional member
     belonging to exactly one of them. This is the ONLY thing that crosses to
     JavaScript: a compiled OCaml variant through [postMessage] would hand the
     page jsoo's private runtime representation, with no stable discriminant
     and no stable field layout. *)
  let jsont =
    Jsont.Object.map ~kind:"responseMeta"
      (fun kind id phase done_ total limits key bytes error ->
        let need what = function
          | Some v -> v
          | None ->
              Jsont.Error.msgf Jsont.Meta.none
                "response kind %S does not carry its own %s" kind what
        in
        match kind with
        | "progress" ->
            Progress
              {
                Progress.id = need "id" id;
                phase = need "phase" phase;
                done_ = need "done" done_;
                total;
              }
        | "session" ->
            Session
              {
                Session.id = need "id" id;
                limits = need "limits" limits;
                bytes = need "bytes" bytes;
              }
        | "delta" ->
            Delta
              {
                Delta.id = need "id" id;
                key = need "key" key;
                bytes = need "bytes" bytes;
              }
        | "failed" ->
            Failed { Failed.id = need "id" id; key; error = need "error" error }
        | "protocol_failure" ->
            Protocol_failure { Protocol_failure.error = need "error" error }
        | _ -> Jsont.Error.msgf Jsont.Meta.none "unknown response kind %S" kind)
    |> Jsont.Object.mem "kind" Jsont.string ~enc:kind
    |> Jsont.Object.opt_mem "id" Me_request.Request_id.jsont ~enc:(function
      | Progress p -> Some p.Progress.id
      | Session s -> Some s.Session.id
      | Delta d -> Some d.Delta.id
      | Failed f -> Some f.Failed.id
      | Protocol_failure _ -> None)
    |> Jsont.Object.opt_mem "phase" Phase.jsont ~enc:(function
      | Progress p -> Some p.Progress.phase
      | _ -> None)
    (* [int64_as_string] for every counter: a progress total is unbounded by
       anything in a profile, and a JSON number comes back through a 32-bit
       [int] under node. *)
    |> Jsont.Object.opt_mem "done" Jsont.int64_as_string ~enc:(function
      | Progress p -> Some p.Progress.done_
      | _ -> None)
    |> Jsont.Object.opt_mem "total" Jsont.int64_as_string ~enc:(function
      | Progress p -> p.Progress.total
      | _ -> None)
    |> Jsont.Object.opt_mem "limits" Me_limits.Wire_limits.jsont ~enc:(function
      | Session s -> Some s.Session.limits
      | _ -> None)
    |> Jsont.Object.opt_mem "key" Me_request.Detail_key.jsont ~enc:(function
      | Delta d -> Some d.Delta.key
      | Failed f -> f.Failed.key
      | _ -> None)
    (* [int], not [int64], and the width rule does not bite: it is the length of
       a string that already exists, bounded by [max_session_bytes], itself an
       [int] because the writer limit is one. *)
    |> Jsont.Object.opt_mem "bytes" Jsont.int ~enc:(function
      | Session s -> Some s.Session.bytes
      | Delta d -> Some d.Delta.bytes
      | _ -> None)
    |> Jsont.Object.opt_mem "error" Me_limits.Diagnostic.jsont ~enc:(function
      | Failed f -> Some f.Failed.error
      | Protocol_failure p -> Some p.Protocol_failure.error
      | _ -> None)
    |> Jsont.Object.finish

  let decode = Jsont_bytesrw.decode_string jsont
end

module Wire = struct
  type t = { meta : string; payload : string option }
  type error = [ `Meta_too_large ]

  let pp_error fmt : [< error ] -> unit = function
    | `Meta_too_large -> Fmt.string fmt "response metadata is over the ceiling"

  (* An AGGREGATE bound, applied by the writer rather than inferred from
     individually-bounded fields: JSON escaping expands, and three bounded
     fields still sum. The limit filter's default action raises, so the
     exception is part of the contract and is caught here. *)
  let encode_bounded ~max_meta_bytes meta =
    let buf = Buffer.create 256 in
    let w =
      Bytesrw.Bytes.Writer.limit max_meta_bytes
        (Bytesrw.Bytes.Writer.of_buffer buf)
        ~eod:true
    in
    match Jsont_bytesrw.encode Meta.jsont meta ~eod:true w with
    | Ok () -> Core.return (Buffer.contents buf)
    | Error _ -> Core.fail `Meta_too_large
    | exception Bytesrw.Bytes.Stream.Error _ -> Core.fail `Meta_too_large

  let wrap ~max_meta_bytes meta payload =
    let open Core.Syntax in
    let+ meta = encode_bounded ~max_meta_bytes meta in
    { meta; payload }

  let of_progress_bounded ~max_meta_bytes p =
    wrap ~max_meta_bytes (Meta.Progress p) None

  (* The document and the count that describes it are produced HERE, in one
     place, from the same string -- which is what makes them unable to
     disagree. *)
  let of_final_bounded ~max_meta_bytes (f : Final.t) =
    match f with
    | Final.Session s ->
        wrap ~max_meta_bytes
          (Meta.Session
             {
               Meta.Session.id = s.Session.id;
               limits = s.Session.limits;
               bytes = String.length s.Session.json;
             })
          (Some s.Session.json)
    | Final.Delta d ->
        wrap ~max_meta_bytes
          (Meta.Delta
             {
               Meta.Delta.id = d.Delta.id;
               key = d.Delta.key;
               bytes = String.length d.Delta.json;
             })
          (Some d.Delta.json)
    | Final.Failed f -> wrap ~max_meta_bytes (Meta.Failed f) None
    | Final.Protocol_failure p ->
        wrap ~max_meta_bytes (Meta.Protocol_failure p) None

  let of_progress p =
    of_progress_bounded ~max_meta_bytes:Hard.max_response_meta_bytes p

  let of_final f =
    of_final_bounded ~max_meta_bytes:Hard.max_response_meta_bytes f

  (* PRECOMPUTED, so the terminal path never runs the encoder. "The fallback
     cannot itself fail" is then a property of a value rather than an argument
     about a function, and the suite can assert the ceiling over exactly these
     bytes. [or_raise] is correct here and nowhere else: this is module
     initialisation over a constant, so a failure is a build-time defect rather
     than a runtime outcome. *)
  let constant_protocol_failure =
    Core.or_raise pp_error
      (of_final
         (Final.Protocol_failure
            {
              Protocol_failure.error =
                Me_limits.Diagnostic.create ~limits:Me_limits.Limits.untrusted
                  Me_limits.Diagnostic.Code.Internal
                  "the worker could not answer this request";
            }))
end
