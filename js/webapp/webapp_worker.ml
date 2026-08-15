open Js_of_ocaml
module Buffer = Webapp_jsoo_buffer

let post wire =
  let message =
    match wire.Me_response.Wire.payload with
    | None ->
        Js.Unsafe.obj [| ("meta", Js.Unsafe.inject (Js.string wire.meta)) |]
    | Some payload ->
        let payload = Buffer.fresh_array_buffer payload in
        Js.Unsafe.obj
          [|
            ("meta", Js.Unsafe.inject (Js.string wire.meta));
            ("payload", Js.Unsafe.inject payload);
          |]
  in
  let args =
    match wire.Me_response.Wire.payload with
    | None -> [| Js.Unsafe.inject message |]
    | Some _ ->
        [|
          Js.Unsafe.inject message;
          Js.Unsafe.inject (Js.array [| Js.Unsafe.get message "payload" |]);
        |]
  in
  ignore
    (Js.Unsafe.fun_call (Js.Unsafe.get Js.Unsafe.global "postMessage") args)

let protocol_failure () = post Me_response.Wire.constant_protocol_failure

let emit progress =
  match Me_response.Wire.of_progress progress with
  | Ok wire -> post wire
  | Error _ -> protocol_failure ()

let handle event =
  try
    let data = Js.Unsafe.get event "data" in
    let request = Js.Unsafe.get data "request" in
    let bytes = Js.Unsafe.get data "bytes" in
    let request_length =
      match Buffer.array_buffer_length request with
      | Error _ -> raise Exit
      | Ok n ->
          Err.or_raise
            ~pp_error:(fun _ _ -> ())
            (Me_bytes.checked_length n
               ~max:(Int64.of_int Me_limits.Hard.max_request_json_bytes))
    in
    let request =
      match
        Jsont_bytesrw.decode_string Me_request.Request.jsont
          (Buffer.to_string (Js.Unsafe.coerce request) ~len:request_length)
      with
      | Ok request -> request
      | Error _ -> raise Exit
    in
    let bytes_length =
      match Buffer.array_buffer_length bytes with
      | Error _ -> raise Exit
      | Ok n ->
          Err.or_raise
            ~pp_error:(fun _ _ -> ())
            (Me_bytes.checked_length n ~max:Me_limits.Hard.jsoo_safe_bytes)
    in
    if
      Int64.compare
        (Int64.of_int bytes_length)
        (Me_request.Source.bytes (Me_request.Request.source request))
      <> 0
    then raise Exit;
    let result =
      Me_export.handle ~emit request
        ~bytes:(Buffer.to_string (Js.Unsafe.coerce bytes) ~len:bytes_length)
    in
    match
      Me_response.Wire.of_final (Me_response.Final.of_handle_result result)
    with
    | Ok wire -> post wire
    | Error _ -> protocol_failure ()
  with e ->
    let detail =
      match e with
      | Err.Exn.E packed -> Format.asprintf "%a" Err.Exn.pp_kind packed
      | e -> Printexc.to_string e
    in
    ignore
      (Js.Unsafe.meth_call Console.console "error"
         [| Js.Unsafe.inject (Js.string detail) |]);
    protocol_failure ()

let () = Js.Unsafe.set Js.Unsafe.global "onmessage" (Js.wrap_callback handle)
