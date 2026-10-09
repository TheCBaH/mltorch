open Err.Syntax

let read ?(max_bytes = 0x4000000) path =
  match
    In_channel.with_open_bin path (fun ic ->
        let len = In_channel.length ic in
        if Int64.compare len (Int64.of_int max_bytes) > 0 then Error len
        else Ok (In_channel.input_all ic))
  with
  | Ok s -> Err.return s
  | Error len ->
      Err.fail
        (`Limit
           {
             Fault.Limit_fault.what = Fault.Archive_bytes;
             limit = Int64.of_int max_bytes;
             actual = len;
           })
  | exception Sys_error m -> Err.fail (`File_io (path, m))

let ensure ?hash ?transport ~layer cache (pin : Cache.Pin.t) =
  let download transport =
    let tmp =
      match
        Filename.temp_file ~temp_dir:(Cache.temp_dir cache) "download" ".part"
      with
      | p -> Ok p
      | exception Sys_error m -> Error m
    in
    match tmp with
    | Error m -> Err.fail (`File_io (Cache.temp_dir cache, m))
    | Ok tmp -> (
        match transport ~url:pin.url ~dest:tmp with
        | Error m ->
            (try Sys.remove tmp with Sys_error _ -> ());
            Err.fail (`Transport (pin.url, m))
        | Ok () -> Cache.promote ?hash ~layer cache ~tmp pin
        | exception e ->
            (try Sys.remove tmp with Sys_error _ -> ());
            Err.fail (`Transport (pin.url, Printexc.to_string e)))
  in
  match (Err.payload (Cache.lookup ?hash ~layer cache pin), transport) with
  | Ok (Cache.Present path), _ -> Err.return path
  | Ok Cache.Absent, Some transport -> download transport
  | Ok Cache.Absent, None -> Err.fail (`Offline pin.name)
  | Error _, Some transport -> download transport
  | Error _, None ->
      (* Offline: report what is wrong with the cached file. *)
      let* _ = Cache.lookup ?hash ~layer cache pin in
      Err.fail (`Offline pin.name)
