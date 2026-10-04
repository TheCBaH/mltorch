open Js_of_ocaml

module Host = struct
  type t = { cache_dir : string; endpoint : string }
end

type error = [ Pt2_safetensors.error | `Hub of string ]

let pp_error ppf : error -> unit = function
  | #Pt2_safetensors.error as e -> Pt2_safetensors.pp_error ppf e
  | `Hub msg -> Fmt.pf ppf "hub download failed: %s" msg

(* The host's copy of a downloaded file, which is what the driver's own result
   does not carry: it names a path, the host owns the bytes behind it. *)
let host_bytes path =
  let host = Js.Unsafe.get Js.Unsafe.global "hfHubHost" in
  Js.Unsafe.meth_call host "bytes" [| Js.Unsafe.inject (Js.string path) |]
  |> Typed_array.String.of_uint8Array

let fetch (host : Host.t) (source : Pt2_safetensors.Checkpoint_map.Source.t)
    callback =
  let args =
    [|
      host.cache_dir;
      host.endpoint;
      "false";
      "";
      source.repo_id;
      source.filename;
      source.revision;
      "model";
    |]
  in
  Driver.download args (fun reply ->
      callback
        (let open Err.Syntax in
         match Array.to_list reply with
         | [ "ok"; path; _commit; etag ] ->
             let bytes = host_bytes path in
             let* () =
               Pt2_safetensors.check_source source
                 ~etag:(if etag = "" then None else Some etag)
                 ~size:(Int64.of_int (String.length bytes))
             in
             Err.return bytes
         | [ "error"; message ] ->
             Err.import ~pos:__POS__ (fun m -> `Hub m) (Error message)
         | _ ->
             Err.import ~pos:__POS__
               (fun m -> `Hub m)
               (Error "unexpected reply from the JavaScript driver")))
