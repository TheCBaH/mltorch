(* The safetensors inference probe with the checkpoint downloaded in the page or
   in node by hf-hub's JavaScript driver, rather than handed over as a file.

   The model files and the input arrive as strings and bytes, because a browser
   has no file system; node's and the browser's drivers differ only in how they
   obtain them. Everything after the download is [Probe_pt2], the source of the
   native golden, so the output is diffed against it line for line.

   [globalThis.safetensorsProbe(options, callback)]: [options] holds the strings
   [program], [weights], [constants] (or null), [map], the [input] Uint8Array,
   and the host's [cacheDir] and [endpoint]. [callback] gets null on success, or
   a message. *)

open Js_of_ocaml

let get options name = Js.Unsafe.get options (Js.string name)
let string options name = Js.to_string (get options name)

let run options finish =
  let files =
    {
      Probe_pt2.Safetensors_files.program = string options "program";
      weights = string options "weights";
      constants =
        Js.Opt.to_option (Js.Opt.return (get options "constants"))
        |> Option.map Js.to_string;
      map = string options "map";
    }
  in
  let host =
    {
      Pt2_safetensors_js.Host.cache_dir = string options "cacheDir";
      endpoint = string options "endpoint";
    }
  in
  let map =
    Err.or_raise ~pp_error:Pt2_safetensors.pp_error
      (Pt2_safetensors.map_of_string files.map)
  in
  Pt2_safetensors_js.fetch host map.source (fun result ->
      finish (fun () ->
          let checkpoint =
            Err.or_raise ~pp_error:Pt2_safetensors_js.pp_error result
          in
          let input = Typed_array.String.of_uint8Array (get options "input") in
          Probe_pt2.run_safetensors_strings files ~checkpoint
            ~input_name:"input.pt" ~input))

(* A boundary that emits outward prints the payload alone: the default
   rendering of [Err.Exn.E] includes the detection stack. *)
let message = function
  | Err.Exn.E packed -> Format.asprintf "%a" Err.Exn.pp_kind packed
  | e -> Printexc.to_string e

let () =
  Js.Unsafe.set Js.Unsafe.global "safetensorsProbe"
    (Js.wrap_callback (fun options callback ->
         let report = function
           | None -> Js.null
           | Some e -> Js.some (Js.string (message e))
         in
         let finish body =
           let outcome =
             match body () with () -> None | exception e -> Some e
           in
           (* The program's main has returned by now, so nothing flushes at
              exit: the last buffered lines would be lost. *)
           flush stdout;
           ignore
             (Js.Unsafe.fun_call callback
                [| Js.Unsafe.inject (report outcome) |])
         in
         try run options finish
         with e ->
           ignore
             (Js.Unsafe.fun_call callback
                [| Js.Unsafe.inject (report (Some e)) |])))
