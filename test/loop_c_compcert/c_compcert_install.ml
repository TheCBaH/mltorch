(* Every [Loop_check.run] in this library, the copied fixtures and the op sweep
   included, also runs the Compcert_scalar C compiled with the host compiler and
   compares it with the reference, after checking that the text is what an
   embedded CompCert accepts: no preprocessor directive, none of the header
   macros the dialect writes out. [installed] exists so a test module can name
   this one and keep it linked. *)

let macros =
  [
    "FLT_";
    "INFINITY";
    "INT64_C";
    "MODEL_ERROR_WORDS";
    "NAN";
    "isfinite";
    "signbit";
  ]

let contains text word =
  let n = String.length text and k = String.length word in
  let rec at i = i + k <= n && (String.sub text i k = word || at (i + 1)) in
  at 0

let leak text =
  let directive l = String.length l > 0 && l.[0] = '#' in
  match List.find_opt directive (String.split_on_char '\n' text) with
  | Some l -> Some ("directive " ^ l)
  | None -> List.find_opt (contains text) macros

let executor : Loop_ir.Loop_check.Executor.t =
 fun p ~bind ->
  match
    Loop_c_exec.unit_text ~dialect:Loop_ir.Loop_c_dialect.Compcert_scalar p
  with
  | Ok (_, text) when leak text <> None ->
      Err.fail
        (`Js_exception ("Compcert_scalar text leaks " ^ Option.get (leak text)))
  | _ -> Loop_c_exec.executor_compcert p ~bind

let () = Loop_ir.Loop_check.install (Some executor)
let installed = true
