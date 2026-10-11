(* Lenient access to a parsed document: pick one member and decode it with a
   strict decoder. Used to read the few manifest fields replay needs without
   restating the whole manifest schema. *)

open Inference_compare_schema

let member name (j : Wire.json) =
  match j with
  | Jsont.Object (mems, _) ->
      List.find_map
        (fun ((n, _), v) -> if String.equal n name then Some v else None)
        mems
  | _ -> None

let decode_member name decoder j =
  match member name j with
  | None -> Error (Format.asprintf "missing member %S" name)
  | Some v -> (
      match decoder v with
      | Ok x -> Ok x
      | Error e ->
          Error (Format.asprintf "%S: %a" name Wire.pp_error (Err.Error.kind e))
      )
