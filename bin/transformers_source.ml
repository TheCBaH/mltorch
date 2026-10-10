open Transformers_metadata.Source

let run () =
  match Array.to_list Sys.argv with
  | [ _; "inventory"; source; flat; output ] -> inventory source flat output
  | [ _; "summary"; inventory; admission; output; pin; consumer; changes ] ->
      summary inventory admission output pin consumer changes
  | _ ->
      bad
        "usage: transformers_source inventory SOURCE FLAT OUTPUT | summary \
         INVENTORY ADMISSION OUTPUT PIN CONSUMER CHANGES"

let () =
  let result =
    try run () with
    | Sys_error e -> bad e
    | Unix.Unix_error (e, op, path) ->
        bad (op ^ " " ^ path ^ ": " ^ Unix.error_message e)
  in
  match result with
  | Ok () -> ()
  | Error e ->
      Fmt.epr "transformers_source: %s@."
        (match Err.Error.kind e with `Source s -> s);
      exit 2
