(* Verifies that native_functions.yaml can be fully parsed into raw records.
   Prints summary stats and spot-checks a few known entries. *)

let () =
  if Array.length Sys.argv < 2 then (
    Printf.eprintf "Usage: aten_schema_raw_test <native_functions.yaml>\n";
    exit 1);
  let path = Sys.argv.(1) in
  let yaml = In_channel.with_open_bin path In_channel.input_all in
  match Aten_raw.of_yaml_string yaml with
  | Error (`Msg e) ->
      Printf.eprintf "YAML parse error: %s\n" e;
      exit 1
  | Ok entries ->
      Printf.printf "total entries: %d\n" (List.length entries);
      Printf.printf "with dispatch:  %d\n"
        (List.length (List.filter (fun e -> e.Aten_raw.dispatch <> []) entries));
      Printf.printf "structured:     %d\n"
        (List.length (List.filter (fun e -> e.Aten_raw.structured) entries));
      Printf.printf "with tags:      %d\n"
        (List.length (List.filter (fun e -> e.Aten_raw.tags <> []) entries));

      (* Spot-check: look up entries by their func prefix *)
      let find prefix =
        List.find_opt
          (fun e ->
            let n = String.length prefix in
            String.length e.Aten_raw.func >= n
            && String.sub e.Aten_raw.func 0 n = prefix)
          entries
      in
      let show label prefix =
        match find prefix with
        | None -> Printf.printf "%-20s NOT FOUND\n" label
        | Some e ->
            Printf.printf "%-20s dispatch=%d structured=%-5b tags=[%s]\n" label
              (List.length e.Aten_raw.dispatch)
              e.Aten_raw.structured
              (String.concat ","
                 (List.map Aten_raw.Tag.to_string e.Aten_raw.tags))
      in
      show "conv2d" "conv2d(";
      show "relu_" "relu_(";
      show "abs.out" "abs.out(";
      show "add.out" "add.out(";
      show "batch_norm" "batch_norm(";
      show "chunk" "chunk(";
      show "split.Tensor" "split.Tensor(";
      show "abs" "abs("
