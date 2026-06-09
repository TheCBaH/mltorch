let () =
  if Array.length Sys.argv < 2 then (
    Printf.eprintf "Usage: aten_schema_func_test <native_functions.yaml>\n";
    exit 1);
  let path = Sys.argv.(1) in
  let yaml = In_channel.with_open_bin path In_channel.input_all in
  match Raw.of_yaml_string yaml with
  | Error (`Msg e) ->
      Printf.eprintf "YAML parse error: %s\n" e;
      exit 1
  | Ok entries ->
      let total = List.length entries in
      let errors = ref 0 in
      List.iter
        (fun (e : Raw.t) ->
          match Func_schema.parse e.func with
          | Ok _ -> ()
          | Error msg ->
              incr errors;
              Printf.eprintf "FAIL: %s\n  error: %s\n" e.func msg)
        entries;
      Printf.printf "total: %d\n" total;
      Printf.printf "errors: %d\n" !errors
