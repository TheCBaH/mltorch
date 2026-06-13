(* Coverage of the binding generator over the full native_functions.yaml:
   how many ops the current gated type-support can emit, and why the rest are
   skipped (bucketed by reason / unsupported type). *)

let () =
  if Array.length Sys.argv < 2 then (
    Printf.eprintf "Usage: aten_gen_coverage <native_functions.yaml>\n";
    exit 1);
  let yaml = In_channel.with_open_bin Sys.argv.(1) In_channel.input_all in
  match Raw.of_yaml_string yaml with
  | Error (`Msg e) ->
      Printf.eprintf "YAML parse error: %s\n" e;
      exit 1
  | Ok entries ->
      let total = List.length entries in
      let generated = ref 0 and skipped = ref 0 in
      let reasons : (string, int) Hashtbl.t = Hashtbl.create 64 in
      let bump k =
        Hashtbl.replace reasons k
          (1 + try Hashtbl.find reasons k with Not_found -> 0)
      in
      List.iter
        (fun (e : Raw.t) ->
          match Func_schema.parse e.func with
          | Error _ -> ()
          | Ok op -> (
              match Aten_gen.Gen.generate op with
              | Generated _ -> incr generated
              | Skipped r ->
                  incr skipped;
                  bump r))
        entries;
      Printf.printf "total: %d\ngenerated: %d\nskipped: %d\n" total !generated
        !skipped;
      Printf.printf "top skip reasons:\n";
      Hashtbl.fold (fun k v acc -> (k, v) :: acc) reasons []
      |> List.sort (fun (a, na) (b, nb) ->
          if nb <> na then compare nb na else compare a b)
      |> List.filteri (fun i _ -> i < 12)
      |> List.iter (fun (k, v) -> Printf.printf "  %4d  %s\n" v k)
