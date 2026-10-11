(* fixture_gen DIR SCENARIO [--retention MODE] [--required] [--disk-bytes N]
   [--retain-unit U] [--donor-right] writes a complete experiment directory. *)

module G = Inference_compare_test.Experiment_gen

let () =
  let args = Array.to_list Sys.argv |> List.tl in
  let dir, scenario, rest =
    match args with
    | d :: s :: r -> (d, s, r)
    | _ ->
        prerr_endline "usage";
        exit 2
  in
  let rec opt name = function
    | n :: v :: _ when n = name -> Some v
    | _ :: tl -> opt name tl
    | [] -> None
  in
  let flag name = List.mem name rest in
  G.generate ~dir ~scenario
    {
      retention =
        (match opt "--retention" rest with
        | Some "all" -> All
        | Some "first_mismatch" -> First_mismatch
        | Some "selected" -> Selected_units
        | _ -> No_retention);
      required = flag "--required";
      disk_bytes =
        (match opt "--disk-bytes" rest with
        | Some n -> Int64.of_string n
        | None -> 1_000_000L);
      retain_unit = opt "--retain-unit" rest;
      donor_right = flag "--donor-right";
    }
