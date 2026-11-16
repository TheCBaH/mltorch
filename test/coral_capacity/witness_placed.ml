(* Each baseline profile loaded from the release (its simulator's SHA-256
   verified), both witness images placed by the real layout. Exits 1 on a
   refusal of the profile; a refusal of an image is printed, not fatal. *)

open Coral_capacity

let code = Code_bytes.of_int64 4_960L

let () =
  let dir = Sys.argv.(1) in
  List.iter
    (fun name ->
      match Err.payload (Coral_profile.load ~dir name) with
      | Error e ->
          Fmt.epr "%a@." Coral_profile.pp_error e;
          exit 1
      | Ok p ->
          Fmt.pr "%s@." name;
          List.iter
            (fun (image : Image.t) ->
              match Err.payload (Capacity.place p image) with
              | Ok (placed : Capacity.Placed.t) ->
                  Fmt.pr "  %s@." image.name;
                  List.iter
                    (fun (n, (s : Coral_layout.Placed_slot.t)) ->
                      Fmt.pr "    %-8s 0x%08Lx %9Ld %s@." n s.address
                        s.slot.size s.region)
                    placed.slots
              | Error e ->
                  Fmt.pr "  %s: refused: %a@." image.name Capacity.pp_error e)
            (Witness.all ~code))
    Coral_profile.baseline
