(* The witness check: every block placed exactly once, aligned, inside the
   pool, and no two blocks live at the same time overlapping. Shared by the
   public [check] and by the reference search, which validates its own
   candidates. *)

open Core.Storage_units

module Block = struct
  type 'k t = { key : 'k; offset : Byte_offset.t; size : Byte_size.t }
end

module Misaligned = struct
  type 'k t = { block : 'k Block.t; alignment : Byte_alignment.t }
end

module Out_of_pool = struct
  type 'k t = { block : 'k Block.t; pool : Byte_size.t }
end

module Overlap = struct
  type 'k t = { first : 'k Block.t; second : 'k Block.t }
end

let check (script : _ Ia_script.t) solution =
  let get = Ia_script.get and set = Ia_script.set in
  let indices = Array.to_list (Ia_script.indices script) in
  let key = Ia_script.key script and size = Ia_script.size script in
  let offsets = Ia_script.table script None in
  let ( let* ) = Result.bind in
  let* () =
    Err.List.iter
      (fun (k, offset) ->
        match Ia_script.find script k with
        | None -> Err.fail ~pos:__POS__ (`Unknown_key k)
        | Some i -> (
            match get offsets i with
            | Some _ -> Err.fail ~pos:__POS__ (`Duplicate_placement k)
            | None ->
                set offsets i (Some offset);
                Ok ()))
      (Ia_solution.placements solution)
  in
  let pool = Ia_solution.pool solution in
  let offset_of i =
    match get offsets i with Some o -> o | None -> assert false
  in
  let block i = { Block.key = key i; offset = offset_of i; size = size i } in
  let* () =
    Err.List.iter
      (fun i ->
        if get offsets i = None then Err.fail ~pos:__POS__ (`Unplaced (key i))
        else Ok ())
      indices
  in
  (* Every block's end, once it is known to fit. *)
  let ends = Ia_script.table script Byte_offset.zero in
  let* () =
    Err.List.iter
      (fun i ->
        let offset = offset_of i and alignment = Ia_script.alignment script i in
        if not (Byte_offset.is_aligned offset alignment) then
          Err.fail ~pos:__POS__
            (`Misaligned { Misaligned.block = block i; alignment })
        else
          let* hi =
            Byte_offset.advance offset (size i)
            |> Err.map_error ~pos:__POS__ (fun (`Quantity_overflow _) ->
                `Offset_overflow (key i))
          in
          if Byte_size.compare (Byte_offset.to_size hi) pool > 0 then
            Err.fail ~pos:__POS__
              (`Out_of_pool { Out_of_pool.block = block i; pool })
          else begin
            set ends i hi;
            Ok ()
          end)
      indices
  in
  let disjoint a b =
    Byte_offset.compare (get ends a) (offset_of b) <= 0
    || Byte_offset.compare (get ends b) (offset_of a) <= 0
  in
  let* () =
    Err.List.iter
      (fun a ->
        Err.List.iter
          (fun b ->
            if
              Ia_script.Index.compare a b < 0
              && (not (Ia_script.empty script a))
              && (not (Ia_script.empty script b))
              && Ia_script.conflicts script a b
              && not (disjoint a b)
            then
              Err.fail ~pos:__POS__
                (`Overlap { Overlap.first = block a; second = block b })
            else Ok ())
          indices)
      indices
  in
  Ok solution
