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
  let n = Ia_script.blocks script in
  let keys = script.Ia_script.keys and sizes = script.Ia_script.sizes in
  let offsets = Array.make n None in
  let ( let* ) = Result.bind in
  let* () =
    Err.List.iter
      (fun (k, offset) ->
        match Ia_script.find_index script.Ia_script.equal keys n k with
        | None -> Err.fail ~pos:__POS__ (`Unknown_key k)
        | Some i -> (
            match offsets.(i) with
            | Some _ -> Err.fail ~pos:__POS__ (`Duplicate_placement k)
            | None ->
                offsets.(i) <- Some offset;
                Ok ()))
      (Ia_solution.placements solution)
  in
  let pool = Ia_solution.pool solution in
  let offset_of i =
    match offsets.(i) with Some o -> o | None -> assert false
  in
  let block i =
    { Block.key = keys.(i); offset = offset_of i; size = sizes.(i) }
  in
  let* () =
    let rec unplaced i =
      if i >= n then Ok ()
      else if offsets.(i) = None then Err.fail ~pos:__POS__ (`Unplaced keys.(i))
      else unplaced (i + 1)
    in
    unplaced 0
  in
  (* Every block's end, once it is known to fit. *)
  let ends = Array.make n Byte_offset.zero in
  let* () =
    let rec bounds i =
      if i >= n then Ok ()
      else
        let offset = offset_of i
        and alignment = script.Ia_script.alignments.(i) in
        if not (Byte_offset.is_aligned offset alignment) then
          Err.fail ~pos:__POS__
            (`Misaligned { Misaligned.block = block i; alignment })
        else
          let* hi =
            Byte_offset.advance offset sizes.(i)
            |> Err.map_error ~pos:__POS__ (fun (`Quantity_overflow _) ->
                `Offset_overflow keys.(i))
          in
          if Byte_size.compare (Byte_offset.to_size hi) pool > 0 then
            Err.fail ~pos:__POS__
              (`Out_of_pool { Out_of_pool.block = block i; pool })
          else begin
            ends.(i) <- hi;
            bounds (i + 1)
          end
    in
    bounds 0
  in
  let disjoint a b =
    Byte_offset.compare ends.(a) (offset_of b) <= 0
    || Byte_offset.compare ends.(b) (offset_of a) <= 0
  in
  let rec overlaps a b =
    if a >= n then Ok ()
    else if b >= n then overlaps (a + 1) (a + 2)
    else if
      (not (Ia_script.empty script a))
      && (not (Ia_script.empty script b))
      && Ia_script.conflicts script a b
      && not (disjoint a b)
    then
      Err.fail ~pos:__POS__
        (`Overlap { Overlap.first = block a; second = block b })
    else overlaps a (b + 1)
  in
  let* () = overlaps 0 1 in
  Ok solution
