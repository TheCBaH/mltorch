(* The witness check: every block placed exactly once, inside the pool, and no
   two blocks live at the same time overlapping. Shared by the public [check]
   and by the reference search, which validates its own candidates. *)

module Block = struct
  type 'k t = { key : 'k; offset : int64; size : int64 }
end

module Out_of_pool = struct
  type 'k t = { block : 'k Block.t; pool : int64 }
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
  let* () =
    let rec bounds i =
      if i >= n then Ok ()
      else
        let offset = offset_of i in
        if Int64.compare offset 0L < 0 then
          Err.fail ~pos:__POS__ (`Negative_offset keys.(i))
        else
          match Ia_script.add_checked offset sizes.(i) with
          | None -> Err.fail ~pos:__POS__ (`Offset_overflow keys.(i))
          | Some hi ->
              if Int64.compare hi pool > 0 then
                Err.fail ~pos:__POS__
                  (`Out_of_pool { Out_of_pool.block = block i; pool })
              else bounds (i + 1)
    in
    bounds 0
  in
  let disjoint a b =
    let lo_a = offset_of a and lo_b = offset_of b in
    Int64.compare (Int64.add lo_a sizes.(a)) lo_b <= 0
    || Int64.compare (Int64.add lo_b sizes.(b)) lo_a <= 0
  in
  let rec overlaps a b =
    if a >= n then Ok ()
    else if b >= n then overlaps (a + 1) (a + 2)
    else if
      sizes.(a) <> 0L
      && sizes.(b) <> 0L
      && Ia_script.conflicts script a b
      && not (disjoint a b)
    then
      Err.fail ~pos:__POS__
        (`Overlap { Overlap.first = block a; second = block b })
    else overlaps a (b + 1)
  in
  let* () = overlaps 0 1 in
  Ok solution
