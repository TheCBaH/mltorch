(* Turns an order of blocks into offsets: each block, in that order, goes into
   the first (or best) gap between the already-placed blocks it conflicts
   with that holds it at its alignment. *)

open Core.Storage_units
module Script = Ia_script

module Mode = struct
  type t = Best_fit | First_fit
end

(* A decoded order: offsets and ends by block, and the pool they need. *)
module Placed = struct
  type t = {
    offsets : Byte_offset.t array;
    ends : Byte_offset.t array;
    pool : Byte_size.t;
  }
end

(* The already-placed, non-empty blocks that conflict with [b], as
   (offset, end) sorted by offset. *)
let busy script offsets ends placed b =
  let acc = ref [] in
  for a = 0 to Script.blocks script - 1 do
    if
      placed.(a) && (not (Script.empty script a)) && Script.conflicts script a b
    then acc := (offsets.(a), ends.(a)) :: !acc
  done;
  List.sort (fun (a, _) (b, _) -> Byte_offset.compare a b) !acc

(* The lowest (first fit) or tightest (best fit) aligned start that holds
   [size] bytes below the next busy range, else the aligned start past them
   all. *)
let place mode size alignment ranges =
  let ( let* ) = Result.bind in
  let rec go cursor best = function
    | [] -> (
        match best with
        | Some (_, offset) -> Ok offset
        | None -> Byte_offset.align_up cursor alignment)
    | (lo, hi) :: rest -> (
        let next = Byte_offset.max cursor hi in
        let* start = Byte_offset.align_up cursor alignment in
        let* stop = Byte_offset.advance start size in
        if Byte_offset.compare stop lo > 0 then go next best rest
        else
          let gap = Script.invariant (Byte_offset.distance ~from:start lo) in
          match (mode, best) with
          | Mode.First_fit, _ -> Ok start
          | Mode.Best_fit, Some (g, _) when Byte_size.compare gap g >= 0 ->
              go next best rest
          | Mode.Best_fit, _ -> go next (Some (gap, start)) rest)
  in
  go Byte_offset.zero None ranges

(* [order] must be a permutation of the block indices. *)
let offsets mode script order =
  let ( let* ) = Result.bind in
  let n = Script.blocks script in
  let offsets = Array.make n Byte_offset.zero
  and ends = Array.make n Byte_offset.zero
  and placed = Array.make n false in
  let rec go i pool =
    if i >= n then Ok { Placed.offsets; ends; pool }
    else
      let b = order.(i) in
      let size = script.Script.sizes.(b) in
      let* offset, stop =
        (if Script.empty script b then Ok (Byte_offset.zero, Byte_offset.zero)
         else
           let* offset =
             place mode size
               script.Script.alignments.(b)
               (busy script offsets ends placed b)
           in
           let* stop = Byte_offset.advance offset size in
           Ok (offset, stop))
        |> Err.map_error ~pos:__POS__ (fun (`Quantity_overflow _) ->
            `Pool_overflow script.Script.keys.(b))
      in
      offsets.(b) <- offset;
      ends.(b) <- stop;
      placed.(b) <- true;
      go (i + 1) (Byte_size.max pool (Byte_offset.to_size stop))
  in
  go 0 Byte_size.zero

let solution script { Placed.offsets; pool; _ } =
  Ia_solution.Unsafe.make ~pool
    (List.init (Script.blocks script) (fun i ->
         (script.Script.keys.(i), offsets.(i))))

(* [size * lifetime], saturating: only the ordering matters. *)
let area size life =
  if life = 0L then 0L
  else if Int64.compare size (Int64.div Int64.max_int life) > 0 then
    Int64.max_int
  else Int64.mul size life

(* A strategy's block order and placement rule. Largest key first; ties keep
   allocation order. A key is a heuristic score, not a quantity, so a size
   leaves its type here to be scored. *)
let plan strategy script =
  let size i = Byte_size.to_int64 script.Script.sizes.(i) in
  let life i =
    Int64.of_int (script.Script.free_at.(i) - script.Script.alloc_at.(i))
  in
  let key =
    match strategy with
    | Ia_strategy.Greedy_by_area -> fun i -> area (size i) (life i)
    | Greedy_by_lifetime -> life
    | Greedy_by_size | Greedy_by_size_best_fit -> size
  in
  let order =
    List.init (Script.blocks script) Fun.id
    |> List.stable_sort (fun a b -> Int64.compare (key b) (key a))
    |> Array.of_list
  in
  let mode =
    match strategy with
    | Greedy_by_size_best_fit -> Mode.Best_fit
    | Greedy_by_area | Greedy_by_lifetime | Greedy_by_size -> Mode.First_fit
  in
  (mode, order)
