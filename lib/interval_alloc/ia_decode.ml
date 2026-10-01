(* Turns an order of blocks into offsets: each block, in that order, goes into
   a gap between the already-placed blocks it conflicts with. *)

module Script = Ia_script

module Mode = struct
  type t = Best_fit | First_fit
end

(* The already-placed, non-empty blocks that conflict with [b], as
   (offset, end) sorted by offset. *)
let busy script offsets placed b =
  let acc = ref [] in
  for a = 0 to Script.blocks script - 1 do
    if
      placed.(a) && script.Script.sizes.(a) <> 0L && Script.conflicts script a b
    then
      acc :=
        (offsets.(a), Int64.add offsets.(a) script.Script.sizes.(a)) :: !acc
  done;
  List.sort (fun (a, _) (b, _) -> Int64.compare a b) !acc

let place mode size ranges =
  let rec go cursor best = function
    | [] -> ( match best with Some (_, offset) -> offset | None -> cursor)
    | (lo, hi) :: rest -> (
        let gap = Int64.sub lo cursor in
        let next = Int64.max cursor hi in
        if Int64.compare gap size < 0 then go next best rest
        else
          match (mode, best) with
          | Mode.First_fit, _ -> cursor
          | Mode.Best_fit, Some (g, _) when Int64.compare gap g >= 0 ->
              go next best rest
          | Mode.Best_fit, _ -> go next (Some (gap, cursor)) rest)
  in
  go 0L None ranges

(* [order] must be a permutation of the block indices. Offsets by block, and
   the pool they need. *)
let offsets mode script order =
  let n = Script.blocks script in
  let offsets = Array.make n 0L and placed = Array.make n false in
  let pool = ref 0L in
  let rec go i =
    if i >= n then Ok (offsets, !pool)
    else
      let b = order.(i) in
      let size = script.Script.sizes.(b) in
      let offset =
        if size = 0L then 0L else place mode size (busy script offsets placed b)
      in
      match Script.add_checked offset size with
      | None -> Err.fail ~pos:__POS__ (`Pool_overflow script.Script.keys.(b))
      | Some hi ->
          offsets.(b) <- offset;
          placed.(b) <- true;
          pool := Int64.max !pool hi;
          go (i + 1)
  in
  go 0

let solution script (offsets, pool) =
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
   allocation order. *)
let plan strategy script =
  let size i = script.Script.sizes.(i) in
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
