(* A validated alloc/free script, stored by allocation order: block [i] is the
   [i]-th [Alloc], live over the half-open event positions [alloc_at.(i),
   free_at.(i)). An [Alloc] at position [p] and a [Free] at [q > p] make the
   block live during [p, q); a block never freed lives to [length]. Two blocks
   conflict when those ranges intersect, so an operand freed only after its
   consumer's output was allocated conflicts with it. *)

module Event = struct
  type 'k t = Alloc of { key : 'k; size : int64 } | Free of 'k
end

module Negative_size = struct
  type 'k t = { key : 'k; size : int64 }
end

type 'k t = {
  equal : 'k -> 'k -> bool;
  events : 'k Event.t list;
  keys : 'k array;
  sizes : int64 array;
  alloc_at : int array;
  free_at : int array;
  length : int;
}

let find_index equal keys count k =
  let rec go i =
    if i >= count then None else if equal keys.(i) k then Some i else go (i + 1)
  in
  go 0

let validate ~equal events =
  let length = List.length events in
  let allocs =
    List.fold_left
      (fun n -> function Event.Alloc _ -> n + 1 | Event.Free _ -> n)
      0 events
  in
  let keys = ref [||] and sizes = Array.make allocs 0L in
  let alloc_at = Array.make allocs 0 and free_at = Array.make allocs length in
  let freed = Array.make allocs false in
  let count = ref 0 in
  let rec go pos = function
    | [] -> Ok ()
    | Event.Alloc { key; size } :: rest ->
        if Int64.compare size 0L < 0 then
          Err.fail ~pos:__POS__ (`Negative_size { Negative_size.key; size })
        else if find_index equal !keys !count key <> None then
          Err.fail ~pos:__POS__ (`Double_alloc key)
        else begin
          if !count = 0 then keys := Array.make allocs key;
          !keys.(!count) <- key;
          sizes.(!count) <- size;
          alloc_at.(!count) <- pos;
          incr count;
          go (pos + 1) rest
        end
    | Event.Free key :: rest -> (
        match find_index equal !keys !count key with
        | None -> Err.fail ~pos:__POS__ (`Free_unknown key)
        | Some i ->
            if freed.(i) then Err.fail ~pos:__POS__ (`Double_free key)
            else begin
              freed.(i) <- true;
              free_at.(i) <- pos;
              go (pos + 1) rest
            end)
  in
  Result.map
    (fun () ->
      { equal; events; keys = !keys; sizes; alloc_at; free_at; length })
    (go 0 events)

let events t = t.events
let blocks t = Array.length t.sizes

(* Two blocks may not share cells iff their live ranges intersect. *)
let conflicts t a b =
  t.alloc_at.(a) < t.free_at.(b) && t.alloc_at.(b) < t.free_at.(a)

let add_checked a b =
  let s = Int64.add a b in
  if Int64.compare s a < 0 then None else Some s

let lower_bound t =
  let rec go live best = function
    | [] -> Ok best
    | Event.Alloc { key; size } :: rest -> (
        match add_checked live size with
        | None -> Err.fail ~pos:__POS__ (`Live_overflow key)
        | Some live -> go live (Int64.max best live) rest)
    | Event.Free key :: rest ->
        let i =
          match find_index t.equal t.keys (blocks t) key with
          | Some i -> i
          | None -> assert false
        in
        go (Int64.sub live t.sizes.(i)) best rest
  in
  go 0L 0L t.events
