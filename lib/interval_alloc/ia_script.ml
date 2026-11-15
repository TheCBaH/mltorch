(* A validated alloc/free script, stored by allocation order: block [i] is the
   [i]-th [Alloc], live over the half-open event positions [alloc_at.(i),
   free_at.(i)). An [Alloc] at position [p] and a [Free] at [q > p] make the
   block live during [p, q); a block never freed lives to [length]. Two blocks
   conflict when those ranges intersect, so an operand freed only after its
   consumer's output was allocated conflicts with it. *)

open Core.Storage_units

module Event = struct
  type 'k t =
    | Alloc of { key : 'k; size : Byte_size.t; alignment : Byte_alignment.t }
    | Free of 'k
end

(* A block's place in allocation order: the index of every per-block array. *)
module Index =
  Core.Tagged_int.Make
    (struct
      let prefix = "b"
    end)
    ()

let get a (i : Index.t) = a.((i :> int))
let set a (i : Index.t) v = a.((i :> int)) <- v

(* A position in the event list. *)
module Position =
  Core.Tagged_int.Make
    (struct
      let prefix = "@"
    end)
    ()

type 'k t = {
  equal : 'k -> 'k -> bool;
  events : 'k Event.t list;
  keys : 'k array;
  sizes : Byte_size.t array;
  alignments : Byte_alignment.t array;
  alloc_at : Position.t array;
  free_at : Position.t array;
  length : Position.t;
  indices : Index.t array;  (** every block, in allocation order *)
}

(* A failure the caller has already ruled out, such as the end of a block
   the checker has placed inside its pool. *)
let invariant r = Err.or_raise ~pp_error r

(* The first of the [upto] blocks allocated so far whose key is [k]. *)
let find_index equal keys ~upto k =
  let rec go i =
    if i >= (upto : Index.Next.t :> int) then None
    else if equal keys.(i) k then Some (Index.of_int i)
    else go (i + 1)
  in
  go 0

let find t k =
  find_index t.equal t.keys ~upto:(Index.Next.of_int (Array.length t.keys)) k

let validate ~equal events =
  let length = List.length events in
  let allocs =
    List.fold_left
      (fun n -> function Event.Alloc _ -> n + 1 | Event.Free _ -> n)
      0 events
  in
  (* Neither a key nor an alignment has a default, so both arrays are made
     from the first block. *)
  let keys = ref [||] and alignments = ref [||] in
  let sizes = Array.make allocs Byte_size.zero in
  let alloc_at = Array.make allocs (Position.of_int 0)
  and free_at = Array.make allocs (Position.of_int length) in
  let freed = Array.make allocs false in
  let count = ref Index.Next.first in
  let rec go pos = function
    | [] -> Ok ()
    | Event.Alloc { key; size; alignment } :: rest ->
        if find_index equal !keys ~upto:!count key <> None then
          Err.fail ~pos:__POS__ (`Double_alloc key)
        else begin
          if Index.Next.equal !count Index.Next.first then begin
            keys := Array.make allocs key;
            alignments := Array.make allocs alignment
          end;
          let i, next = Index.Next.alloc !count in
          set !keys i key;
          set sizes i size;
          set !alignments i alignment;
          set alloc_at i pos;
          count := next;
          go (Position.succ pos) rest
        end
    | Event.Free key :: rest -> (
        match find_index equal !keys ~upto:!count key with
        | None -> Err.fail ~pos:__POS__ (`Free_unknown key)
        | Some i ->
            if get freed i then Err.fail ~pos:__POS__ (`Double_free key)
            else begin
              set freed i true;
              set free_at i pos;
              go (Position.succ pos) rest
            end)
  in
  Result.map
    (fun () ->
      {
        equal;
        events;
        keys = !keys;
        sizes;
        alignments = !alignments;
        alloc_at;
        free_at;
        length = Position.of_int length;
        indices = Array.init allocs Index.of_int;
      })
    (go (Position.of_int 0) events)

let events t = t.events
let indices t = t.indices
let key t i = get t.keys i
let size t i = get t.sizes i
let alignment t i = get t.alignments i
let empty t i = Byte_size.equal (size t i) Byte_size.zero

(* A table with one cell per block, read and written by [get] and
   [set]. *)
let table t cell = Array.make (Array.length t.indices) cell

(* How many events a block is live for: a score, not a quantity. *)
let lifetime t i =
  Int64.of_int
    ((get t.free_at i : Position.t :> int)
    - (get t.alloc_at i : Position.t :> int))

(* Two blocks may not share cells iff their live ranges intersect. *)
let conflicts t a b =
  Position.compare (get t.alloc_at a) (get t.free_at b) < 0
  && Position.compare (get t.alloc_at b) (get t.free_at a) < 0

let lower_bound t =
  let rec go live best = function
    | [] -> Ok best
    | Event.Alloc { key; size; _ } :: rest ->
        Result.bind
          (Byte_size.add live size
          |> Err.map_error ~pos:__POS__ (fun (`Quantity_overflow _) ->
              `Live_overflow key))
          (fun live -> go live (Byte_size.max best live) rest)
    | Event.Free key :: rest ->
        let i = match find t key with Some i -> i | None -> assert false in
        (* [live] holds the block: validation frees only what it allocated. *)
        go (invariant (Byte_size.sub live (size t i))) best rest
  in
  go Byte_size.zero Byte_size.zero t.events
