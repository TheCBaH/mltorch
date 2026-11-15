(* Order search: keep an order of blocks, decode it, and try a move; keep the
   move if the pool is no larger. One iteration is one move plus one decode.
   Nothing reads a clock: the same order, budget and seed give the same result,
   and the moves of iteration [i] do not depend on the budget, so a larger
   budget only extends the same walk. *)

open Core.Storage_units
module Script = Ia_script

module Stop = struct
  type t = Budget_exhausted | Lower_bound
end

module Effort = struct
  type t = { iterations : int64; stop : Stop.t }
end

module Budget = struct
  type t = { iterations : int64; seed : int64 }

  let create ~iterations ~seed = { iterations = Int64.max 0L iterations; seed }
end

(* An LCG on [int64]: identical on every backend. *)
let next s = Int64.add (Int64.mul s 6364136223846793005L) 1442695040888963407L

(* A place in an order of blocks, which is not a block's own index. *)
module Slot =
  Core.Tagged_int.Make
    (struct
      let prefix = "#"
    end)
    ()

let slot_get order (slot : Slot.t) = order.((slot :> int))
let slot_set order (slot : Slot.t) b = order.((slot :> int)) <- b

let swap order a b =
  let t = slot_get order a in
  slot_set order a (slot_get order b);
  slot_set order b t

(* Move the block at slot [src] to slot [dst]. *)
let move order src dst =
  let b = slot_get order src in
  let src = (src : Slot.t :> int) and dst' = (dst : Slot.t :> int) in
  if dst' < src then Array.blit order dst' order (dst' + 1) (src - dst')
  else Array.blit order (src + 1) order src (dst' - src);
  slot_set order dst b

(* Draws from the generator state [s], each typed by what it picks. The bound
   of a draw is the length of what it picks from, so it never leaves here. *)
module Draw : sig
  val before : int64 -> Slot.t -> Slot.t
  (** A slot below a slot other than the first. *)

  val element : int64 -> 'a array -> 'a
  (** One of a non-empty array. *)

  val one_in_four : int64 -> bool

  val slot : int64 -> 'a array -> Slot.t
  (** A slot of an order, which is non-empty. *)
end = struct
  let below s bound =
    Int64.to_int
      (Int64.rem (Int64.shift_right_logical s 33) (Int64.of_int bound))

  let before s (slot : Slot.t) = Slot.of_int (below s (slot :> int))
  let element s a = a.(below s (Array.length a))
  let one_in_four s = below s 4 = 0
  let slot s order = Slot.of_int (below s (Array.length order))
end

(* Blocks whose top edge sets the pool. *)
let peak_positions script order (placed : Ia_decode.Placed.t) =
  let acc = ref [] in
  Array.iteri
    (fun pos b ->
      if
        (not (Script.empty script b))
        && Byte_size.equal
             (Byte_offset.to_size (Script.get placed.Ia_decode.Placed.ends b))
             placed.pool
      then acc := Slot.of_int pos :: !acc)
    order;
  Array.of_list (List.rev !acc)

(* One walk to the largest of [checkpoints], reporting the best result at each
   of them (sorted, deduplicated) as a run with that budget would: the walk's
   iteration [i] does not depend on the budget, so every shorter run is a
   prefix of this one. A walk that reaches the lower bound stops, and every
   later checkpoint reports it. [at c] is called as checkpoint [c] is
   recorded, in increasing order. *)
let improve_at ?(at = ignore) mode ~checkpoints ~seed script ~lower_bound order
    (start : Ia_decode.Placed.t) =
  let checkpoints =
    List.sort_uniq Int64.compare (List.map (Int64.max 0L) checkpoints)
  in
  let last = List.fold_left Int64.max 0L checkpoints in
  let n = Array.length (Script.indices script) in
  let best = ref (Array.copy order, start) in
  let rng = ref seed and used = ref 0L in
  let pending = ref checkpoints and recorded = ref [] in
  let stop_of pool =
    if Byte_size.compare pool lower_bound <= 0 then Stop.Lower_bound
    else Stop.Budget_exhausted
  in
  let record ~iterations c =
    let _, placed = !best in
    at c;
    recorded :=
      ( c,
        placed,
        { Effort.iterations; stop = stop_of placed.Ia_decode.Placed.pool } )
      :: !recorded
  in
  let ( let* ) = Result.bind in
  let rec go () =
    (match !pending with
    | c :: rest when Int64.equal c !used ->
        record ~iterations:c c;
        pending := rest
    | _ -> ());
    let _, { Ia_decode.Placed.pool; _ } = !best in
    if
      n < 2
      || Byte_size.compare pool lower_bound <= 0
      || Int64.compare !used last >= 0
    then Ok ()
    else begin
      used := Int64.succ !used;
      let cur, cur_placed = !best in
      let cand = Array.copy cur in
      rng := next !rng;
      let peaks = peak_positions script cur cur_placed in
      let src =
        if Array.length peaks = 0 then Draw.slot !rng cand
        else Draw.element !rng peaks
      in
      rng := next !rng;
      (* A peak block moves earlier, so it is placed before what boxed it in. *)
      let dst =
        if Slot.equal src (Slot.of_int 0) then src else Draw.before !rng src
      in
      rng := next !rng;
      if Draw.one_in_four !rng then begin
        let a = Draw.slot !rng cand in
        rng := next !rng;
        let b = Draw.slot !rng cand in
        swap cand a b
      end
      else move cand src dst;
      let* placed = Ia_decode.offsets mode script cand in
      if
        Byte_size.compare placed.Ia_decode.Placed.pool
          cur_placed.Ia_decode.Placed.pool
        <= 0
      then best := (cand, placed);
      go ()
    end
  in
  let* () = go () in
  (* Stopped early: the later checkpoints are this result. *)
  List.iter (fun c -> record ~iterations:!used c) !pending;
  Ok (List.rev !recorded)

let improve mode budget script ~lower_bound order start =
  let ( let* ) = Result.bind in
  let* results =
    improve_at mode
      ~checkpoints:[ budget.Budget.iterations ]
      ~seed:budget.Budget.seed script ~lower_bound order start
  in
  match results with
  | [ (_, result, effort) ] -> Ok (result, effort)
  | _ -> assert false
