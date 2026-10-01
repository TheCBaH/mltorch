(* Order search: keep an order of blocks, decode it, and try a move; keep the
   move if the pool is no larger. One iteration is one move plus one decode.
   Nothing reads a clock: the same order, budget and seed give the same result,
   and the moves of iteration [i] do not depend on the budget, so a larger
   budget only extends the same walk. *)

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

let below s bound =
  Int64.to_int (Int64.rem (Int64.shift_right_logical s 33) (Int64.of_int bound))

(* Move the block at position [src] to position [dst] (both in the order). *)
let move order src dst =
  let b = order.(src) in
  if dst < src then Array.blit order dst order (dst + 1) (src - dst)
  else Array.blit order (src + 1) order src (dst - src);
  order.(dst) <- b

(* Blocks whose top edge sets the pool. *)
let peak_positions script order offsets pool =
  let acc = ref [] in
  Array.iteri
    (fun pos b ->
      if
        script.Script.sizes.(b) <> 0L
        && Int64.add offsets.(b) script.Script.sizes.(b) = pool
      then acc := pos :: !acc)
    order;
  Array.of_list (List.rev !acc)

let improve mode budget script ~lower_bound order (offsets0, pool0) =
  let n = Script.blocks script in
  let best = ref (Array.copy order, offsets0, pool0) in
  let rng = ref budget.Budget.seed and used = ref 0L in
  let ( let* ) = Result.bind in
  let rec go () =
    let _, _, pool = !best in
    if
      n < 2
      || Int64.compare pool lower_bound <= 0
      || Int64.compare !used budget.Budget.iterations >= 0
    then Ok ()
    else begin
      used := Int64.succ !used;
      let cur, cur_offsets, cur_pool = !best in
      let cand = Array.copy cur in
      rng := next !rng;
      let peaks = peak_positions script cur cur_offsets cur_pool in
      let src =
        if Array.length peaks = 0 then below !rng n
        else peaks.(below !rng (Array.length peaks))
      in
      rng := next !rng;
      (* A peak block moves earlier, so it is placed before what boxed it in. *)
      let dst = if src = 0 then 0 else below !rng src in
      rng := next !rng;
      if below !rng 4 = 0 then begin
        let a = below !rng n in
        rng := next !rng;
        let b = below !rng n in
        let t = cand.(a) in
        cand.(a) <- cand.(b);
        cand.(b) <- t
      end
      else move cand src dst;
      let* offsets, pool = Ia_decode.offsets mode script cand in
      if Int64.compare pool cur_pool <= 0 then best := (cand, offsets, pool);
      go ()
    end
  in
  let* () = go () in
  let _, offsets, pool = !best in
  let stop =
    if Int64.compare pool lower_bound <= 0 then Stop.Lower_bound
    else Stop.Budget_exhausted
  in
  Ok ((offsets, pool), { Effort.iterations = !used; stop })
