(* PCG32 (pcg_setseq_64_xsh_rr_32): a small, fast, statistically strong RNG,
   kept purely functional here — [next] returns the value and the advanced
   state, so a whole operation's tensor payloads are synthesized from one
   threaded instance and are reproducible from the seed alone.

   Float draws get special treatment for verification:
   - uniforms are built from the high 24 bits (the f32 mantissa width) to avoid
     low-bit bias, and land exactly on f32-representable fractions;
   - every result is rounded to its float32-canonical value ([Float32.to_f32]),
     so it equals exactly what an f32 tensor stores;
   - the Box-Muller normal draws u1 from (0,1] so [log] never sees 0 — no NaN or
     infinity is ever produced (which would break the element-wise comparison). *)

type t = { state : int64; inc : int64 }

let mult = 0x5851f42d4c957f2dL
let step { state; inc } = { state = Int64.add (Int64.mul state mult) inc; inc }

(* Seed a stream: [seq] selects an independent sequence; [seed] sets the start
   state (standard PCG seeding: advance, add seed, advance). *)
let seed ~seed ~seq =
  let inc = Int64.logor (Int64.shift_left seq 1) 1L in
  let s = step { state = 0L; inc } in
  let s = { s with state = Int64.add s.state seed } in
  step s

(* The default stream, used when a spec carries no seed of its own (the
   constants are PCG's reference defaults). A future per-tensor seed in the
   Random payload could override this. *)
let default = seed ~seed:0x853c49e6748fea9bL ~seq:0xda3e39cb94b95bdbL

(* The next 32-bit output in [0, 2^32) as an [int64], plus the advanced state.
   The XSH-RR output (xorshift then random rotate) is computed in Int64 anyway.

   [int64], not [int], because [0, 2^32) does not fit a non-negative OCaml [int]
   on every runtime we target: js_of_ocaml's [int] is 32 bits ([Sys.int_size =
   32]), so [Int64.to_int] here would hand back a NEGATIVE number for every draw
   >= 2^31 — and [Walk.pick]'s [List.nth] then raises. Callers narrow to [int]
   only after bounding the value ([lsr 8] below, [Int64.rem] in [Walk]); doing it
   here would push an impossible contract onto them. *)
let next t =
  let s = t.state in
  let xorshifted =
    Int64.logand
      (Int64.shift_right_logical
         (Int64.logxor (Int64.shift_right_logical s 18) s)
         27)
      0xFFFFFFFFL
  in
  let rot = Int64.to_int (Int64.shift_right_logical s 59) land 31 in
  let out =
    Int64.logand
      (Int64.logor
         (Int64.shift_right_logical xorshifted rot)
         (Int64.shift_left xorshifted ((32 - rot) land 31)))
      0xFFFFFFFFL
  in
  (out, step t)

let two_pow_neg24 = 1.0 /. 16777216.0

(* The top 24 bits of a draw, as an [int]. That is the f32 mantissa width, so the
   result is < 2^24 and fits every runtime's [int] — this is the narrowing the
   comment on [next] refers to. *)
let top24 n = Int64.to_int (Int64.shift_right_logical n 8)

(* A uniform in [0,1) at f32 mantissa precision (top 24 bits of the output). *)
let uniform_unit t =
  let n, t = next t in
  (float_of_int (top24 n) *. two_pow_neg24, t)

let uniform ~low ~high t =
  let u, t = uniform_unit t in
  (Float32.to_f32 (low +. (u *. (high -. low))), t)

let two_pi = 2.0 *. Float.pi

(* Box-Muller: one normal draw per call (the paired value is discarded to keep
   the state purely a function of how many draws were taken). *)
let normal ~mean ~std t =
  let n1, t = next t in
  let u2, t = uniform_unit t in
  let u1 = float_of_int (top24 n1 + 1) *. two_pow_neg24 in
  let z = sqrt (-2.0 *. log u1) *. cos (two_pi *. u2) in
  (Float32.to_f32 (mean +. (std *. z)), t)
