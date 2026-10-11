(* Deterministic accumulators. Every one consumes values in the order it is
   given and records non-finite intermediate results instead of propagating
   them, so an overflow is reported rather than turned into a spurious
   infinity or zero. *)

module Sum = struct
  (* Neumaier compensated summation. *)
  type t = {
    mutable sum : float;
    mutable comp : float;
    mutable overflow : bool;
  }

  let create () = { sum = 0.0; comp = 0.0; overflow = false }

  let add t x =
    let s = t.sum +. x in
    if not (Float.is_finite s) then t.overflow <- true
    else (
      if Float.abs t.sum >= Float.abs x then
        t.comp <- t.comp +. (t.sum -. s +. x)
      else t.comp <- t.comp +. (x -. s +. t.sum);
      t.sum <- s)

  let value t = t.sum +. t.comp
  let overflowed t = t.overflow || not (Float.is_finite (value t))
end

module Sum_squares = struct
  (* Scaled sum of squares (the LAPACK xLASSQ recurrence): the sum is
     [scale^2 * ssq], so squaring a large finite value cannot overflow. *)
  type t = { mutable scale : float; mutable ssq : float }

  let create () = { scale = 0.0; ssq = 1.0 }

  let add t x =
    if x <> 0.0 then
      let a = Float.abs x in
      if t.scale < a then (
        let r = t.scale /. a in
        t.ssq <- 1.0 +. (t.ssq *. r *. r);
        t.scale <- a)
      else
        let r = a /. t.scale in
        t.ssq <- t.ssq +. (r *. r)

  let norm t = t.scale *. Float.sqrt t.ssq

  (* sqrt(sum / n) without forming the sum. *)
  let root_mean t n = t.scale *. Float.sqrt (t.ssq /. n)
end

module Welford = struct
  type t = {
    mutable n : float;
    mutable mean : float;
    mutable m2 : float;
    mutable overflow : bool;
  }

  let create () = { n = 0.0; mean = 0.0; m2 = 0.0; overflow = false }

  let add t x =
    t.n <- t.n +. 1.0;
    let delta = x -. t.mean in
    t.mean <- t.mean +. (delta /. t.n);
    t.m2 <- t.m2 +. (delta *. (x -. t.mean));
    if not (Float.is_finite t.mean && Float.is_finite t.m2) then
      t.overflow <- true

  let count t = t.n
  let mean t = t.mean
  let overflowed t = t.overflow

  (* Standard deviation for the given divisor; [None] when undefined. *)
  let deviation t ~sample =
    if sample then
      if t.n < 2.0 then None else Some (Float.sqrt (t.m2 /. (t.n -. 1.0)))
    else if t.n < 1.0 then None
    else Some (Float.sqrt (t.m2 /. t.n))
end
