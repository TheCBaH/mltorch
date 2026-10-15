let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

module type OPS = sig
  type t

  val const : float -> t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val select_lt : t -> t -> t -> t -> t
  val select_eq : t -> t -> t -> t -> t
  val share : t -> t
end

module Make (O : OPS) = struct
  open O

  let c = const

  let zero = c 0.
  and one = c 1.
  and two = c 2.
  and half = c 0.5

  let neg x = sub zero x
  let inf () = div one zero
  let nan () = div zero zero
  let abs x = select_lt x zero (neg x) x

  (* Truncation toward zero by add and subtract of 2^23, which round to an
     integer exactly below 2^23 in magnitude: round to nearest, then step back
     one where that went away from zero. A value at or above 2^23 is already an
     integer, a NaN falls through the comparison and stays, and a zero result is
     [+0]. Only operations every target has. *)
  let trunc x =
    let x = share x in
    let m = c 8388608. in
    let rp = share (sub (add x m) m) in
    let tp = share (select_lt x rp (sub rp one) rp) in
    let rn = share (add (sub x m) m) in
    let tn = share (select_lt rn x (add rn one) rn) in
    let t = share (select_lt x zero tn tp) in
    select_lt (abs x) m t x

  (* [horner [c0; c1; ...] x] is (((c0 x + c1) x + c2) ...). *)
  let horner cs x =
    match cs with
    | [] -> invalid_arg "F32_math.horner"
    | c0 :: rest ->
        List.fold_left (fun acc k -> share (add (mul acc x) (c k))) (c c0) rest

  (* [2^n] for an integer-valued [n] of magnitude at most 127, by the bits of
     [|n|] from the top: each set bit multiplies by [2^(2^b)], exact; a negative
     [n] divides one by the result, exact too. *)
  let pow2 n =
    let n = share n in
    let a = share (abs n) in
    let rec go f a = function
      | [] -> f
      | [ b ] ->
          let tb = c b and cb = c (Float.ldexp 1. (int_of_float b)) in
          share (select_lt a tb f (mul f cb))
      | b :: rest ->
          let tb = c b and cb = c (Float.ldexp 1. (int_of_float b)) in
          let f' = share (select_lt a tb f (mul f cb)) in
          let a' = share (select_lt a tb a (sub a tb)) in
          go f' a' rest
    in
    let f = go one a [ 64.; 32.; 16.; 8.; 4.; 2.; 1. ] in
    select_lt n zero (div one f) f

  let exp x =
    let x = share x in
    let lo = c (-104.) and hi = c 89. in
    (* a NaN falls through both comparisons and stays *)
    let xc = share (select_lt x lo lo (select_lt hi x hi x)) in
    let t = share (mul xc (c 1.44269504088896341)) in
    let k = share (trunc (add t (select_lt t zero (c (-0.5)) half))) in
    let r =
      share (sub (sub xc (mul k (c 0.693359375))) (mul k (c (-2.12194440e-4))))
    in
    let z = share (mul r r) in
    let p =
      horner
        [
          1.9875691500E-4;
          1.3981999507E-3;
          8.3334519073E-3;
          4.1665795894E-2;
          1.6666665459E-1;
          5.0000001201E-1;
        ]
        r
    in
    let y = share (add (add (mul p z) r) one) in
    (* two factors keep each one finite: k reaches 129 and -150 *)
    let k1 = share (trunc (mul k half)) in
    let k2 = share (sub k k1) in
    mul (mul y (pow2 k1)) (pow2 k2)

  let log x0 =
    let x0 = share x0 in
    (* anything below 2^-63 is scaled up by 2^96 first, which the 127-bit down
       cascade then brings into range; zero stays zero *)
    let tiny = c (Float.ldexp 1. (-63)) in
    let x = share (select_lt x0 tiny (mul x0 (c (Float.ldexp 1. 96))) x0) in
    let e = share (select_lt x0 tiny (c (-96.)) zero) in
    let down (x, e) b =
      let tb = c b in
      let y = share (mul x (c (Float.ldexp 1. (-int_of_float b)))) in
      (share (select_lt y one x y), share (select_lt y one e (add e tb)))
    in
    let up (x, e) b =
      let tb = c b in
      let y = share (mul x (c (Float.ldexp 1. (int_of_float b)))) in
      (share (select_lt y two y x), share (select_lt y two (sub e tb) e))
    in
    let x, e = List.fold_left down (x, e) [ 64.; 32.; 16.; 8.; 4.; 2.; 1. ] in
    let x, e = List.fold_left up (x, e) [ 32.; 16.; 8.; 4.; 2.; 1. ] in
    (* x in [1, 2): fold the mantissa into [sqrt 1/2, sqrt 2) *)
    let sqrt2 = c 1.41421356237309505 in
    let f = share (select_lt x sqrt2 (sub x one) (sub (mul x half) one)) in
    let e = share (select_lt x sqrt2 e (add e one)) in
    let z = share (mul f f) in
    let p =
      horner
        [
          7.0376836292E-2;
          -1.1514610310E-1;
          1.1676998740E-1;
          -1.2420140846E-1;
          1.4249322787E-1;
          -1.6668057665E-1;
          2.0000714765E-1;
          -2.4999993993E-1;
          3.3333331174E-1;
        ]
        f
    in
    let y = share (mul (mul p f) z) in
    let y = share (add y (mul e (c (-2.12194440e-4)))) in
    let y = share (sub y (mul z half)) in
    let f = share (add f y) in
    let r = add f (mul e (c 0.693359375)) in
    let r = select_eq x0 zero (neg (inf ())) r in
    let r = select_lt x0 zero (nan ()) r in
    select_lt (c 3.40282346638528860e38) x0 (inf ()) r

  (* The shared reduction of sin and cos: the octant [j] made even, the sign
     bookkeeping left to the caller, and the reduced argument and its square. *)
  let reduce x =
    let ax = share (abs x) in
    let j = share (trunc (mul ax (c 1.27323954473516))) in
    let h = share (trunc (mul j half)) in
    let odd = sub j (add h h) in
    let j = share (add j odd) in
    let jm = share (sub j (mul (trunc (mul j (c 0.125))) (c 8.))) in
    let xr =
      share
        (sub
           (sub
              (sub ax (mul j (c 0.78515625)))
              (mul j (c 2.4187564849853515625e-4)))
           (mul j (c 3.77489497744594108e-8)))
    in
    let z = share (mul xr xr) in
    (jm, xr, z)

  let sin_poly xr z =
    let p = horner [ -1.9515295891E-4; 8.3321608736E-3; -1.6666654611E-1 ] z in
    add (mul (mul p z) xr) xr

  let cos_poly z =
    let p =
      horner
        [
          2.443315711809948E-005;
          -1.388731625493765E-003;
          4.166664568298827E-002;
        ]
        z
    in
    add (sub (mul (mul p z) z) (mul z half)) one

  let sin x =
    let x = share x in
    let jm, xr, z = reduce x in
    let sign = share (select_lt x zero (c (-1.)) one) in
    let flip = select_lt jm (c 4.) one (c (-1.)) in
    let sign = share (mul sign flip) in
    let jm = share (select_lt jm (c 4.) jm (sub jm (c 4.))) in
    let y = select_eq jm two (cos_poly z) (sin_poly xr z) in
    mul sign y

  let cos x =
    let jm, xr, z = reduce x in
    let big = select_lt (c 3.) jm (c (-1.)) one in
    let jm = share (select_lt (c 3.) jm (sub jm (c 4.)) jm) in
    let sign = share (select_lt one jm (neg big) big) in
    let y = select_eq jm two (sin_poly xr z) (cos_poly z) in
    mul sign y

  (* Abramowitz-Stegun 7.1.26, with the exponential of the binary32 [exp]. *)
  let erf x =
    let x = share x in
    let sign = select_lt x zero (c (-1.)) one in
    let ax = share (abs x) in
    let t = share (div one (add one (mul (c 0.3275911) ax))) in
    let poly =
      mul t
        (horner
           [ 1.061405429; -1.453152027; 1.421413741; -0.284496736; 0.254829592 ]
           t)
    in
    let e = exp (neg (mul ax ax)) in
    mul sign (sub one (mul poly e))
end

module Value_ops = struct
  type t = float

  let const x = round32 x
  let add a b = round32 (a +. b)
  let sub a b = round32 (a -. b)
  let mul a b = round32 (a *. b)
  let div a b = round32 (a /. b)
  let select_lt a b c d = if a < b then c else d
  let select_eq a b c d = if a = b then c else d
  let share x = x
end

module Value = struct
  module M = Make (Value_ops)

  let arg f x = f (round32 x)
  let cos = arg M.cos
  let erf = arg M.erf
  let exp = arg M.exp
  let log = arg M.log
  let sin = arg M.sin
end
