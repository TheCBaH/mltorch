type t = { lo : int64; hi : int64 }

(* Saturation ceiling: far outside the domain, far inside int64, so one
   saturating operation on two saturated values still cannot wrap. *)
let ceiling = Int64.shift_left 1L 62
let floor = Int64.neg ceiling

let clamp x =
  if Int64.compare x ceiling > 0 then ceiling
  else if Int64.compare x floor < 0 then floor
  else x

let point x = { lo = clamp x; hi = clamp x }

let span ~lo ~hi =
  let lo = Int64.of_int lo in
  {
    lo;
    hi =
      (if Int64.compare (Int64.of_int hi) lo > 0 then
         Int64.pred (Int64.of_int hi)
       else lo);
  }

let unbounded = { lo = floor; hi = ceiling }
let domain = { lo = Int64.neg 2147483648L; hi = 2147483647L }

let within ~inner ~outer =
  Int64.compare inner.lo outer.lo >= 0 && Int64.compare inner.hi outer.hi <= 0

let sat_add a b = clamp (Int64.add a b)
let saturating_add = sat_add

(* Both factors are at most 2^62 in magnitude, so the product is computed only
   after dividing the ceiling by one of them: the check is on the operands, never
   on a product that may already have wrapped. *)
let sat_mul a b =
  if Int64.equal a 0L || Int64.equal b 0L then 0L
  else
    let negative = Int64.compare a 0L < 0 <> (Int64.compare b 0L < 0) in
    let a = Int64.abs a and b = Int64.abs b in
    if Int64.compare a (Int64.div ceiling b) > 0 then
      if negative then floor else ceiling
    else
      let p = Int64.mul a b in
      if negative then Int64.neg p else p

let saturating_mul k x = sat_mul (Int64.of_int k) x

module Env = struct
  type range = t
  type t = { vars : range Loop_var.Map.t; temps : range Loop_temp.Map.t ref }

  let create () = { vars = Loop_var.Map.empty; temps = ref Loop_temp.Map.empty }
  let add_var v r env = { env with vars = Loop_var.Map.add v r env.vars }
  let set_temp t r env = env.temps := Loop_temp.Map.add t r !(env.temps)
end

let scale k { lo; hi } =
  let k = Int64.of_int k in
  let a = sat_mul k lo and b = sat_mul k hi in
  { lo = Stdlib.min a b; hi = Stdlib.max a b }

let div_pos ~round d { lo; hi } =
  let d = Int64.of_int d in
  let floor_div n =
    let q = Int64.div n d and r = Int64.rem n d in
    if Int64.compare r 0L < 0 then Int64.pred q else q
  in
  let f n =
    match round with
    | `Floor -> floor_div n
    | `Ceil -> Int64.neg (floor_div (Int64.neg n))
  in
  { lo = f lo; hi = f hi }

let rec of_index env : Loop_index.t -> t = function
  | Loop_index.Add (a, b) ->
      let a = of_index env a and b = of_index env b in
      { lo = sat_add a.lo b.lo; hi = sat_add a.hi b.hi }
  | Loop_index.Ceil_div_pos (a, d) -> div_pos ~round:`Ceil d (of_index env a)
  | Loop_index.Clamp_low a ->
      let a = of_index env a in
      { lo = Stdlib.max 0L a.lo; hi = Stdlib.max 0L a.hi }
  | Loop_index.Const n -> point (Int64.of_int n)
  | Loop_index.Floor_div_pos (a, d) -> div_pos ~round:`Floor d (of_index env a)
  | Loop_index.Max (a, b) ->
      let a = of_index env a and b = of_index env b in
      { lo = Stdlib.max a.lo b.lo; hi = Stdlib.max a.hi b.hi }
  | Loop_index.Min (a, b) ->
      let a = of_index env a and b = of_index env b in
      { lo = Stdlib.min a.lo b.lo; hi = Stdlib.min a.hi b.hi }
  | Loop_index.Scale (k, a) -> scale k (of_index env a)
  | Loop_index.Temp t ->
      Option.value ~default:unbounded
        (Loop_temp.Map.find_opt t !(env.Env.temps))
  | Loop_index.Var v ->
      Option.value ~default:unbounded (Loop_var.Map.find_opt v env.Env.vars)

let rec proven env (i : Loop_index.t) =
  match i with
  | Loop_index.Add (a, b) ->
      proven env a && proven env b
      && within ~inner:(of_index env i) ~outer:domain
  | Loop_index.Scale (_, a) ->
      proven env a && within ~inner:(of_index env i) ~outer:domain
  | Loop_index.Ceil_div_pos (a, _)
  | Loop_index.Clamp_low a
  | Loop_index.Floor_div_pos (a, _) ->
      proven env a
  | Loop_index.Max (a, b) | Loop_index.Min (a, b) ->
      proven env a && proven env b
  | Loop_index.Const _ | Loop_index.Temp _ | Loop_index.Var _ -> true
