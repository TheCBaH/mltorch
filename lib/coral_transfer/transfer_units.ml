let positive what n =
  if n < 0 then invalid_arg ("Transfer_units: negative " ^ what) else n

module Rows = struct
  type t = int

  let of_int n = positive "rows" n
  let to_int n = n
  let one = 1
  let is_positive n = n > 0
  let compare = Int.compare
  let equal = Int.equal
  let min = Stdlib.min
  let rec gcd a b = if b = 0 then a else gcd b (a mod b)
  let lcm a b = a * b / gcd a b
  let is_multiple ~of_ r = r mod of_ = 0
  let round_down ~to_ r = r / to_ * to_
  let round_up ~to_ r = (r + to_ - 1) / to_ * to_
  let pp = Fmt.int
end

module Bytes = struct
  type t = int

  let of_int n = positive "bytes" n
  let to_int n = n
  let zero = 0
  let add = ( + )
  let sub_floor a b = if b > a then 0 else a - b
  let times rows per_row = rows * per_row
  let times_extent n per = (n : Core.Dim.extent Core.Dim.t :> int) * per
  let round_up ~to_ n = (n + to_ - 1) / to_ * to_
  let double n = 2 * n
  let max = Stdlib.max
  let compare = Int.compare
  let equal = Int.equal
  let rows_fitting n ~per_row = n / Stdlib.max 1 per_row
  let is_whole ~of_ n = n mod of_ = 0
  let pp = Fmt.int
end
