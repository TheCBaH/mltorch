type t =
  | Greedy_by_area
  | Greedy_by_lifetime
  | Greedy_by_size
  | Greedy_by_size_best_fit

let all =
  [
    Greedy_by_area; Greedy_by_lifetime; Greedy_by_size; Greedy_by_size_best_fit;
  ]

let equal (a : t) b = a = b

let name = function
  | Greedy_by_area -> "greedy_by_area"
  | Greedy_by_lifetime -> "greedy_by_lifetime"
  | Greedy_by_size -> "greedy_by_size"
  | Greedy_by_size_best_fit -> "greedy_by_size_best_fit"

let pp ppf t = Format.pp_print_string ppf (name t)
