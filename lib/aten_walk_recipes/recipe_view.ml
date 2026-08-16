(* view.default / _unsafe_view.default walk recipe: a rank-4 input shape plus
   a reshape TARGET drawn as one correlated axis, not picked independently --
   a valid reshape target is a factorization of the source's element count, so
   an independently mutated target would usually be invalid and degrade to a
   skip (the same failure mode [Recipe_binary]'s header warns about).
   [pattern] names which factorization; [cascade] recomputes the concrete
   [target] list from [pattern] and the CURRENT shape after every mutation
   (shape axes included, not only [pattern] itself), so [target] stays valid
   regardless of which axis a step touched. One pattern carries a [-1]
   (PyTorch's inferred-dimension form) and the others change rank, both up and
   down from the source's 4. *)

type pattern =
  | Flatten  (** rank 4 -> 1: the whole source on C *)
  | Merge_hw  (** rank 4 -> 3: H and W merged onto one axis *)
  | Split_c  (** rank 4 -> 5: C split into two factors *)
  | Minus_one_c  (** rank 4 -> 3: [-1; h; w], inferring n*c *)

type t = {
  n : int;
  c : int;
  h : int;
  w : int;
  pattern : pattern;
  target : int list;
}

let numel c = c.n * c.c * c.h * c.w

let target_of pattern c =
  match pattern with
  | Flatten -> [ numel c ]
  | Merge_hw -> [ c.n; c.c; c.h * c.w ]
  | Split_c -> [ c.n; c.c / 2; 2; c.h; c.w ]
  | Minus_one_c -> [ -1; c.h; c.w ]

(* [Split_c] needs an even [c] to split exactly; every other pattern accepts
   any positive [c]. Repairing here (rather than filtering the [c] candidate
   set) is what keeps [target] a real factorization after EITHER a [pattern]
   step or a [c] step -- the engine applies [cascade] after every mutation,
   whichever axis it touched (walk_core/walk.ml's [run]). *)
let cascade t =
  let t =
    if t.pattern = Split_c && t.c mod 2 <> 0 then { t with c = t.c + 1 } else t
  in
  { t with target = target_of t.pattern t }

let all_patterns = [ Flatten; Merge_hw; Split_c; Minus_one_c ]
let self_shape c = [ c.n; c.c; c.h; c.w ]
let target c = c.target

let axes ~n ~c ~h ~w ~pattern =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "c" c (fun cf v -> { cf with c = v });
      field_axis "h" h (fun (c : t) v -> { c with h = v });
      field_axis "w" w (fun (c : t) v -> { c with w = v });
      field_axis "pattern" pattern (fun c v -> { c with pattern = v });
    ]

let pp_pattern ppf = function
  | Flatten -> Format.fprintf ppf "flatten"
  | Merge_hw -> Format.fprintf ppf "merge_hw"
  | Split_c -> Format.fprintf ppf "split_c"
  | Minus_one_c -> Format.fprintf ppf "minus_one_c"

let pp ppf c =
  let target_str =
    "[" ^ String.concat "," (List.map string_of_int c.target) ^ "]"
  in
  Format.fprintf ppf "{shape=[%d,%d,%d,%d] pattern=%a target=%s}" c.n c.c c.h
    c.w pp_pattern c.pattern target_str
