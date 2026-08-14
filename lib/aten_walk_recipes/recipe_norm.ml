(* rms_norm.default walk recipe: a rank-N input normalized over its trailing k
   axes, with an optional weight carrying exactly those extents.

   The three shapes are CORRELATED by construction rather than by a cascade:
   [normalized] is one list, and both the input's trailing extents and the
   weight's shape are that same list. An uncorrelated walk here would generate
   specs ATen rejects, which degrade to skips -- and a suite of skips reads as
   coverage while being none.

   [leading] carries the whole leading shape as one value, so a step varies the
   input's RANK as well as its extents; [normalized] does the same for k. *)

type t = {
  leading : int list;
  normalized : int list;
  eps : float option;
  weight : bool;
}

let cascade c = c
let input_shape c = c.leading @ c.normalized
let normalized_shape c = c.normalized
let weight_shape c = c.normalized
let has_weight c = c.weight
let eps c = c.eps

let axes ~leading ~normalized ~eps ~weight =
  Walk.
    [
      field_axis "leading" leading (fun c v -> { c with leading = v });
      field_axis "normalized" normalized (fun c v -> { c with normalized = v });
      field_axis "eps" eps (fun c v -> { c with eps = v });
      field_axis "weight" weight (fun c v -> { c with weight = v });
    ]

let pp ppf c =
  Format.fprintf ppf "{input=[%s] normalized=[%s] eps=%s weight=%b}"
    (String.concat "," (List.map string_of_int (input_shape c)))
    (String.concat "," (List.map string_of_int c.normalized))
    (match c.eps with None -> "default" | Some e -> Printf.sprintf "%g" e)
    c.weight
