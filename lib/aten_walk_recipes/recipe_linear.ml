(* linear.default walk recipe: a rank-N activation reduced on its trailing axis
   against a rank-2 [out_features, in_features] weight.

   [cascade] is the identity because there is only one cross-argument
   constraint and it is expressed structurally: [in_features] appears once, in
   both the input's trailing extent and the weight's second dimension, so the
   two cannot disagree. Contrast [Recipe_conv], where the channel count has to
   be rounded up to a multiple of the group count.

   [leading] is the input's shape ABOVE the feature axis, so a step can vary the
   rank as well as the extents -- the property that a wrong feature axis would
   break and a rank-2-only walk could never see.

   [bias] is an axis too, for the same reason it is one in [Recipe_norm]:
   [Graph_ir]'s [Linear] carries it as an OPTION, and the absent case is a
   different graph rather than a zero tensor. A walk that always supplied one
   left that path unfuzzed. *)

type t = {
  leading : int list;
  in_features : int;
  out_features : int;
  bias : bool;
}

let cascade c = c
let input_shape c = c.leading @ [ c.in_features ]
let weight_shape c = [ c.out_features; c.in_features ]
let bias_shape c = [ c.out_features ]
let has_bias c = c.bias

let axes ~leading ~in_features ~out_features ~bias =
  Walk.
    [
      field_axis "leading" leading (fun c v -> { c with leading = v });
      field_axis "in_features" in_features (fun c v ->
          { c with in_features = v });
      field_axis "out_features" out_features (fun c v ->
          { c with out_features = v });
      field_axis "bias" bias (fun c v -> { c with bias = v });
    ]

let pp ppf c =
  Format.fprintf ppf "{input=[%s,%d] out_features=%d bias=%b}"
    (String.concat "," (List.map string_of_int c.leading))
    c.in_features c.out_features c.bias
