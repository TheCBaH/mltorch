(* addmm.default walk recipe. The bridge (op_bridge.ml) lowers addmm via
   Native's [Linear]: mat1 is the rank-2 input, mat2 (rank-2, checked) is
   permuted into [Linear]'s [out,in] weight layout, and [self] is passed
   straight through as the bias operand with no reshape -- so [self] must be
   exactly [out_features], not the fully general addmm broadcast shape.
   [beta]/[alpha] are held at their schema default (1); the bridge comment
   says as much ("alpha=beta=1 assumed"), so a non-default value is a bridge
   gap to pin with a hand fixture, not a walk axis. *)

type t = { n : int; in_features : int; out_features : int }

let cascade c = c
let self_shape c = [ c.out_features ]
let mat1_shape c = [ c.n; c.in_features ]
let mat2_shape c = [ c.in_features; c.out_features ]

let axes ~n ~in_features ~out_features =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "in_features" in_features (fun c v ->
          { c with in_features = v });
      field_axis "out_features" out_features (fun c v ->
          { c with out_features = v });
    ]

let pp ppf c =
  Format.fprintf ppf "{mat1=[%d,%d] mat2=[%d,%d] self=[%d]}" c.n c.in_features
    c.in_features c.out_features c.out_features
