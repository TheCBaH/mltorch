(* Bridge a neutral walk-config shape (Walk_core.Shape.t, used by the random-walk
   language) to the engine's NHWC Vec6 shape. The two share axis order
   (N T D H W C), so this is a direct field map. *)

let vec6 (s : Walk_core.Shape.t) =
  let { Walk_core.Shape.n; t; d; h; w; c } = s in
  Vec6.shape ~n ~t ~d ~h ~w ~c
