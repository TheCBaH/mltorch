(* The random-walk config spaces of the ops in [Linear], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Linear = struct
  open Linear.Linear

  (* The config space is smaller than every other structured op's, and that is
     the point rather than an omission: [in_features] IS the activation's C
     extent, so the correlation a recipe normally has to maintain by hand is
     structural here. Only [out_features] is free.

     The shapes are deliberately NON-SQUARE by default -- [W] and [C] differ in
     [initial], and [out_features] differs from both -- because a transposed
     weight layout is buildable and merely wrong when they agree.

     [bias] is an axis because [Graph_ir]'s [Linear] carries it as an OPTION and
     [Eval_op] fills a zero one when it is absent: the two states are different
     graphs through different code, and a walk that always supplied a bias left
     the other one unfuzzed. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; out_features : int; bias : bool }

    let l = L.limits

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 1; w = 4; c = 8 };
        out_features = 6;
        bias = true;
      }

    let cascade c = c
    let bias_present (c : cfg) = c.bias

    let params (c : cfg) : params =
      { in_features = Dim.extent c.shape.Walk_core.Shape.c }

    let x_shape (c : cfg) = Walk_shape.vec6 c.shape

    (* [Out, 1, 1, 1, 1, In] -- the native linear weight layout, which is what
       both importers relayout ATen's rank-2 weight into. *)
    let weight_shape (c : cfg) =
      Vec6.shape ~n:c.out_features ~t:1 ~d:1 ~h:1 ~w:1
        ~c:c.shape.Walk_core.Shape.c

    let bias_shape (c : cfg) =
      Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:c.out_features

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" l
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          int_axis "out_features" ~lo:1 ~hi:l.max_channels (fun c v ->
              { c with out_features = v });
          field_axis "bias" [ true; false ] (fun c v -> { c with bias = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a out_features=%d bias=%b}" Walk_core.Shape.pp
        c.shape c.out_features c.bias
  end
end
