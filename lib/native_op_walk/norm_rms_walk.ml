(* The random-walk config spaces of the ops in [Norm_rms], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module RmsNorm = struct
  open Norm_rms.RmsNorm

  (* The normalized axes are the innermost [k] of the frame, which is what both
     importers derive from ATen's trailing [normalized_shape]. [k] is an axis of
     the walk rather than a fixed choice, because the single- and multi-axis
     cases divide by different counts and read the weight on different axes --
     and both preserve the output shape, so a walk that only ever tried one
     would look like coverage and be none.

     [weight] is an axis too: [Graph_ir]'s [Rms_norm] carries it as an option
     and the absent case is a different code path, not a ones tensor. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = {
      shape : Walk_core.Shape.t;
      k : int;
      eps : float;
      weight : bool;
    }

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 3; c = 5 };
        k = 1;
        eps = 1e-5;
        weight = true;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let weight_present (c : cfg) = c.weight
    let dims (c : cfg) = List.filteri (fun i _ -> i >= 6 - c.k) Axis.all
    let params (c : cfg) : params = { dims = dims c; eps = c.eps }

    (* ATen indexes the weight by the normalized dims only, so it is extent-1
       everywhere else and broadcasts. *)
    let weight_shape (c : cfg) =
      let x = shape c in
      List.fold_left
        (fun acc a -> Vec6.set acc a (Vec6.get x a))
        (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
        (dims c)

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "k" [ 1; 2; 3 ] (fun c v -> { c with k = v });
          field_axis "eps" [ 1e-5; 0.; 1e-3 ] (fun c v -> { c with eps = v });
          field_axis "weight" [ true; false ] (fun c v -> { c with weight = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a k=%d eps=%g weight=%b}" Walk_core.Shape.pp
        c.shape c.k c.eps c.weight
  end
end
