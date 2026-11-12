(* The random-walk config spaces of the ops in [Norm_layer], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

open Norm_shared

module LayerNorm = struct
  open Norm_layer.LayerNorm

  (* [k] (how many trailing axes are normalised) and the two affine states are
     walk axes rather than fixed choices, for [RmsNorm.Walk]'s reasons and one
     more: layer_norm's weight and bias are INDEPENDENTLY optional, so all four
     combinations are structurally different graphs.

     [k] stops at 3 and the shape axis keeps the extents small on purpose. The
     symbolic program nests one reduction per normalised axis and does it twice,
     so both the expression size and the staging cost grow with [k]; the
     ceilings are [Kernel.Limits.Hard.depth]/[eval_depth]. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = {
      shape : Walk_core.Shape.t;
      k : int;
      eps : float;
      weight : bool;
      bias : bool;
    }

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 3; c = 5 };
        k = 1;
        eps = 1e-5;
        weight = true;
        bias = true;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let weight_present (c : cfg) = c.weight
    let bias_present (c : cfg) = c.bias
    let dims (c : cfg) = List.filteri (fun i _ -> i >= 6 - c.k) Axis.all
    let params (c : cfg) : params = { dims = dims c; eps = c.eps }

    let affine_shape (c : cfg) =
      normalized_shape ~x_shape:(shape c) ~dims:(dims c)

    (* Both epsilons the corpus contains, plus 0 -- which is the interesting
       one, since it is where a variance of 0 (a constant normalised slice)
       divides by zero rather than by eps. *)
    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "k" [ 1; 2; 3 ] (fun c v -> { c with k = v });
          field_axis "eps" [ 1e-5; 1e-6; 0. ] (fun c v -> { c with eps = v });
          field_axis "weight" [ true; false ] (fun c v -> { c with weight = v });
          field_axis "bias" [ true; false ] (fun c v -> { c with bias = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a k=%d eps=%g weight=%b bias=%b}"
        Walk_core.Shape.pp c.shape c.k c.eps c.weight c.bias
  end
end
