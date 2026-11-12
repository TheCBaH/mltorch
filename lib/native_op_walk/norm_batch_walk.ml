(* The random-walk config spaces of the ops in [Norm_batch], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module BatchNorm = struct
  open Norm_batch.BatchNorm

  (* Walk config: just the input shape and eps; the per-channel [C] vectors are
     derived from the shape's C extent by the walk's [build] (see
     lib/native_op_walk/batch_norm_nwalk.ml), which also keeps running_var
     non-negative so sqrt is real. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; eps : float }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        eps = 1e-5;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let params (c : cfg) : params = { channel = Axis.C; eps = c.eps }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "eps" [ 1e-5; 0.; 1e-3 ] (fun r v -> { r with eps = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a eps=%g}" Walk_core.Shape.pp c.shape c.eps
  end
end

module BatchNormNoStats = struct
  open Norm_batch.BatchNormNoStats

  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = {
      shape : Walk_core.Shape.t;
      eps : float;
      weight : bool;
      bias : bool;
    }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 3; w = 2; c = 4 };
        eps = 1e-5;
        weight = true;
        bias = true;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let params (c : cfg) = { channel = Axis.C; eps = c.eps }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c shape -> { c with shape });
          field_axis "eps" [ 1e-5; 1e-8 ] (fun c eps -> { c with eps });
          field_axis "weight" [ true; false ] (fun c weight ->
              { c with weight });
          field_axis "bias" [ true; false ] (fun c bias -> { c with bias });
        ]

    let pp fmt (c : cfg) =
      Fmt.pf fmt "{shape=%a eps=%g weight=%b bias=%b}" Walk_core.Shape.pp
        c.shape c.eps c.weight c.bias
  end
end
