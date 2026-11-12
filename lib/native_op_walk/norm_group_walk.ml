(* The random-walk config spaces of the ops in [Norm_group], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module GroupNorm = struct
  open Norm_group.GroupNorm

  (* Walk config: [groups] must divide the channel extent, so it is derived
     from the drawn shape's C via [divisor] rather than an independent axis --
     the same "derive from the drawn extent" shape [Slice.Walk]'s [bounds] and
     [Split_with_sizes.Walk]'s [sizes_for] use, and for the same reason: an
     independent [groups] axis would draw far more configurations
     [output_shape] refuses (indivisible) than ones it accepts. The three
     candidates span group_norm's range: 1 (whole-channel), 2, and C itself
     (per-channel groups -- group_norm's instance-norm corner). *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = {
      shape : Walk_core.Shape.t;
      divisor : int;
      eps : float;
      weight : bool;
      bias : bool;
    }

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 3; w = 3; c = 6 };
        divisor = 1;
        eps = 1e-5;
        weight = true;
        bias = true;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

    let groups (c : cfg) =
      let c_extent = (Vec6.get (shape c) Axis.C :> int) in
      let candidates =
        List.filter (fun g -> c_extent mod g = 0) [ 1; 2; c_extent ]
      in
      List.nth candidates (c.divisor mod List.length candidates)

    let weight_present (c : cfg) = c.weight
    let bias_present (c : cfg) = c.bias

    let params (c : cfg) : params =
      {
        batch = Axis.N;
        channel = Axis.C;
        groups = Op_config.Pos.of_int (groups c);
        eps = c.eps;
      }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "divisor" [ 0; 1; 2 ] (fun c v -> { c with divisor = v });
          field_axis "eps" [ 1e-5; 0.; 1e-3 ] (fun c v -> { c with eps = v });
          field_axis "weight" [ true; false ] (fun c v -> { c with weight = v });
          field_axis "bias" [ true; false ] (fun c v -> { c with bias = v });
        ]

    let pp fmt (c : cfg) =
      let p = params c in
      Format.fprintf fmt "{shape=%a groups=%d eps=%g weight=%b bias=%b}"
        Walk_core.Shape.pp c.shape
        (p.groups :> int)
        c.eps c.weight c.bias
  end
end
