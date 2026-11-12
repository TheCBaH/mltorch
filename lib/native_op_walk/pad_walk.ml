(* The random-walk config spaces of the ops in [Pad], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Pad = struct
  open Pad.Pad

  (* This op's random-walk config space: correlated by construction. The pad
     entries are derived from the drawn shape rather than mutated beside it, so
     no step can produce a reflect pad wider than its axis or a crop that empties
     one -- an invalid config would be reported as a walk mismatch rather than as
     the rejection it is. Rank/dim and mode coverage belong to the ATEN recipe,
     where the shape is a rank-bearing list. *)
  module Walk (L : Walk_core.Limits.S) = struct
    (* ONE correlated axis of whole valid configurations, not a pattern axis
       beside a mode axis. Two independent axes would spend most steps
       re-drawing the value they already had (the walk picks uniformly, current
       value included), so reflect would appear only by luck; and any pair the
       op rejects would be reported as a walk mismatch rather than as the
       refusal it is. *)
    type pattern =
      | Crop_h  (** both entries negative *)
      | Mixed_hw  (** pad one axis while cropping another *)
      | Pad_asymmetric_w  (** before <> after, so a swap is observable *)
      | Pad_hw  (** symmetric spatial padding, the common convolution shape *)
      | Reflect_asymmetric_w  (** the mirror with unequal sides *)
      | Reflect_crop_h  (** reflect mode with a negative entry *)
      | Reflect_hw  (** the mirror, on both spatial axes *)

    type cfg = { shape : Walk_core.Shape.t; pattern : pattern }

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 6; w = 6; c = 3 };
        pattern = Pad_hw;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

    (* Every amount is a fraction of the extent it applies to, which is what
       makes the config space valid at EVERY drawn shape: reflect needs each
       positive pad strictly below the extent, and a crop needs to leave at
       least one element.

       Both bounds have to be applied, and the [max 1] alone was not enough.
       The shape axis's floor is 1, and at extent 1 [max 1 (1 / 3)] is 1 -- a
       reflect of 1 on an extent-1 axis, which is exactly what
       [Pad.output_shape] refuses, so the walk raised out of [Err.or_raise]
       instead of stepping. It reproduced at seeds 9, 21, 26, 33 and 39 (12
       steps); the sweep never saw it only because its own seed happens not to
       draw a 1 in 5 steps, which is what made it a latent defect rather than a
       red test. Clamping to [n - 1] gives 0 there, and changes nothing at any
       extent >= 2. *)
    let params (c : cfg) : params =
      let h = c.shape.Walk_core.Shape.h and w = c.shape.Walk_core.Shape.w in
      let third n = Stdlib.min (Stdlib.max 1 (n / 3)) (n - 1) in
      let pads =
        match c.pattern with
        | Crop_h -> [ (Axis.H, { before = -third h; after = 0 }) ]
        | Mixed_hw ->
            [
              (Axis.H, { before = -third h; after = 0 });
              (Axis.W, { before = 0; after = third w });
            ]
        | Pad_asymmetric_w -> [ (Axis.W, { before = third w; after = 0 }) ]
        | Pad_hw ->
            [
              (Axis.H, { before = third h; after = third h });
              (Axis.W, { before = third w; after = third w });
            ]
        | Reflect_asymmetric_w -> [ (Axis.W, { before = third w; after = 0 }) ]
        | Reflect_crop_h ->
            [
              (Axis.H, { before = -third h; after = 0 });
              (Axis.W, { before = third w; after = 0 });
            ]
        | Reflect_hw ->
            [
              (Axis.H, { before = third h; after = third h });
              (Axis.W, { before = third w; after = third w });
            ]
      in
      let mode =
        match c.pattern with
        | Crop_h -> Constant 0.5
        | Mixed_hw -> Constant 0.5
        | Pad_asymmetric_w -> Constant 0.5
        | Pad_hw -> Constant 0.5
        | Reflect_asymmetric_w -> Reflect
        | Reflect_crop_h -> Reflect
        | Reflect_hw -> Reflect
      in
      { pads; mode }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "pattern"
            [
              Pad_hw;
              Pad_asymmetric_w;
              Crop_h;
              Mixed_hw;
              Reflect_hw;
              Reflect_asymmetric_w;
              Reflect_crop_h;
            ] (fun r v -> { r with pattern = v });
        ]

    let pp_pattern = function
      | Crop_h -> "crop_h"
      | Mixed_hw -> "mixed_hw"
      | Pad_asymmetric_w -> "pad_asym_w"
      | Pad_hw -> "pad_hw"
      | Reflect_asymmetric_w -> "reflect_asym_w"
      | Reflect_crop_h -> "reflect_crop_h"
      | Reflect_hw -> "reflect_hw"

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a pattern=%s}" Walk_core.Shape.pp c.shape
        (pp_pattern c.pattern)
  end
end
