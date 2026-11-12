(* The random-walk config spaces of the ops in [Factory], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Zeros = struct
  (* Shape-only; [fmt] is fixed at F32 by each walk file (plan S6/T6.6),
     mirroring [Pointwise_unary.To_copy.Walk]'s own convention -- a factory
     has no operand to declare a format on, so the format is the walk's own
     choice, not a config-space axis. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "shape" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun _ s -> { shape = s });
        ]

    let pp fmt (c : cfg) = Walk_core.Shape.pp fmt c.shape
  end
end

module Eye = struct
  (* Shape-only, mirroring [Zeros.Walk]'s own convention. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "shape" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun _ s -> { shape = s });
        ]

    let pp fmt (c : cfg) = Walk_core.Shape.pp fmt c.shape
  end
end

module Arange = struct
  (* [count] is the one config axis (plan S6/T6.6/W): [length]/[length_exact]
     both derive an element count from [start]/[stop]/[step], so varying the
     count directly (rather than the three bounds separately, which could
     draw an empty or over-limit range) always yields a valid subject.
     Bounded by [L.limits.max_channels], not [max_extent]/[max_numel]: the
     output lands entirely on [C] ([output_shape]'s own `~c:count`), the same
     axis [max_channels] caps for every other walk. Each of [Arange]'s two
     walk FILES builds [params] from [count] its own way -- a plain
     [0, count)] float range for the default arm, an offset int64 range past
     2^53 for the exact arm -- so this functor only owns the shared count
     axis, not [params] construction itself. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { count : int }

    let initial = { count = 4 }
    let cascade c = c
    let count (c : cfg) = c.count

    let axes =
      Walk_core.Walk.
        [
          int_axis "count" ~lo:1 ~hi:L.limits.Walk_core.Limits.max_channels
            (fun _ v -> { count = v });
        ]

    let pp fmt (c : cfg) = Fmt.pf fmt "count=%d" c.count
  end
end
