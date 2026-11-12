(* The random-walk config spaces of the ops in [Pointwise_binary], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Bin = struct
  (* Walk config space shared by the binary pointwise ops (Add/Mul): one shape
     used for both operands (equal-shape; broadcast not exercised). A functor
     over the global Limits; no constraint, so [cascade] is identity. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun _ s -> { shape = s });
        ]

    let pp fmt (c : cfg) = Walk_core.Shape.pp fmt c.shape
  end
end

module Scalar_bin = struct
  (* Walk config shared by the scalar pointwise ops: one shape plus a scalar
     drawn from a finite candidate set. Zero is deliberately absent so the same
     space is safe for [Div_scalar] — unlike [Div], whose random *tensor*
     divisor is what makes it unwalkable. Explicit zero behaviour is pinned by
     the direct unit tests instead. The list mixes integer-like values (what
     MobileNet actually serialises), a negative, and an f32-inexact value. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; scalar : float }

    let candidates = [ 3.; 6.; -2.; 0.5; 0.1 ]

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 };
        scalar = 3.;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "scalar" candidates (fun c v -> { c with scalar = v });
        ]

    let pp fmt (c : cfg) =
      Fmt.pf fmt "%a scalar=%a" Walk_core.Shape.pp c.shape Fmt.float c.scalar
  end
end
