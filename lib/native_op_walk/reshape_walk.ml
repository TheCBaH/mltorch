(* The random-walk config spaces of the ops in [Reshape], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Reshape = struct
  open Reshape.Reshape

  (* Walk config: just the input shape; the target flattens all elements onto C
     (always numel-preserving), which fully exercises the delinearize path. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 4 } }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

    let params (c : cfg) : params =
      let numel = (Vec6.numel (Walk_shape.vec6 c.shape) :> int) in
      { shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:numel }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun _ s -> { shape = s });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a -> flat}" Walk_core.Shape.pp c.shape
  end
end
