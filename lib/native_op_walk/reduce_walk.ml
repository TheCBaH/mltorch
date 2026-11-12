(* The random-walk config spaces of the ops in [Reduce], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Mean = struct
  open Reduce.Mean

  (* This op's random-walk config space: a 4D NHWC tensor reduced over a curated
     set of axis subsets with keepdim toggled. Any non-empty subset is valid, so
     [cascade] is identity. Not shared with other backends. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; dims : Axis.t list; keepdim : bool }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        dims = [ Axis.H; Axis.W ];
        keepdim = false;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let params (c : cfg) : params = { dims = c.dims; keepdim = c.keepdim }

    let dim_sets =
      Axis.[ [ H; W ]; [ H; W; C ]; [ C ]; [ H ]; [ W ]; [ N; H; W ] ]

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "dims" dim_sets (fun r v -> { r with dims = v });
          field_axis "keepdim" [ true; false ] (fun r v ->
              { r with keepdim = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a dims=[%a] keepdim=%b}" Walk_core.Shape.pp
        c.shape
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
           Axis.pp)
        c.dims c.keepdim
  end
end

module Sum = struct
  open Reduce.Sum

  (* Same config space as [Mean]'s walk, minus the divisor concern -- any
     non-empty dim subset is valid, so [cascade] is identity. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; dims : Axis.t list; keepdim : bool }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        dims = [ Axis.H; Axis.W ];
        keepdim = false;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let params (c : cfg) : params = { dims = c.dims; keepdim = c.keepdim }

    let dim_sets =
      Axis.[ [ H; W ]; [ H; W; C ]; [ C ]; [ H ]; [ W ]; [ N; H; W ] ]

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "dims" dim_sets (fun r v -> { r with dims = v });
          field_axis "keepdim" [ true; false ] (fun r v ->
              { r with keepdim = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a dims=[%a] keepdim=%b}" Walk_core.Shape.pp
        c.shape
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
           Axis.pp)
        c.dims c.keepdim
  end
end

module Amax = struct
  open Reduce.Amax

  (* Same config space as [Mean]'s walk, minus the divisor -- any non-empty
     dim subset is valid, so [cascade] is identity. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; dims : Axis.t list; keepdim : bool }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        dims = [ Axis.H; Axis.W ];
        keepdim = false;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let params (c : cfg) : params = { dims = c.dims; keepdim = c.keepdim }

    let dim_sets =
      Axis.[ [ H; W ]; [ H; W; C ]; [ C ]; [ H ]; [ W ]; [ N; H; W ] ]

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "dims" dim_sets (fun r v -> { r with dims = v });
          field_axis "keepdim" [ true; false ] (fun r v ->
              { r with keepdim = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a dims=[%a] keepdim=%b}" Walk_core.Shape.pp
        c.shape
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
           Axis.pp)
        c.dims c.keepdim
  end
end

module MaxDim = struct
  open Reduce.MaxDim

  (* Same config space as [Softmax]'s walk (any axis, any extent), plus
     [Amax]'s [keepdim] toggle. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; axis : Axis.t; keepdim : bool }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        axis = Axis.C;
        keepdim = false;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let params (c : cfg) : params = { axis = c.axis; keepdim = c.keepdim }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "axis" Axis.all (fun c v -> { c with axis = v });
          field_axis "keepdim" [ true; false ] (fun c v ->
              { c with keepdim = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a axis=%a keepdim=%b}" Walk_core.Shape.pp
        c.shape Axis.pp c.axis c.keepdim
  end
end

module Vector_norm = struct
  open Reduce.Vector_norm

  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; dims : Axis.t list; keepdim : bool }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        dims = [ Axis.H; Axis.W ];
        keepdim = false;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let params (c : cfg) : params = { dims = c.dims; keepdim = c.keepdim }

    let dim_sets =
      Axis.[ [ H; W ]; [ H; W; C ]; [ C ]; [ H ]; [ W ]; [ N; H; W ] ]

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "dims" dim_sets (fun r v -> { r with dims = v });
          field_axis "keepdim" [ true; false ] (fun r v ->
              { r with keepdim = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a dims=[%a] keepdim=%b}" Walk_core.Shape.pp
        c.shape
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
           Axis.pp)
        c.dims c.keepdim
  end
end

module Softmax = struct
  open Reduce.Softmax

  (* This op's random-walk config space: a 4D NHWC tensor, softmaxed over one
     of the six frame axes. Any axis is valid at any extent (unlike
     [Mean]/[Amax]'s [dims] SUBSET, there is nothing to combine), so
     [cascade] is identity. Not shared with other backends. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; axis : Axis.t }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        axis = Axis.C;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let params (c : cfg) : params = { axis = c.axis }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "axis" Axis.all (fun c v -> { c with axis = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a axis=%a}" Walk_core.Shape.pp c.shape Axis.pp
        c.axis
  end
end

module Cumsum = struct
  open Reduce.Cumsum

  (* This op's random-walk config space: a 4D NHWC tensor, cumulatively
     summed over one of the six frame axes. Any axis is valid at any extent
     (same as [Softmax.Walk]: nothing to combine), so [cascade] is identity. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; axis : Axis.t }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        axis = Axis.C;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let params (c : cfg) : params = { axis = c.axis }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "axis" Axis.all (fun c v -> { c with axis = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a axis=%a}" Walk_core.Shape.pp c.shape Axis.pp
        c.axis
  end
end
