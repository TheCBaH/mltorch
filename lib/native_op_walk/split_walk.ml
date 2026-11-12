(* The random-walk config spaces of the ops in [Split], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Unbind = struct
  open Split.Unbind

  (* This op's random-walk config space: a 4D tensor unbound along one of the
     four mutable axes. [Walk_core.Shape.t] carries no rank and fixes T/D at 1,
     so the axis is a curated field over exactly the axes that shape can vary —
     there is nothing to correlate and [cascade] is the identity. The per-axis
     caps and the numel budget already bound the output count well below
     [output_limit]. Rank/dim correlation belongs in the ATEN recipe, where the
     shape is a rank-bearing list. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; axis : Axis.t }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        axis = Axis.H;
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
          field_axis "axis" Axis.[ N; H; W; C ] (fun r v -> { r with axis = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a axis=%a}" Walk_core.Shape.pp c.shape Axis.pp
        c.axis
  end
end

module Split_with_sizes = struct
  open Split.Split_with_sizes

  (* This op's random-walk config space: a 4D tensor split along one of the
     four mutable axes into a small number of windows. [pattern] resolves
     against whatever extent [Walk_core.Shape.t] draws for [axis] -- the same
     "derive from the drawn extent" shape [Slice.Walk]'s [bounds] uses, and
     for the same reason: an independent [sizes] axis would draw far more
     configurations [output_shapes] refuses (wrong sum) than ones it accepts. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type pattern = All_ones | Halves | Head_and_rest | Thirds
    type cfg = { shape : Walk_core.Shape.t; axis : Axis.t; pattern : pattern }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        axis = Axis.H;
        pattern = Halves;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

    (* Every pattern sums exactly to [n] for any [n >= 1]: the remainder of an
       uneven split lands entirely in the LAST piece, so there is no pattern
       here that [output_shapes] would refuse. *)
    let sizes_for ~n = function
      | All_ones -> List.init n (fun _ -> 1)
      | Halves ->
          let half = max 1 (n / 2) in
          if half >= n then [ n ] else [ half; n - half ]
      | Head_and_rest -> if n <= 1 then [ n ] else [ 1; n - 1 ]
      | Thirds ->
          let third = max 1 (n / 3) in
          if 2 * third >= n then [ third; n - third ]
          else [ third; third; n - (2 * third) ]

    let params (c : cfg) : params =
      let n = (Vec6.get (shape c) c.axis :> int) in
      { axis = c.axis; sizes = List.map Dim.extent (sizes_for ~n c.pattern) }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "axis" Axis.[ N; H; W; C ] (fun r v -> { r with axis = v });
          field_axis "pattern" [ Halves; Thirds; Head_and_rest; All_ones ]
            (fun r v -> { r with pattern = v });
        ]

    let pp fmt (c : cfg) =
      let p = params c in
      Format.fprintf fmt "{shape=%a axis=%a sizes=%a}" Walk_core.Shape.pp
        c.shape Axis.pp p.axis
        (Fmt.brackets (Fmt.list ~sep:Fmt.comma Dim.pp))
        p.sizes
  end
end

module Slice = struct
  open Split.Slice

  (* This op's random-walk config space. The bounds are DERIVED from the drawn
     extent rather than being axes of their own: a (start, stop) pair is only
     valid for a specific extent, so independent axes would generate the empty
     and out-of-range configurations [output_shape] refuses, and a walk of
     refusals reads as coverage. The pattern names a whole intent — take the
     head, take the tail, take every second element — and [bounds] resolves it
     against whatever extent is current, which is why [cascade] is the identity.

     [Prefix] is deliberately the identity slice on any extent. It is the one
     configuration a Default-tier walk would produce, kept here so the golden
     shows it beside the configurations that actually move data. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type pattern = Head | Interior | Prefix | Stride2 | Stride3 | Tail
    type cfg = { shape : Walk_core.Shape.t; axis : Axis.t; pattern : pattern }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        axis = Axis.H;
        pattern = Head;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

    (* Every result is non-empty for any extent >= 1: [Head] and [Tail] keep at
       least one element because [max 1] and [min (n-1)] pin them, and the
       strided forms start at 0. *)
    let bounds ~n = function
      | Head -> (0, max 1 (n / 2), 1)
      | Interior -> (min (n - 1) 1, n, 1)
      | Prefix -> (0, n, 1)
      | Stride2 -> (0, n, 2)
      | Stride3 -> (0, n, 3)
      | Tail -> (min (n - 1) (n / 2), n, 1)

    let params (c : cfg) : params =
      let n = (Vec6.get (shape c) c.axis :> int) in
      let start, stop, step = bounds ~n c.pattern in
      {
        axis = c.axis;
        start = Dim.fence start;
        stop = Dim.fence stop;
        step = Op_config.Pos.of_int step;
      }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "axis" Axis.[ N; H; W; C ] (fun r v -> { r with axis = v });
          field_axis "pattern"
            [ Prefix; Head; Tail; Interior; Stride2; Stride3 ] (fun r v ->
              { r with pattern = v });
        ]

    let pp fmt (c : cfg) =
      let p = params c in
      Format.fprintf fmt "{shape=%a axis=%a start=%a stop=%a step=%d}"
        Walk_core.Shape.pp c.shape Axis.pp p.axis Dim.pp p.start Dim.pp p.stop
        (p.step :> int)
  end
end
