(* Assembles Upsample_bilinear2d's walk: an ad hoc (n, c, input_h, input_w,
   out_h, out_w, align_corners) config space, mirroring
   [Adaptive_avg_pool2d_nwalk]'s independent axes. No pair needs correlating:
   [Resize.Bilinear_axis.check] only rejects when the coordinate-transform
   aggregate overflows a hard limit far above anything this walk's small
   literal bounds can reach, so every combination in range is valid by
   construction (same F4 instance the adaptive-pool walk's header cites). *)

module M = struct
  type cfg = {
    n : Dim.extent Dim.t;
    c : Dim.extent Dim.t;
    input_h : Dim.extent Dim.t;
    input_w : Dim.extent Dim.t;
    out_h : Op_config.Pos.t;
    out_w : Op_config.Pos.t;
    align_corners : bool;
  }

  type subject = Native_subject.t

  let target = "upsample_bilinear2d"

  let initial =
    {
      n = Dim.one;
      c = Dim.extent 4;
      input_h = Dim.extent 8;
      input_w = Dim.extent 8;
      out_h = Op_config.Pos.of_int 4;
      out_w = Op_config.Pos.of_int 4;
      align_corners = true;
    }

  let cascade c = c

  let axes =
    Walk_core.Walk.
      [
        range_axis
          (module Extent)
          "n" ~lo:Extent.one
          ~hi:(Extent.of_int Walk_limits.L.limits.max_batch)
          (fun c v -> { c with n = v });
        range_axis
          (module Extent)
          "c" ~lo:Extent.one ~hi:(Extent.of_int 8)
          (fun c v -> { c with c = v });
        range_axis
          (module Extent)
          "input_h" ~lo:Extent.one ~hi:(Extent.of_int 12)
          (fun c v -> { c with input_h = v });
        range_axis
          (module Extent)
          "input_w" ~lo:Extent.one ~hi:(Extent.of_int 12)
          (fun c v -> { c with input_w = v });
        range_axis
          (module Pos)
          "out_h" ~lo:(Pos.of_int 1) ~hi:(Pos.of_int 7)
          (fun c v -> { c with out_h = v });
        range_axis
          (module Pos)
          "out_w" ~lo:(Pos.of_int 1) ~hi:(Pos.of_int 7)
          (fun c v -> { c with out_w = v });
        field_axis "align_corners" [ true; false ] (fun c v ->
            { c with align_corners = v });
      ]

  let pp fmt c =
    Format.fprintf fmt
      "{shape=[%d,%d,%d,%d] output_size=[%d,%d] align_corners=%b}"
      (c.n :> int)
      (c.c :> int)
      (c.input_h :> int)
      (c.input_w :> int)
      (c.out_h :> int)
      (c.out_w :> int)
      c.align_corners

  let shape c =
    Vec6.make ~n:c.n ~t:Dim.one ~d:Dim.one ~h:c.input_h ~w:c.input_w ~c:c.c

  let params c : Resize.Bilinear2d.params =
    {
      output_size = { Op_config.Hw.h = c.out_h; w = c.out_w };
      align_corners = c.align_corners;
    }

  let build pcg c =
    let shape = shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"upsample_bilinear2d" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          upsample_bilinear2d ~name:"out" (params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
