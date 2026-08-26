(* Assembles Upsample_bilinear2d's walk: an ad hoc (n, c, input_h, input_w,
   out_h, out_w, align_corners) config space, mirroring
   [Adaptive_avg_pool2d_nwalk]'s independent axes. No pair needs correlating:
   [Resize.Bilinear_axis.check] only rejects when the coordinate-transform
   aggregate overflows a hard limit far above anything this walk's small
   literal bounds can reach, so every combination in range is valid by
   construction (same F4 instance the adaptive-pool walk's header cites). *)

module M = struct
  type cfg = {
    n : int;
    c : int;
    input_h : int;
    input_w : int;
    out_h : int;
    out_w : int;
    align_corners : bool;
  }

  type subject = Native_subject.t

  let target = "upsample_bilinear2d"

  let initial =
    {
      n = 1;
      c = 4;
      input_h = 8;
      input_w = 8;
      out_h = 4;
      out_w = 4;
      align_corners = true;
    }

  let cascade c = c

  let axes =
    Walk_core.Walk.
      [
        int_axis "n" ~lo:1 ~hi:Walk_limits.L.limits.max_batch (fun c v ->
            { c with n = v });
        int_axis "c" ~lo:1 ~hi:8 (fun c v -> { c with c = v });
        int_axis "input_h" ~lo:1 ~hi:12 (fun c v -> { c with input_h = v });
        int_axis "input_w" ~lo:1 ~hi:12 (fun c v -> { c with input_w = v });
        int_axis "out_h" ~lo:1 ~hi:7 (fun c v -> { c with out_h = v });
        int_axis "out_w" ~lo:1 ~hi:7 (fun c v -> { c with out_w = v });
        field_axis "align_corners" [ true; false ] (fun c v ->
            { c with align_corners = v });
      ]

  let pp fmt c =
    Format.fprintf fmt
      "{shape=[%d,%d,%d,%d] output_size=[%d,%d] align_corners=%b}" c.n c.c
      c.input_h c.input_w c.out_h c.out_w c.align_corners

  let shape c = Vec6.shape ~n:c.n ~t:1 ~d:1 ~h:c.input_h ~w:c.input_w ~c:c.c

  let params c : Resize.Bilinear2d.params =
    {
      output_size =
        {
          Op_config.Hw.h = Op_config.Pos.of_int c.out_h;
          w = Op_config.Pos.of_int c.out_w;
        };
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
