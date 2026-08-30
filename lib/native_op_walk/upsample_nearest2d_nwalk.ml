(* Assembles Upsample_nearest2d's walk: an ad hoc (n, c, input_h, input_w,
   out_h, out_w) config space, mirroring [Upsample_bilinear2d_nwalk]'s -- no
   [align_corners] axis, since [upsample_nearest2d.vec] has no such
   parameter (see [Resize.Nearest_axis]'s module doc). No pair needs
   correlating: [Resize.Nearest_axis.check] only rejects when the
   coordinate-transform aggregate overflows a hard limit far above anything
   this walk's small literal bounds can reach, same F4 instance
   [Upsample_bilinear2d_nwalk]'s own header cites. *)

module M = struct
  type cfg = {
    n : int;
    c : int;
    input_h : int;
    input_w : int;
    out_h : int;
    out_w : int;
  }

  type subject = Native_subject.t

  let target = "upsample_nearest2d"
  let initial = { n = 1; c = 4; input_h = 8; input_w = 8; out_h = 4; out_w = 4 }
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
      ]

  let pp fmt c =
    Format.fprintf fmt "{shape=[%d,%d,%d,%d] output_size=[%d,%d]}" c.n c.c
      c.input_h c.input_w c.out_h c.out_w

  let shape c = Vec6.shape ~n:c.n ~t:1 ~d:1 ~h:c.input_h ~w:c.input_w ~c:c.c

  let params c : Resize.Nearest2d.params =
    {
      output_size =
        {
          Op_config.Hw.h = Op_config.Pos.of_int c.out_h;
          w = Op_config.Pos.of_int c.out_w;
        };
    }

  let build pcg c =
    let shape = shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"upsample_nearest2d" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          upsample_nearest2d ~name:"out" (params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
