(* Assembles AdaptiveAvgPool2d's walk: an ad hoc (n, c, input_h, input_w,
   out_h, out_w) config space, all six drawn independently. Unlike
   e.g. [Recipe_conv]'s correlated axes, no pair here needs correlating:
   [Pool.Adaptive_axis.check] only rejects when input_extent * output_size
   overflows a hard limit far above anything this walk's small literal
   bounds can reach, so every combination in range is valid by construction
   (see the `.ai/` native walk design's F4 for the general rule this is an
   instance of, not an exception to). *)

module M = struct
  type cfg = {
    n : Dim.extent Dim.t;
    c : Dim.extent Dim.t;
    input_h : Dim.extent Dim.t;
    input_w : Dim.extent Dim.t;
    out_h : Op_config.Pos.t;
    out_w : Op_config.Pos.t;
  }

  type subject = Native_subject.t

  let target = "adaptive_avg_pool2d"

  let initial =
    {
      n = Dim.one;
      c = Dim.extent 4;
      input_h = Dim.extent 8;
      input_w = Dim.extent 8;
      out_h = Op_config.Pos.of_int 4;
      out_w = Op_config.Pos.of_int 4;
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
      ]

  let pp fmt c =
    Format.fprintf fmt "{shape=[%d,%d,%d,%d] output_size=[%d,%d]}"
      (c.n :> int)
      (c.c :> int)
      (c.input_h :> int)
      (c.input_w :> int)
      (c.out_h :> int)
      (c.out_w :> int)

  let shape c =
    Vec6.make ~n:c.n ~t:Dim.one ~d:Dim.one ~h:c.input_h ~w:c.input_w ~c:c.c

  let params c : Pool.AdaptiveAvgPool2d.params =
    { output_size = { Op_config.Hw.h = c.out_h; w = c.out_w } }

  let build pcg c =
    let shape = shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"adaptive_avg_pool2d" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          adaptive_avg_pool2d ~name:"out" (params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
