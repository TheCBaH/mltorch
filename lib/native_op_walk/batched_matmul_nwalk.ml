(* Assembles Batched_matmul's walk: an ad hoc (d, h, n, m, p) config space,
   mirroring [Bmm_nwalk] but with an extra [d] batch axis alongside [h] --
   [Matmul.Batched_matmul] has no shared Walk functor either, for the same
   reason [Bmm] doesn't. [d]/[h] both reuse [max_batch] the same way [Bmm]'s
   own [batch] does. *)

module M = struct
  type cfg = {
    d : Dim.extent Dim.t;
    heads : Dim.extent Dim.t;
    n : Dim.extent Dim.t;
    m : Dim.extent Dim.t;
    p : Dim.extent Dim.t;
  }

  type subject = Native_subject.t

  let target = "batched_matmul"

  let initial =
    {
      d = Dim.one;
      heads = Dim.extent 2;
      n = Dim.extent 2;
      m = Dim.extent 3;
      p = Dim.extent 4;
    }

  let cascade c = c

  let axes =
    Walk_core.Walk.
      [
        range_axis
          (module Extent)
          "d" ~lo:Extent.one
          ~hi:(Extent.of_int Walk_limits.L.limits.max_batch)
          (fun c v -> { c with d = v });
        range_axis
          (module Extent)
          "heads" ~lo:Extent.one
          ~hi:(Extent.of_int Walk_limits.L.limits.max_batch)
          (fun c v -> { c with heads = v });
        range_axis
          (module Extent)
          "n" ~lo:Extent.one ~hi:(Extent.of_int 8)
          (fun c v -> { c with n = v });
        range_axis
          (module Extent)
          "m" ~lo:Extent.one ~hi:(Extent.of_int 8)
          (fun c v -> { c with m = v });
        range_axis
          (module Extent)
          "p" ~lo:Extent.one ~hi:(Extent.of_int 8)
          (fun c v -> { c with p = v });
      ]

  let pp fmt c =
    Format.fprintf fmt "{d=%d heads=%d n=%d m=%d p=%d}"
      (c.d :> int)
      (c.heads :> int)
      (c.n :> int)
      (c.m :> int)
      (c.p :> int)

  let input_shape c =
    Vec6.make ~n:Dim.one ~t:Dim.one ~d:c.d ~h:c.heads ~w:c.n ~c:c.m

  let mat2_shape c =
    Vec6.make ~n:Dim.one ~t:Dim.one ~d:c.d ~h:c.heads ~w:c.m ~c:c.p

  let build pcg c =
    let input_v, pcg = Native_tensor.synth pcg (input_shape c) in
    let mat2_v, pcg = Native_tensor.synth pcg (mat2_shape c) in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"batched_matmul" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape:(input_shape c) ~name:"input" () in
          let* yi = input ~shape:(mat2_shape c) ~name:"mat2" () in
          batched_matmul ~name:"out" xi yi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ input_v; mat2_v ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
