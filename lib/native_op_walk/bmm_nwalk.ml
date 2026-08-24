(* Assembles Bmm's walk: an ad hoc (batch, n, m, p) config space. Native's Bmm
   has no shared Walk functor in lib/native/ops/matmul.ml (unlike the
   pointwise/pool ops this directory otherwise reuses), so the axes are
   defined directly here, mirroring the shape contract documented on
   [Matmul.Bmm]: input[H=batch,W=n,C=m] x mat2[H=batch,W=m,C=p]. Bounds are
   small literals, not [Walk_limits] (there is no per-matrix-dim cap in
   [Limits.t]), except [batch], which reuses [max_batch] since it maps onto
   the same [N]-like role the rest of this directory bounds that way. *)

module M = struct
  type cfg = { batch : int; n : int; m : int; p : int }
  type subject = Native_subject.t

  let target = "bmm"
  let initial = { batch = 1; n = 2; m = 3; p = 4 }
  let cascade c = c

  let axes =
    Walk_core.Walk.
      [
        int_axis "batch" ~lo:1 ~hi:Walk_limits.L.limits.max_batch (fun c v ->
            { c with batch = v });
        int_axis "n" ~lo:1 ~hi:8 (fun c v -> { c with n = v });
        int_axis "m" ~lo:1 ~hi:8 (fun c v -> { c with m = v });
        int_axis "p" ~lo:1 ~hi:8 (fun c v -> { c with p = v });
      ]

  let pp fmt c =
    Format.fprintf fmt "{batch=%d n=%d m=%d p=%d}" c.batch c.n c.m c.p

  let input_shape c = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:c.batch ~w:c.n ~c:c.m
  let mat2_shape c = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:c.batch ~w:c.m ~c:c.p

  let build pcg c =
    let input_v, pcg = Native_tensor.synth pcg (input_shape c) in
    let mat2_v, pcg = Native_tensor.synth pcg (mat2_shape c) in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"bmm" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape:(input_shape c) ~name:"input" () in
          let* yi = input ~shape:(mat2_shape c) ~name:"mat2" () in
          bmm ~name:"out" xi yi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ input_v; mat2_v ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
