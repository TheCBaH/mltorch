(* The single definition of [Attention_layer]'s meaning: the subgraph of
   existing Native ops it stands for, over the node's own operands. Every node
   is one an importer or the bridge builds for the source program the layer is
   found in (Linear's [dot from +0, then + bias], the head split as a reshape
   and a permute, the rotation as HF spells it -- two slices, [mul_scalar (-1)]
   for the negation, a concat, two products and a sum -- Sdpa, the merge,
   the output Linear, the layer scale's [mul], the residual's [add]), so the
   node equals its decomposition bit for bit by construction, and the rewrite
   that finds the layer is this function's inverse. *)

module L = Attention_layer.Attention_layer
open Graph_builder

let swap_h_w : Permute.Permute.perm =
  [
    (Axis.N, Axis.N);
    (Axis.T, Axis.T);
    (Axis.D, Axis.D);
    (Axis.H, Axis.W);
    (Axis.W, Axis.H);
    (Axis.C, Axis.C);
  ]

let decompose (t : L.t) : Tensor_id.t Graph_builder.t =
  let* x_shape = shape_of t.L.x in
  let b = Dim.to_int (Vec6.get x_shape Axis.D)
  and l = Dim.to_int (Vec6.get x_shape Axis.W)
  and e = Dim.to_int (Vec6.get x_shape Axis.C) in
  let heads = Op_config.Pos.to_int t.L.params.L.heads in
  let d = e / heads in
  let norm u (n : L.Norm.t) =
    match n.L.Norm.kind with
    | L.Norm.Kind.Layer ->
        layer_norm
          { Norm.LayerNorm.dims = [ Axis.C ]; eps = n.L.Norm.eps }
          ~x:u ?weight:n.L.Norm.weight ?bias:n.L.Norm.bias ()
    | L.Norm.Kind.Rms ->
        rms_norm
          { Norm.RmsNorm.dims = [ Axis.C ]; eps = n.L.Norm.eps }
          ~x:u ?weight:n.L.Norm.weight ()
  in
  let placed p =
    match t.L.norm with
    | Some n when n.L.Norm.placement = p -> Some n
    | Some _ | None -> None
  in
  let project u (p : L.Projection.t) =
    linear
      { Linear.Linear.in_features = Dim.extent e }
      ~x:u ~weight:p.L.Projection.weight ?bias:p.L.Projection.bias ()
  in
  (* [1,1,B,1,L,E] -> [1,1,B,L,heads,d] -> [1,1,B,heads,L,d] *)
  let split z =
    let* r =
      reshape
        { Reshape.Reshape.shape = Vec6.shape ~n:1 ~t:1 ~d:b ~h:l ~w:heads ~c:d }
        z
    in
    permute swap_h_w r
  in
  let half = d / 2 in
  let slice_c start stop z =
    slice
      {
        Split.Slice.axis = Axis.C;
        start = Dim.fence start;
        stop = Dim.fence stop;
        step = Op_config.Pos.of_int 1;
      }
      z
  in
  (* z * cos + cat(-z[d/2:], z[:d/2]) * sin, the tables broadcast over heads *)
  let rotate z =
    match t.L.rope with
    | None -> return z
    | Some r ->
        let* lo = slice_c 0 half z in
        let* hi = slice_c half d z in
        let* neg = mul_scalar (-1.) hi in
        let* rot = concat { Concat.Concat.axis = Axis.C } [ neg; lo ] in
        let* zc = mul z r.L.Rope.cos in
        let* rs = mul rot r.L.Rope.sin in
        add zc rs
  in
  let* u =
    match placed L.Placement.Pre with
    | Some n -> norm t.L.x n
    | None -> return t.L.x
  in
  let* q = project u t.L.q in
  let* q = split q in
  let* q = rotate q in
  let* k = project u t.L.k in
  let* k = split k in
  let* k = rotate k in
  let* v = project u t.L.v in
  let* v = split v in
  let* a =
    sdpa
      { Attention.Sdpa.scale = t.L.params.L.scale }
      ~query:q ~key:k ~value:v ?mask:t.L.mask ()
  in
  (* [1,1,B,heads,L,d] -> [1,1,B,L,heads,d] -> [1,1,B,1,L,E] *)
  let* a = permute swap_h_w a in
  let* a =
    reshape
      { Reshape.Reshape.shape = Vec6.shape ~n:1 ~t:1 ~d:b ~h:1 ~w:l ~c:e }
      a
  in
  let* o = project a t.L.o in
  let* o = match t.L.layer_scale with Some g -> mul o g | None -> return o in
  let* r = add t.L.x o in
  match placed L.Placement.Post with Some n -> norm r n | None -> return r
