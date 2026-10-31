(* Find an encoder attention layer spelled as [Attention_layer_decompose] spells
   it, and replace the subgraph with the one [Attention_layer] node it stands
   for: the pass is [decompose]'s inverse, so the node equals the subgraph it
   replaces bit for bit and the output's claim is [Identical]. Off by default:
   no pipeline runs it; a caller that grants the layer does.

   The match is anchored on the layer's output and walks upstream: an optional
   Post norm, the residual [add], an optional layer-scale [mul], the output
   Linear, the head merge (permute, reshape), the Sdpa, and per projection a
   Linear, the head split (reshape, permute) and, for Q and K, the rotate-half
   rotation (two slices, [mul_scalar (-1)], concat, two products, a sum). Every
   interior value must have no consumer outside the match and not be a graph
   output; anything else -- a second consumer, a boolean mask, a rotation over
   part of the head, a norm over other axes -- leaves the graph as it was. *)

open Graph_ir
module L = Attention_layer.Attention_layer

(* The attention at the layer's core: an Sdpa node, or the eager spelling of
   it ([matmul], a scale, an optional mask [add], [softmax], [matmul]). *)
module Core = struct
  (* Alphabetical. *)
  type form =
    | Eager_div  (** [div c]: normalized to [* RN(1/c)], not bitwise (D13) *)
    | Eager_mul  (** [mul c]: equal in exact arithmetic, rounding differs *)
    | Fused  (** an Sdpa node: the node's own decomposition, bit for bit *)

  type t = {
    query : Tensor_id.t;
    key : Tensor_id.t;
    value : Tensor_id.t;
    mask : Tensor_id.t option;
    scale : Attention.Sdpa.Scale.t;
    form : form;
  }
end

(* fastvit's spelling: the layer in the NCHW frame around a token-frame core,
   one fused qkv Linear, the layer scale and the residual in NCHW. What the
   rewrite inserts to give the node token-frame operands. *)
module Nchw = struct
  type t = {
    qkv : Tensor_id.t;  (** the [3E, E] projection weight, sliced three ways *)
    residual : Tensor_id.t;  (** [1,1,1,E,h,w] *)
    gamma : Tensor_id.t;  (** [1,1,1,E,1,1] *)
    width : int;
    tokens : int;
    frame : Vec6.shape;  (** the output's NCHW shape *)
  }
end

type match_ = {
  output : Tensor_id.t;
  output_sig : Tensor_sig.t;
  layer : L.t;
  nodes : Node_id.t list;
  form : Core.form;
  nchw : Nchw.t option;
}

let as_add = function Add (b : Pointwise.Bin.t) -> Some b | _ -> None
let as_unbind = function Unbind (u : Split.Unbind.t) -> Some u | _ -> None
let as_mul = function Mul (b : Pointwise.Bin.t) -> Some b | _ -> None
let as_linear = function Linear (l : Linear.Linear.t) -> Some l | _ -> None

let as_reshape = function
  | Reshape (r : Reshape.Reshape.t) -> Some r
  | _ -> None

let as_permute = function
  | Permute (p : Permute.Permute.t) -> Some p
  | _ -> None

let as_sdpa = function Sdpa (s : Attention.Sdpa.t) -> Some s | _ -> None
let as_slice = function Slice (s : Split.Slice.t) -> Some s | _ -> None
let as_concat = function Concat (c : Concat.Concat.t) -> Some c | _ -> None

let as_bmm = function
  | Batched_matmul (b : Matmul.Batched_matmul.t) -> Some b
  | _ -> None

let as_softmax = function Softmax (x : Reduce.Softmax.t) -> Some x | _ -> None

let as_div_scalar = function
  | Div_scalar (m : Pointwise.Scalar_bin.t) -> Some m
  | _ -> None

let as_mul_scalar = function
  | Mul_scalar (m : Pointwise.Scalar_bin.t) -> Some m
  | _ -> None

let as_norm = function
  | Layer_norm (n : Norm.LayerNorm.t) when n.params.dims = [ Axis.C ] ->
      Some
        ( L.Norm.Kind.Layer,
          n.Norm.LayerNorm.x,
          n.Norm.LayerNorm.params.Norm.LayerNorm.eps,
          n.Norm.LayerNorm.weight,
          n.Norm.LayerNorm.bias )
  | Rms_norm (n : Norm.RmsNorm.t) when n.params.dims = [ Axis.C ] ->
      Some
        ( L.Norm.Kind.Rms,
          n.Norm.RmsNorm.x,
          n.Norm.RmsNorm.params.Norm.RmsNorm.eps,
          n.Norm.RmsNorm.weight,
          None )
  | _ -> None

let swaps_h_w perm =
  List.for_all
    (fun a ->
      Axis.equal
        (Permute.Permute.lookup perm a)
        (match a with Axis.H -> Axis.W | Axis.W -> Axis.H | a -> a))
    Axis.all

let extent (sg : Tensor_sig.t) axis =
  Dim.to_int (Vec6.get sg.Tensor_sig.shape axis)

open Pattern

(* the edge's producer, a projector away, with the edge consumed only inside *)
let inner edge proj =
  let* () = interior edge in
  def edge proj

let as_clone = function Clone (c : Pointwise.Clone.t) -> Some c | _ -> None

(* An exporter's [clone]s (dropout at inference, a contiguous copy) are the
   identity: an interior chain of them is consumed on the way upstream. *)
let rec unclone edge =
  (let* (c : Pointwise.Clone.t), _ = inner edge as_clone in
   unclone c.Pointwise.Clone.x)
  <|> return edge

(* [z] -> [Linear(u)] through the head split; yields the projection, [u] and the
   head count *)
let split_heads z =
  let* (p : Permute.Permute.t), _ = def z as_permute in
  let* () = guard (swaps_h_w p.Permute.Permute.perm) in
  let* (r : Reshape.Reshape.t), _ = inner p.Permute.Permute.x as_reshape in
  let* (l : Linear.Linear.t), _ = inner r.Reshape.Reshape.x as_linear in
  let* split = sig_of r.Reshape.Reshape.x in
  let heads =
    Dim.to_int (Vec6.get r.Reshape.Reshape.params.Reshape.Reshape.shape Axis.W)
  in
  let* () = guard (extent split Axis.C mod heads = 0) in
  return
    ( {
        L.Projection.weight = l.Linear.Linear.weight;
        bias = l.Linear.Linear.bias;
      },
      l.Linear.Linear.x,
      heads )

(* the rotate-half rotation of [z]: [z * cos + cat(-z[d/2:], z[:d/2]) * sin] *)
let rotation sum =
  let* (s : Pointwise.Bin.t), _ = def sum as_add in
  let* (zc : Pointwise.Bin.t), _ = inner s.a as_mul in
  let* (rs : Pointwise.Bin.t), _ = inner s.b as_mul in
  let z = zc.a and cos = zc.b and sin = rs.b in
  let* (c : Concat.Concat.t), _ = inner rs.a as_concat in
  let* () =
    guard (Axis.equal c.Concat.Concat.params.Concat.Concat.axis Axis.C)
  in
  let* neg, lo =
    match c.Concat.Concat.xs with
    | [ n; l ] -> return (n, l)
    | _ -> fail Rejected
  in
  let* (m : Pointwise.Scalar_bin.t), _ = inner neg as_mul_scalar in
  let* () = guard (m.Pointwise.Scalar_bin.scalar = -1.) in
  let* (hi_s : Split.Slice.t), _ = inner m.Pointwise.Scalar_bin.x as_slice in
  let* (lo_s : Split.Slice.t), _ = inner lo as_slice in
  let* zs = sig_of z in
  let d = extent zs Axis.C in
  let half (sl : Split.Slice.t) start =
    let p = sl.Split.Slice.params in
    Axis.equal p.Split.Slice.axis Axis.C
    && Dim.to_int p.Split.Slice.start = start
    && Dim.to_int p.Split.Slice.stop = start + (d / 2)
    && Op_config.Pos.to_int p.Split.Slice.step = 1
    && Tensor_id.equal sl.Split.Slice.x z
  in
  (* the full head dimension, halves swapped: anything else is another
     rotation (a partial or interleaved one) and is left alone *)
  let* () = guard (d mod 2 = 0 && half hi_s (d / 2) && half lo_s 0) in
  (* [z] feeds exactly the product and the two slices *)
  let* zu = uses z in
  let* () = guard (List.length zu = 3) in
  let* o = is_graph_output z in
  let* () = guard (not o) in
  return (z, { L.Rope.cos; sin })

let swaps_w_c perm =
  List.for_all
    (fun a ->
      Axis.equal
        (Permute.Permute.lookup perm a)
        (match a with Axis.W -> Axis.C | Axis.C -> Axis.W | a -> a))
    Axis.all

let fused ctx =
  let* (s : Attention.Sdpa.t), _ = inner ctx as_sdpa in
  return
    {
      Core.query = s.Attention.Sdpa.query;
      key = s.Attention.Sdpa.key;
      value = s.Attention.Sdpa.value;
      mask = s.Attention.Sdpa.mask;
      scale = s.Attention.Sdpa.params.Attention.Sdpa.scale;
      form = Core.Fused;
    }

(* [softmax((Q K^T) op c [+ mask]) V], op a [mul] or a [div] by a constant *)
let eager ctx =
  let* (pv : Matmul.Batched_matmul.t), _ = inner ctx as_bmm in
  let* (sm : Reduce.Softmax.t), _ =
    let* probs = unclone pv.Matmul.Batched_matmul.input in
    inner probs as_softmax
  in
  let* () =
    guard (Axis.equal sm.Reduce.Softmax.params.Reduce.Softmax.axis Axis.C)
  in
  let z = sm.Reduce.Softmax.x in
  let* scaled, mask =
    (let* (a : Pointwise.Bin.t), _ = inner z as_add in
     return (a.a, Some a.b))
    <|> return (z, None)
  in
  let* s, scale, form =
    (let* (m : Pointwise.Scalar_bin.t), _ = inner scaled as_mul_scalar in
     return
       (m.Pointwise.Scalar_bin.x, m.Pointwise.Scalar_bin.scalar, Core.Eager_mul))
    <|>
    let* (m : Pointwise.Scalar_bin.t), _ = inner scaled as_div_scalar in
    let c = m.Pointwise.Scalar_bin.scalar in
    let* () = guard (c <> 0. && Float.is_finite c) in
    return
      (m.Pointwise.Scalar_bin.x, Json_util.f32_to_f32 (1. /. c), Core.Eager_div)
  in
  let* () = guard (Float.is_finite scale && scale > 0.) in
  let* (qk : Matmul.Batched_matmul.t), _ = inner s as_bmm in
  let* (kt : Permute.Permute.t), _ =
    inner qk.Matmul.Batched_matmul.mat2 as_permute
  in
  let* () = guard (swaps_w_c kt.Permute.Permute.perm) in
  return
    {
      Core.query = qk.Matmul.Batched_matmul.input;
      key = kt.Permute.Permute.x;
      value = pv.Matmul.Batched_matmul.mat2;
      mask;
      scale = Attention.Sdpa.Scale.Explicit scale;
      form;
    }

(* [prescaled]: the query may be multiplied by a constant before its rotation
   (esm2 scales Q there and multiplies the scores by 1.0) *)
let query_or_key ?(prescaled = false) edge =
  let* rope = optional (rotation edge) in
  let z = match rope with Some (z, _) -> z | None -> edge in
  let* z, prescale =
    if prescaled then
      (* the rotation already proved its input's consumers are its own *)
      (let* (m : Pointwise.Scalar_bin.t), _ =
         if rope <> None then def z as_mul_scalar else inner z as_mul_scalar
       in
       return (m.Pointwise.Scalar_bin.x, Some m.Pointwise.Scalar_bin.scalar))
      <|> return (z, None)
    else return (z, None)
  in
  let* proj, u, heads = split_heads z in
  return (proj, u, heads, Option.map snd rope, prescale)

(* the residual's other operand: the output projection, optionally scaled *)
let output_branch y =
  let* y = unclone y in
  let scaled =
    let* (m : Pointwise.Bin.t), _ = inner y as_mul in
    let* a = unclone m.a in
    let* (l : Linear.Linear.t), _ = inner a as_linear in
    return (l, Some m.b)
  in
  let plain =
    let* (l : Linear.Linear.t), _ = inner y as_linear in
    return (l, None)
  in
  scaled <|> plain

let pattern anchor =
  let post =
    let* n, _ = def anchor as_norm in
    return n
  in
  let* post = optional post in
  let r = match post with Some (_, x, _, _, _) -> x | None -> anchor in
  let* () =
    match post with
    | Some _ -> interior r
    | None ->
        (* a sum read only by a norm over C is a Post layer's: the match
           anchored on the norm takes it whole *)
        let* consumers = uses r in
        let* out = is_graph_output r in
        guard
          (not
             ((not out)
             &&
             match consumers with
             | [ n ] -> Option.is_some (as_norm n.Graph_common.Node.op)
             | _ -> false))
  in
  let* (res : Pointwise.Bin.t), _ = def r as_add in
  (* the residual on either side of the sum *)
  let* x, ((lo : Linear.Linear.t), gamma) =
    (let+ b = output_branch res.b in
     (res.a, b))
    <|>
    let+ a = output_branch res.a in
    (res.b, a)
  in
  let* merged = unclone lo.Linear.Linear.x in
  let* (mr : Reshape.Reshape.t), _ = inner merged as_reshape in
  let* heads_out = unclone mr.Reshape.Reshape.x in
  let* (mp : Permute.Permute.t), _ = inner heads_out as_permute in
  let* () = guard (swaps_h_w mp.Permute.Permute.perm) in
  let* (core : Core.t) =
    fused mp.Permute.Permute.x <|> eager mp.Permute.Permute.x
  in
  let* () = interior core.Core.query in
  let* () = interior core.Core.key in
  let* () = interior core.Core.value in
  let* q, uq, hq, rq, prescale =
    query_or_key ~prescaled:(core.Core.form <> Core.Fused) core.Core.query
  in
  let* k, uk, hk, rk, _ = query_or_key core.Core.key in
  (* a scale on the query moves after the product (D13): a multiply by one
     there leaves the arithmetic equal in exact terms, anything else is the
     product of two roundings *)
  let* core =
    match (prescale, core.Core.scale) with
    | None, _ -> return core
    | Some c, Attention.Sdpa.Scale.Explicit s ->
        return
          {
            core with
            Core.scale =
              Attention.Sdpa.Scale.Explicit (Json_util.f32_to_f32 (c *. s));
            form = (if s = 1. then core.Core.form else Core.Eager_div);
          }
    | Some _, Attention.Sdpa.Scale.Default -> fail Rejected
  in
  let* v, uv, hv = split_heads core.Core.value in
  let* () =
    guard (Tensor_id.equal uq uk && Tensor_id.equal uq uv && hq = hk && hq = hv)
  in
  (* one rotation table pair for both, or none *)
  let* rope =
    match (rq, rk) with
    | None, None -> return None
    | Some a, Some b
      when Tensor_id.equal a.L.Rope.cos b.L.Rope.cos
           && Tensor_id.equal a.L.Rope.sin b.L.Rope.sin ->
        return (Some a)
    | _ -> fail Rejected
  in
  (* an additive binary32 mask only *)
  let* () =
    match core.Core.mask with
    | None -> return ()
    | Some m ->
        let* ms = sig_of m in
        guard (ms.Tensor_sig.fmt = Payload.Fmt Payload.F32)
  in
  (* a Pre norm between x and the projections, read by the three of them only *)
  let pre =
    let* n, _ = def uq as_norm in
    let* uses = uses uq in
    let* () = guard (List.length uses = 3) in
    let* o = is_graph_output uq in
    let* () = guard (not o) in
    return n
  in
  let* norm, src =
    match post with
    | Some (kind, _, eps, weight, bias) ->
        return
          ( Some { L.Norm.bias; eps; kind; placement = L.Placement.Post; weight },
            uq )
    | None -> (
        let* pre = optional pre in
        match pre with
        | Some (kind, src, eps, weight, bias) ->
            return
              ( Some
                  {
                    L.Norm.bias;
                    eps;
                    kind;
                    placement = L.Placement.Pre;
                    weight;
                  },
                src )
        | None -> return (None, uq))
  in
  let* () = guard (Tensor_id.equal src x) in
  let* claimed = claimed in
  let layer =
    {
      L.k;
      residual = None;
      layer_scale = gamma;
      mask = core.Core.mask;
      norm;
      o =
        {
          L.Projection.weight = lo.Linear.Linear.weight;
          bias = lo.Linear.Linear.bias;
        };
      params = { L.heads = Op_config.Pos.of_int hq; scale = core.Core.scale };
      q;
      rope;
      v;
      x;
    }
  in
  (* the node's own shape rule: a layer the node cannot carry is left alone *)
  let rec shapes acc = function
    | [] -> return acc
    | r :: rest ->
        let* sg = sig_of r in
        shapes ((r, sg.Tensor_sig.shape) :: acc) rest
  in
  let* shapes = shapes [] (L.operands layer) in
  let shape_of r =
    snd (List.find (fun (id, _) -> Tensor_id.equal id r) shapes)
  in
  let* () =
    guard (Result.is_ok (Err.payload (L.output_shape layer ~shape_of)))
  in
  let* output_sig = sig_of anchor in
  return
    {
      output = anchor;
      output_sig;
      layer;
      nodes = Node_id.Set.elements claimed;
      form = core.Core.form;
      nchw = None;
    }

(* The fused form is the node's own decomposition: the output keeps its id,
   [Identical]. An eager form computes the same in exact arithmetic but rounds
   differently ([Eager_mul]: [Equivalent]), or takes the scale's reciprocal
   rounded ([Eager_div]: [Approximate], D13), so the node defines a fresh id
   the consumers are rewired to. *)
let swap_w_c_perm : Permute.Permute.perm =
  Axis.[ (N, N); (T, T); (D, D); (H, H); (W, C); (C, W) ]

(* The output in the NCHW frame: [x_nchw + gamma * view(Linear_o(merge(Sdpa(
   unbind(view(Linear_qkv(x_tokens)))))))], with x_tokens and the residual
   different tensors (the residual is the BatchNorm's input). *)
let nchw anchor =
  let* (res : Pointwise.Bin.t), _ = def anchor as_add in
  let scaled_branch residual scaled =
    let* (m : Pointwise.Bin.t), _ = inner scaled as_mul in
    let* (r : Reshape.Reshape.t), _ = inner m.a as_reshape in
    let* (p : Permute.Permute.t), _ = inner r.Reshape.Reshape.x as_permute in
    let* () = guard (swaps_w_c p.Permute.Permute.perm) in
    let* y = unclone p.Permute.Permute.x in
    return (residual, m.b, r.Reshape.Reshape.params.Reshape.Reshape.shape, y)
  in
  let* residual, gamma, frame, y =
    scaled_branch res.a res.b <|> scaled_branch res.b res.a
  in
  let* (lo : Linear.Linear.t), _ = inner y as_linear in
  let* merged = unclone lo.Linear.Linear.x in
  let* (mr : Reshape.Reshape.t), _ = inner merged as_reshape in
  let* (mp : Permute.Permute.t), _ = inner mr.Reshape.Reshape.x as_permute in
  let* () = guard (swaps_h_w mp.Permute.Permute.perm) in
  let* (s : Attention.Sdpa.t), _ = inner mp.Permute.Permute.x as_sdpa in
  let* () = guard (s.Attention.Sdpa.mask = None) in
  let* () = interior s.Attention.Sdpa.query in
  let* () = interior s.Attention.Sdpa.key in
  let* () = interior s.Attention.Sdpa.value in
  let* (u : Split.Unbind.t), un = def s.Attention.Sdpa.query as_unbind in
  let* () =
    guard
      (un.Graph_common.Node.outputs
       = [
           s.Attention.Sdpa.query; s.Attention.Sdpa.key; s.Attention.Sdpa.value;
         ]
      && Axis.equal u.Split.Unbind.params.Split.Unbind.axis Axis.T)
  in
  let* (p3 : Permute.Permute.t), _ = inner u.Split.Unbind.x as_permute in
  let* (r3 : Reshape.Reshape.t), _ = inner p3.Permute.Permute.x as_reshape in
  let* (lq : Linear.Linear.t), _ = inner r3.Reshape.Reshape.x as_linear in
  let* () = guard (lq.Linear.Linear.bias = None) in
  let* xs = sig_of lq.Linear.Linear.x in
  let* qs = sig_of s.Attention.Sdpa.query in
  let e = Dim.to_int (Vec6.get xs.Tensor_sig.shape Axis.C)
  and l = Dim.to_int (Vec6.get xs.Tensor_sig.shape Axis.W) in
  let heads = Dim.to_int (Vec6.get qs.Tensor_sig.shape Axis.H)
  and d = Dim.to_int (Vec6.get qs.Tensor_sig.shape Axis.C) in
  (* the split must put Q, K, V each as [heads, d] of one third of the
     features: the reshape's [L, 3, heads, d] and the permute to
     [3, 1, heads, L, d] *)
  let rshape = r3.Reshape.Reshape.params.Reshape.Reshape.shape in
  let at a = Dim.to_int (Vec6.get rshape a) in
  let* () =
    guard
      (at Axis.D = l
      && at Axis.H = 3
      && at Axis.W = heads
      && at Axis.C = d
      && heads * d = e
      && Permute.Permute.lookup p3.Permute.Permute.perm Axis.T = Axis.H
      && Permute.Permute.lookup p3.Permute.Permute.perm Axis.H = Axis.W
      && Permute.Permute.lookup p3.Permute.Permute.perm Axis.W = Axis.D)
  in
  let* rs = sig_of residual in
  let* gs = sig_of gamma in
  let* () =
    guard
      (Dim.to_int (Vec6.get rs.Tensor_sig.shape Axis.H) = e
      && Dim.to_int (Vec6.get gs.Tensor_sig.shape Axis.H) = e
      && Vec6.numel gs.Tensor_sig.shape
         = Vec6.numel (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:e)
      && Vec6.numel rs.Tensor_sig.shape = Vec6.numel xs.Tensor_sig.shape)
  in
  let* claimed = claimed in
  let* output_sig = sig_of anchor in
  let proj = { L.Projection.weight = lq.Linear.Linear.weight; bias = None } in
  return
    {
      output = anchor;
      output_sig;
      layer =
        {
          L.k = proj;
          residual = None;
          layer_scale = None;
          mask = None;
          norm = None;
          o =
            {
              L.Projection.weight = lo.Linear.Linear.weight;
              bias = lo.Linear.Linear.bias;
            };
          params =
            {
              L.heads = Op_config.Pos.of_int heads;
              scale = s.Attention.Sdpa.params.Attention.Sdpa.scale;
            };
          q = proj;
          rope = None;
          v = proj;
          x = lq.Linear.Linear.x;
        };
      nodes = Node_id.Set.elements claimed;
      form = Core.Fused;
      nchw =
        Some
          {
            Nchw.qkv = lq.Linear.Linear.weight;
            residual;
            gamma;
            width = e;
            tokens = l;
            frame;
          };
    }

let pattern anchor = pattern anchor <|> nchw anchor

(* The fastvit form: the qkv weight sliced into three, the residual and the
   layer scale viewed in the token frame, the node, and its output viewed back
   in the NCHW frame under the anchor's id. Each inserted node moves values
   without arithmetic, and the node's projections compute each output's sum
   as the fused one did, so the anchor is [Identical]. *)
let build_nchw m (n : Nchw.t) =
  let open Recipe in
  let* out = existing m.output in
  let s6 = Vec6.shape and e = n.Nchw.width and l = n.Nchw.tokens in
  let weight = s6 ~n:e ~t:1 ~d:1 ~h:1 ~w:1 ~c:e in
  let* wq = fresh weight in
  let* wk = fresh weight in
  let* wv = fresh weight in
  let* res_t = fresh (s6 ~n:1 ~t:1 ~d:1 ~h:1 ~w:e ~c:l) in
  let* res = fresh (s6 ~n:1 ~t:1 ~d:1 ~h:1 ~w:l ~c:e) in
  let* gamma = fresh (s6 ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:e) in
  let* y = fresh (s6 ~n:1 ~t:1 ~d:1 ~h:1 ~w:l ~c:e) in
  let* y_t = fresh (s6 ~n:1 ~t:1 ~d:1 ~h:1 ~w:e ~c:l) in
  let f = raw_fresh in
  let slice third w =
    {
      Recipe.op =
        Slice
          {
            Split.Slice.params =
              {
                Split.Slice.axis = Axis.N;
                start = Dim.fence (third * e);
                stop = Dim.fence ((third + 1) * e);
                step = Op_config.Pos.of_int 1;
              };
            x = n.Nchw.qkv;
          };
      outputs = [ Fresh w ];
      from = m.nodes;
    }
  in
  let node op target = { Recipe.op; outputs = [ target ]; from = m.nodes } in
  let proj w = { L.Projection.weight = f w; bias = None } in
  let layer =
    {
      m.layer with
      L.q = proj wq;
      k = proj wk;
      v = proj wv;
      residual = Some (f res);
      layer_scale = Some (f gamma);
    }
  in
  emit
    {
      empty_replacement with
      remove = Node_id.Set.of_list m.nodes;
      insert =
        [
          slice 0 wq;
          slice 1 wk;
          slice 2 wv;
          node
            (Reshape
               {
                 Reshape.Reshape.params =
                   { Reshape.Reshape.shape = s6 ~n:1 ~t:1 ~d:1 ~h:1 ~w:e ~c:l };
                 x = n.Nchw.residual;
               })
            (Fresh res_t);
          node
            (Permute { Permute.Permute.perm = swap_w_c_perm; x = f res_t })
            (Fresh res);
          node
            (Reshape
               {
                 Reshape.Reshape.params =
                   { Reshape.Reshape.shape = s6 ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:e };
                 x = n.Nchw.gamma;
               })
            (Fresh gamma);
          node (Attention_layer layer) (Fresh y);
          node
            (Permute { Permute.Permute.perm = swap_w_c_perm; x = f y })
            (Fresh y_t);
          node
            (Reshape
               {
                 Reshape.Reshape.params =
                   { Reshape.Reshape.shape = n.Nchw.frame };
                 x = f y_t;
               })
            (Preserved out);
        ];
      value_claims = [ (out, Preserved out, Correspondence.Identical) ];
    }

let build_tokens m _region =
  let open Recipe in
  let* out = existing m.output in
  let insert target =
    [
      {
        Recipe.op = Attention_layer m.layer;
        outputs = [ target ];
        from = m.nodes;
      };
    ]
  in
  match m.form with
  | Core.Fused ->
      replace ~remove:m.nodes ~insert:(insert (Preserved out))
        ~claims:[ (out, Preserved out, Correspondence.Identical) ]
        ()
  | Core.Eager_div | Core.Eager_mul ->
      let* y =
        fresh ~fmt:m.output_sig.Tensor_sig.fmt m.output_sig.Tensor_sig.shape
      in
      let relation =
        match m.form with
        | Core.Eager_mul -> Correspondence.Equivalent
        | Core.Eager_div | Core.Fused ->
            Correspondence.Approximate Correspondence.Precision.Set.empty
      in
      emit
        {
          empty_replacement with
          remove = Node_id.Set.of_list m.nodes;
          insert = insert (Fresh y);
          subst = [ (Old out, New y) ];
          value_claims = [ (out, Fresh y, relation) ];
        }

let build m region =
  match m.nchw with Some n -> build_nchw m n | None -> build_tokens m region

let pass =
  Pass.of_pattern ~name:"fuse_attention_layer" ~pattern ~build:{ Pass.build }
