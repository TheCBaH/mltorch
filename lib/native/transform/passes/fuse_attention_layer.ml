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

type match_ = {
  output : Tensor_id.t;
  output_sig : Tensor_sig.t;
  layer : L.t;
  nodes : Node_id.t list;
  form : Core.form;
}

let as_add = function Add (b : Pointwise.Bin.t) -> Some b | _ -> None
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
    inner pv.Matmul.Batched_matmul.input as_softmax
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

let query_or_key edge =
  let* rope = optional (rotation edge) in
  let z = match rope with Some (z, _) -> z | None -> edge in
  let* proj, u, heads = split_heads z in
  return (proj, u, heads, Option.map snd rope)

(* the residual's other operand: the output projection, optionally scaled *)
let output_branch y =
  let scaled =
    let* (m : Pointwise.Bin.t), _ = inner y as_mul in
    let* (l : Linear.Linear.t), _ = inner m.a as_linear in
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
  let* (mr : Reshape.Reshape.t), _ = inner lo.Linear.Linear.x as_reshape in
  let* (mp : Permute.Permute.t), _ = inner mr.Reshape.Reshape.x as_permute in
  let* () = guard (swaps_h_w mp.Permute.Permute.perm) in
  let* (core : Core.t) =
    fused mp.Permute.Permute.x <|> eager mp.Permute.Permute.x
  in
  let* () = interior core.Core.query in
  let* () = interior core.Core.key in
  let* () = interior core.Core.value in
  let* q, uq, hq, rq = query_or_key core.Core.query in
  let* k, uk, hk, rk = query_or_key core.Core.key in
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
    }

(* The fused form is the node's own decomposition: the output keeps its id,
   [Identical]. An eager form computes the same in exact arithmetic but rounds
   differently ([Eager_mul]: [Equivalent]), or takes the scale's reciprocal
   rounded ([Eager_div]: [Approximate], D13), so the node defines a fresh id
   the consumers are rewired to. *)
let build m _region =
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

let pass =
  Pass.of_pattern ~name:"fuse_attention_layer" ~pattern ~build:{ Pass.build }
