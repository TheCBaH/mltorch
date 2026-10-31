open Graph_ir
module A = Attention_block
module AL = Attention_layer_block
module L = Attention_layer.Attention_layer
module S = Sdpa_select.Reason

module Reason = struct
  type t =
    | Attention of Sdpa_select.Reason.t
    | Not_attention_layer
    | Precision of A.Precision.t

  let pp ppf = function
    | Attention r -> Sdpa_select.Reason.pp ppf r
    | Not_attention_layer -> Fmt.string ppf "not an attention layer"
    | Precision p ->
        Fmt.pf ppf "an attention layer at %s has no body" (A.Precision.name p)
end

let extent shape axis = Dim.to_int (Vec6.get shape axis)

(* Row-major element strides of a dense tensor, 0 on an axis of extent 1. *)
let stride shape axis =
  let rec after = function
    | [] -> []
    | a :: rest -> if a = axis then rest else after rest
  in
  if extent shape axis = 1 then 0L
  else
    Int64.of_int
      (List.fold_left (fun n a -> n * extent shape a) 1 (after Axis.all))

let is_f32 (sg : Tensor_sig.t) =
  match sg.Tensor_sig.fmt with Payload.Fmt Payload.F32 -> true | _ -> false

let choose ~grant (_ : Contraction.Select.Capabilities.t) (g : graph) id =
  let ( let* ) = Result.bind in
  let attention r = Error (Reason.Attention r) in
  let* (grant : Sdpa_select.Grant.t) =
    Option.to_result ~none:(Reason.Attention S.Ungranted) grant
  in
  let precision = grant.Sdpa_select.Grant.precision in
  let* () =
    match precision with
    | A.Precision.Bf16_paired -> Error (Reason.Precision precision)
    | A.Precision.F32_fused | A.Precision.F32_ordered -> Ok ()
  in
  let* node =
    Option.to_result ~none:(Reason.Attention S.Unknown_node)
      (List.find_opt (fun (n : node) -> Node_id.equal n.Node.id id) (nodes g))
  in
  let* l =
    match node.Node.op with
    | Attention_layer l -> Ok l
    | _ -> Error Reason.Not_attention_layer
  in
  let sig_of r = Tensor_id.Map.find r g.Graph.tensors in
  let output =
    match node.Node.outputs with
    | [ o ] -> o
    | _ -> invalid_arg "Layer_select: an attention layer has one output"
  in
  let roles =
    {
      Layer_site.Roles.x = l.L.x;
      norm_weight = Option.bind l.L.norm (fun n -> n.L.Norm.weight);
      norm_bias = Option.bind l.L.norm (fun n -> n.L.Norm.bias);
      wq = l.L.q.L.Projection.weight;
      bq = l.L.q.L.Projection.bias;
      wk = l.L.k.L.Projection.weight;
      bk = l.L.k.L.Projection.bias;
      wv = l.L.v.L.Projection.weight;
      bv = l.L.v.L.Projection.bias;
      wo = l.L.o.L.Projection.weight;
      bo = l.L.o.L.Projection.bias;
      gamma = l.L.layer_scale;
      cos = Option.map (fun r -> r.L.Rope.cos) l.L.rope;
      sin = Option.map (fun r -> r.L.Rope.sin) l.L.rope;
      mask = l.L.mask;
      output;
    }
  in
  let r = roles in
  let named =
    [
      ("input", Some r.Layer_site.Roles.x);
      ("norm weight", r.Layer_site.Roles.norm_weight);
      ("norm bias", r.Layer_site.Roles.norm_bias);
      ("query weight", Some r.Layer_site.Roles.wq);
      ("query bias", r.Layer_site.Roles.bq);
      ("key weight", Some r.Layer_site.Roles.wk);
      ("key bias", r.Layer_site.Roles.bk);
      ("value weight", Some r.Layer_site.Roles.wv);
      ("value bias", r.Layer_site.Roles.bv);
      ("output weight", Some r.Layer_site.Roles.wo);
      ("output bias", r.Layer_site.Roles.bo);
      ("layer scale", r.Layer_site.Roles.gamma);
      ("cos", r.Layer_site.Roles.cos);
      ("sin", r.Layer_site.Roles.sin);
    ]
  in
  let* () =
    List.fold_left
      (fun acc (role, t) ->
        let* () = acc in
        match t with
        | Some t when not (is_f32 (sig_of t)) -> attention (S.Not_f32 role)
        | Some _ | None -> Ok ())
      (Ok ()) named
  in
  let* () =
    match l.L.mask with
    | Some m when not (is_f32 (sig_of m)) -> attention S.Mask_not_f32
    | Some _ | None -> Ok ()
  in
  let xs = (sig_of l.L.x).Tensor_sig.shape in
  let batch = extent xs Axis.D
  and tokens = extent xs Axis.W
  and width = extent xs Axis.C in
  let heads = Op_config.Pos.to_int l.L.params.L.heads in
  let d = width / heads in
  let i64 = Int64.of_int in
  let scale =
    match l.L.params.L.scale with
    | Attention.Sdpa.Scale.Default -> A.Scale.default ~d:(i64 d)
    | Attention.Sdpa.Scale.Explicit x -> A.Scale.explicit x
  in
  let rope =
    Option.map
      (fun (rp : L.Rope.t) ->
        let cs = (sig_of rp.L.Rope.cos).Tensor_sig.shape in
        {
          AL.Rope.batch = stride cs Axis.D;
          row = stride cs Axis.W;
          col = stride cs Axis.C;
        })
      l.L.rope
  in
  let mask =
    Option.map
      (fun m ->
        let ms = (sig_of m).Tensor_sig.shape in
        {
          A.Strides.batch = stride ms Axis.D;
          head = stride ms Axis.H;
          row = stride ms Axis.W;
          col = stride ms Axis.C;
        })
      l.L.mask
  in
  let norm =
    Option.map
      (fun (n : L.Norm.t) ->
        {
          AL.Norm_spec.kind =
            (match n.L.Norm.kind with
            | L.Norm.Kind.Layer -> Norm_block.Kind.Layer
            | L.Norm.Kind.Rms -> Norm_block.Kind.Rms);
          placement =
            (match n.L.Norm.placement with
            | L.Placement.Post -> AL.Placement.Post
            | L.Placement.Pre -> AL.Placement.Pre);
          eps = n.L.Norm.eps;
          weight = Option.is_some n.L.Norm.weight;
          bias = Option.is_some n.L.Norm.bias;
        })
      l.L.norm
  in
  let flags =
    {
      AL.Flags.q_bias = Option.is_some r.Layer_site.Roles.bq;
      k_bias = Option.is_some r.Layer_site.Roles.bk;
      v_bias = Option.is_some r.Layer_site.Roles.bv;
      o_bias = Option.is_some r.Layer_site.Roles.bo;
      layer_scale = Option.is_some r.Layer_site.Roles.gamma;
    }
  in
  let bk = grant.Sdpa_select.Grant.block_keys in
  let* () =
    if bk > 0 && bk mod 4 = 0 then Ok () else attention (S.Block_keys bk)
  in
  let* block =
    Result.map_error
      (fun e -> Reason.Attention (S.Descriptor (Fmt.str "%a" AL.pp_error e)))
      (Err.payload
         (AL.make ~batch:(i64 batch) ~tokens:(i64 tokens) ~width:(i64 width)
            ~heads:(i64 heads) ?norm ~flags ?rope ?mask ~scale
            ~block_keys:(i64 bk) ~precision ()))
  in
  Ok { Layer_site.block; roles }
