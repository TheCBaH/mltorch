(* An encoder attention layer two ways over the same inputs: as one
   [Attention_layer] node, and as the decomposed subgraph of existing ops an
   importer builds for it, written out here by hand rather than through
   [Attention_layer_decompose] -- so a test that compares the two compares two
   things. Both graphs create their inputs in the same order, so the same ids
   bind the same tensors in either. The decomposition is also the baseline the
   granted routes are measured against. *)

module L = Attention_layer.Attention_layer

module Rope_form = struct
  type t = Batched | No_rope | Shared
end

module Config = struct
  type t = {
    batch : int;
    tokens : int;
    width : int;
    heads : int;
    norm : (L.Norm.Kind.t * L.Placement.t) option;
    norm_weight : bool;
    norm_bias : bool;
    q_bias : bool;
    k_bias : bool;
    v_bias : bool;
    o_bias : bool;
    layer_scale : bool;
    rope : Rope_form.t;
    mask : bool;
    scale : Attention.Sdpa.Scale.t;
  }
end

let small =
  {
    Config.batch = 2;
    tokens = 5;
    width = 8;
    heads = 2;
    norm = None;
    norm_weight = false;
    norm_bias = false;
    q_bias = false;
    k_bias = false;
    v_bias = false;
    o_bias = false;
    layer_scale = false;
    rope = Rope_form.No_rope;
    mask = false;
    scale = Attention.Sdpa.Scale.Default;
  }

let all_flags (c : Config.t) =
  {
    c with
    Config.q_bias = true;
    k_bias = true;
    v_bias = true;
    o_bias = true;
    layer_scale = true;
  }

let with_norm ?(weight = true) ?(bias = true) kind placement (c : Config.t) =
  {
    c with
    Config.norm = Some (kind, placement);
    norm_weight = weight;
    norm_bias = bias && kind = L.Norm.Kind.Layer;
  }

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c

(* Every operand, as an input, in a fixed order. *)
module Operands = struct
  type t = {
    x : Tensor_id.t;
    norm_weight : Tensor_id.t option;
    norm_bias : Tensor_id.t option;
    wq : Tensor_id.t;
    bq : Tensor_id.t option;
    wk : Tensor_id.t;
    bk : Tensor_id.t option;
    wv : Tensor_id.t;
    bv : Tensor_id.t option;
    wo : Tensor_id.t;
    bo : Tensor_id.t option;
    gamma : Tensor_id.t option;
    cos : Tensor_id.t option;
    sin : Tensor_id.t option;
    mask : Tensor_id.t option;
  }
end

let d_of (c : Config.t) = c.Config.width / c.Config.heads

let operands (c : Config.t) =
  let open Graph_builder in
  let e = c.Config.width and b = c.Config.batch and l = c.Config.tokens in
  let d = d_of c in
  let opt present shape =
    if present then
      let+ id = input ~shape () in
      Some id
    else return None
  in
  let vector = s 1 1 1 1 1 e and weight = s e 1 1 1 1 e in
  let* x = input ~shape:(s 1 1 b 1 l e) () in
  let* norm_weight =
    opt (c.Config.norm <> None && c.Config.norm_weight) vector
  in
  let* norm_bias = opt (c.Config.norm <> None && c.Config.norm_bias) vector in
  let* wq = input ~shape:weight () in
  let* bq = opt c.Config.q_bias vector in
  let* wk = input ~shape:weight () in
  let* bk = opt c.Config.k_bias vector in
  let* wv = input ~shape:weight () in
  let* bv = opt c.Config.v_bias vector in
  let* wo = input ~shape:weight () in
  let* bo = opt c.Config.o_bias vector in
  let* gamma = opt c.Config.layer_scale vector in
  let table_batch =
    match c.Config.rope with Rope_form.Batched -> b | _ -> 1
  in
  let rope = c.Config.rope <> Rope_form.No_rope in
  let* cos = opt rope (s 1 1 table_batch 1 l d) in
  let* sin = opt rope (s 1 1 table_batch 1 l d) in
  let+ mask = opt c.Config.mask (s 1 1 b 1 l l) in
  {
    Operands.x;
    norm_weight;
    norm_bias;
    wq;
    bq;
    wk;
    bk;
    wv;
    bv;
    wo;
    bo;
    gamma;
    cos;
    sin;
    mask;
  }

let payload (c : Config.t) (o : Operands.t) : L.t =
  let proj weight bias = { L.Projection.weight; bias } in
  {
    L.k = proj o.Operands.wk o.Operands.bk;
    layer_scale = o.Operands.gamma;
    mask = o.Operands.mask;
    norm =
      Option.map
        (fun (kind, placement) ->
          {
            L.Norm.bias = o.Operands.norm_bias;
            eps = Json_util.f32_to_f32 1e-5;
            kind;
            placement;
            weight = o.Operands.norm_weight;
          })
        c.Config.norm;
    o = proj o.Operands.wo o.Operands.bo;
    params =
      { L.heads = Op_config.Pos.of_int c.Config.heads; scale = c.Config.scale };
    q = proj o.Operands.wq o.Operands.bq;
    rope =
      (match (o.Operands.cos, o.Operands.sin) with
      | Some cos, Some sin -> Some { L.Rope.cos; sin }
      | _ -> None);
    v = proj o.Operands.wv o.Operands.bv;
    x = o.Operands.x;
  }

let build name m =
  Graph_builder.build ~name ~outputs:(fun y -> [ y ]) m
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

let node (c : Config.t) =
  build "attention_layer"
    Graph_builder.(
      let* o = operands c in
      attention_layer (payload c o))

let swap_hw = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, H); (C, C) ]

(* The decomposed layer as an importer spells it, written independently. *)
let decomposed (c : Config.t) =
  build "attention_layer_decomposed"
    Graph_builder.(
      let* o = operands c in
      let b = c.Config.batch and l = c.Config.tokens and e = c.Config.width in
      let h = c.Config.heads and d = d_of c in
      let eps = Json_util.f32_to_f32 1e-5 in
      let norm kind u =
        match kind with
        | L.Norm.Kind.Layer ->
            layer_norm
              { Norm.LayerNorm.dims = [ Axis.C ]; eps }
              ~x:u ?weight:o.Operands.norm_weight ?bias:o.Operands.norm_bias ()
        | L.Norm.Kind.Rms ->
            rms_norm
              { Norm.RmsNorm.dims = [ Axis.C ]; eps }
              ~x:u ?weight:o.Operands.norm_weight ()
      in
      let linear_ ~weight ?bias u =
        linear
          { Linear.Linear.in_features = Dim.extent e }
          ~x:u ~weight ?bias ()
      in
      let heads z =
        let* z = reshape { Reshape.Reshape.shape = s 1 1 b l h d } z in
        permute swap_hw z
      in
      let rope z =
        match (o.Operands.cos, o.Operands.sin) with
        | Some cos, Some sin ->
            let half = d / 2 in
            let slice_ a z' =
              slice
                {
                  Split.Slice.axis = Axis.C;
                  start = Dim.fence a;
                  stop = Dim.fence (a + half);
                  step = Op_config.Pos.of_int 1;
                }
                z'
            in
            let* x1 = slice_ 0 z in
            let* x2 = slice_ half z in
            let* neg = mul_scalar (-1.) x2 in
            let* rotated = concat { Concat.Concat.axis = Axis.C } [ neg; x1 ] in
            let* a = mul z cos in
            let* r = mul rotated sin in
            add a r
        | _ -> return z
      in
      let* u =
        match c.Config.norm with
        | Some (kind, L.Placement.Pre) -> norm kind o.Operands.x
        | _ -> return o.Operands.x
      in
      let* q = linear_ ~weight:o.Operands.wq ?bias:o.Operands.bq u in
      let* q = heads q in
      let* q = rope q in
      let* k = linear_ ~weight:o.Operands.wk ?bias:o.Operands.bk u in
      let* k = heads k in
      let* k = rope k in
      let* v = linear_ ~weight:o.Operands.wv ?bias:o.Operands.bv u in
      let* v = heads v in
      let* a =
        sdpa
          { Attention.Sdpa.scale = c.Config.scale }
          ~query:q ~key:k ~value:v ?mask:o.Operands.mask ()
      in
      let* a = permute swap_hw a in
      let* a = reshape { Reshape.Reshape.shape = s 1 1 b 1 l e } a in
      let* y = linear_ ~weight:o.Operands.wo ?bias:o.Operands.bo a in
      let* y =
        match o.Operands.gamma with Some g -> mul y g | None -> return y
      in
      let* r = add o.Operands.x y in
      match c.Config.norm with
      | Some (kind, L.Placement.Post) -> norm kind r
      | _ -> return r)

(* Deterministic values for every input: small, signed, binary32. A mask input
   holds -inf now and then, never a whole row of it. *)
let inputs ?(salt = 0) (g : Graph_ir.graph) =
  List.mapi
    (fun i id ->
      let sg = Tensor_id.Map.find id g.Graph_ir.Graph.tensors in
      let shape = sg.Tensor_sig.shape in
      let weight = Dim.to_int (Vec6.get shape Axis.N) > 1 in
      ( id,
        Tensor.materialize shape (fun c ->
            let n = (Vec6.offset shape c :> int) in
            let k = ((n * 7919) + (i * 104729) + salt) mod 2003 in
            let v = float_of_int (k - 1001) /. 1001. in
            Json_util.f32_to_f32
              (if weight then
                 v /. sqrt (float_of_int (Dim.to_int (Vec6.get shape Axis.C)))
               else v *. 2.)) ))
    g.Graph_ir.Graph.inputs
