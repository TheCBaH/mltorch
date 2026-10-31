(* A whole encoder attention layer as ONE node: normalization, the query, key
   and value projections, head split, optional rotary embedding, scaled
   dot-product attention, head merge, output projection, layer scale and
   residual.

     Pre:   y = x + gamma * (A(N(x)) W_o^T + b_o)
     Post:  y = N(x + gamma * (A(x) W_o^T + b_o))
     none:  y = x + gamma * (A(x) W_o^T + b_o)
     A(u) = merge_heads (sdpa (R(u W_q^T + b_q), R(u W_k^T + b_k), u W_v^T + b_v, mask))
     R(z) = z * cos + rotate_half(z) * sin, per head, cos/sin shared by heads

   It is the sanctioned exception to "one Native node per ATen op": no importer
   or bridge builds it, only a rewrite that finds the decomposed subgraph, and
   its meaning IS that subgraph -- [Attention_layer_decompose] builds it, and
   [Eval_direct] evaluates it, so the node equals its decomposition by
   construction. Its point is the granted compute path: one node is one
   invocation, one selection site and one arena-invisible set of interiors.

   Axes: [x] is [1,1,B,1,L,E] (sequences on D, tokens on W, features on C,
   Linear's and Sdpa's axes); each weight [E,1,1,1,1,E] (Linear's [Out,..,In]);
   each bias, [layer_scale] and the norm's affine operands [1,1,1,1,1,E];
   [cos]/[sin] [1,1,B|1,1,L,d], d = E / heads; the mask Sdpa's additive f32
   broadcast set against the [1,1,B,heads,L,L] scores. The output is [x]'s. *)

module Attention_layer = struct
  (* Alphabetical. *)
  module Placement = struct
    type t = Post | Pre

    let name = function Post -> "post" | Pre -> "pre"
    let all = [ Post; Pre ]
  end

  module Projection = struct
    type t = { bias : Tensor_ref.t option; weight : Tensor_ref.t }

    let jsont : t Jsont.t =
      Jsont.map ~kind:"attention_layer_projection"
        ~dec:(fun json ->
          let ms = Json_util.req_obj json "projection" in
          {
            bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
            weight =
              Json_util.req_field ms "weight" Tensor_ref.jsont "projection";
          })
        ~enc:(fun t ->
          let ref_ = Json_util.enc Tensor_ref.jsont in
          Json_util.jobj
            ((match t.bias with None -> [] | Some b -> [ ("bias", ref_ b) ])
            @ [ ("weight", ref_ t.weight) ]))
        Jsont.json

    let operands (t : t) = t.weight :: Option.to_list t.bias
    let map f (t : t) = { bias = Option.map f t.bias; weight = f t.weight }
  end

  module Norm = struct
    (* Alphabetical. *)
    module Kind = struct
      type t = Layer | Rms

      let name = function Layer -> "layer" | Rms -> "rms"
      let all = [ Layer; Rms ]
    end

    type t = {
      bias : Tensor_ref.t option;  (** [Layer] only *)
      eps : float;
      kind : Kind.t;
      placement : Placement.t;
      weight : Tensor_ref.t option;
    }

    let of_name ~kind:what all name text =
      match List.find_opt (fun x -> name x = text) all with
      | Some x -> x
      | None ->
          Jsont.Error.msgf Jsont.Meta.none "attention_layer: unknown %s %s" what
            text

    let jsont : t Jsont.t =
      Jsont.map ~kind:"attention_layer_norm"
        ~dec:(fun json ->
          let ms = Json_util.req_obj json "norm" in
          let get k c = Json_util.req_field ms k c "norm" in
          {
            bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
            eps = get "eps" Json_util.f32_jsont;
            kind =
              of_name ~kind:"norm kind" Kind.all Kind.name
                (get "kind" Jsont.string);
            placement =
              of_name ~kind:"placement" Placement.all Placement.name
                (get "placement" Jsont.string);
            weight = Json_util.opt_field ms "weight" Tensor_ref.jsont;
          })
        ~enc:(fun t ->
          let ref_ = Json_util.enc Tensor_ref.jsont in
          let opt k = function None -> [] | Some r -> [ (k, ref_ r) ] in
          Json_util.jobj
            (opt "bias" t.bias
            @ [
                ("eps", Json_util.enc Json_util.f32_jsont t.eps);
                ("kind", Json_util.jstr (Kind.name t.kind));
                ("placement", Json_util.jstr (Placement.name t.placement));
              ]
            @ opt "weight" t.weight))
        Jsont.json

    let operands (t : t) = Option.to_list t.weight @ Option.to_list t.bias

    let map f (t : t) =
      { t with bias = Option.map f t.bias; weight = Option.map f t.weight }
  end

  module Rope = struct
    type t = { cos : Tensor_ref.t; sin : Tensor_ref.t }

    let jsont : t Jsont.t =
      Jsont.Object.map ~kind:"attention_layer_rope" (fun cos sin ->
          { cos; sin })
      |> Jsont.Object.mem "cos" Tensor_ref.jsont ~enc:(fun r -> r.cos)
      |> Jsont.Object.mem "sin" Tensor_ref.jsont ~enc:(fun r -> r.sin)
      |> Jsont.Object.finish
  end

  type params = { heads : Op_config.Pos.t; scale : Attention.Sdpa.Scale.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"attention_layer_params" (fun heads scale ->
        { heads; scale })
    |> Jsont.Object.mem "heads" Op_config.Pos.jsont ~enc:(fun p -> p.heads)
    |> Jsont.Object.mem "scale" Attention.Sdpa.scale_jsont ~enc:(fun p ->
        p.scale)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{heads=%a;@ scale=%a}@]" Op_config.Pos.pp p.heads
      Attention.Sdpa.pp_scale p.scale

  (* Fields alphabetical. *)
  type t = {
    k : Projection.t;
    layer_scale : Tensor_ref.t option;
    mask : Tensor_ref.t option;
    norm : Norm.t option;
    o : Projection.t;
    params : params;
    q : Projection.t;
    rope : Rope.t option;
    v : Projection.t;
    x : Tensor_ref.t;
  }

  let name = "Attention_layer"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          k = get "k" Projection.jsont;
          layer_scale = Json_util.opt_field ms "layer_scale" Tensor_ref.jsont;
          mask = Json_util.opt_field ms "mask" Tensor_ref.jsont;
          norm = Json_util.opt_field ms "norm" Norm.jsont;
          o = get "o" Projection.jsont;
          params = get "params" params_jsont;
          q = get "q" Projection.jsont;
          rope = Json_util.opt_field ms "rope" Rope.jsont;
          v = get "v" Projection.jsont;
          x = get "x" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt k c = function
          | None -> []
          | Some v -> [ (k, Json_util.enc c v) ]
        in
        Json_util.jobj
          ([ ("k", Json_util.enc Projection.jsont t.k) ]
          @ opt "layer_scale" Tensor_ref.jsont t.layer_scale
          @ opt "mask" Tensor_ref.jsont t.mask
          @ opt "norm" Norm.jsont t.norm
          @ [
              ("o", Json_util.enc Projection.jsont t.o);
              ("params", Json_util.enc params_jsont t.params);
              ("q", Json_util.enc Projection.jsont t.q);
            ]
          @ opt "rope" Rope.jsont t.rope
          @ [ ("v", Json_util.enc Projection.jsont t.v); ("x", ref_ t.x) ]))
      Jsont.json

  (* In reading order: the input, the norm's affine operands, each projection,
     the layer scale, the rotary tables, the mask. *)
  let operands (t : t) =
    (t.x :: (match t.norm with None -> [] | Some n -> Norm.operands n))
    @ Projection.operands t.q @ Projection.operands t.k
    @ Projection.operands t.v @ Projection.operands t.o
    @ Option.to_list t.layer_scale
    @ (match t.rope with None -> [] | Some r -> [ r.Rope.cos; r.Rope.sin ])
    @ Option.to_list t.mask

  let map_operands f (t : t) =
    {
      t with
      k = Projection.map f t.k;
      layer_scale = Option.map f t.layer_scale;
      mask = Option.map f t.mask;
      norm = Option.map (Norm.map f) t.norm;
      o = Projection.map f t.o;
      q = Projection.map f t.q;
      rope =
        Option.map
          (fun r -> { Rope.cos = f r.Rope.cos; sin = f r.Rope.sin })
          t.rope;
      v = Projection.map f t.v;
      x = f t.x;
    }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    let opt = Fmt.option ~none:(Fmt.any "none") pp_ref in
    let proj fmt (p : Projection.t) =
      Fmt.pf fmt "@[<h>(%a, %a)@]" pp_ref p.Projection.weight opt
        p.Projection.bias
    in
    Fmt.pf fmt
      "@[<hv 2>attention_layer@ x=%a@ norm=%a@ q=%a@ k=%a@ v=%a@ o=%a@ \
       layer_scale=%a@ rope=%a@ mask=%a@ params=%a@]"
      pp_ref t.x
      (Fmt.option ~none:(Fmt.any "none") (fun fmt (n : Norm.t) ->
           Fmt.pf fmt "@[<h>%s/%s(eps=%a, weight=%a, bias=%a)@]"
             (Norm.Kind.name n.Norm.kind)
             (Placement.name n.Norm.placement)
             Fmt.float n.Norm.eps opt n.Norm.weight opt n.Norm.bias))
      t.norm proj t.q proj t.k proj t.v proj t.o opt t.layer_scale
      (Fmt.option ~none:(Fmt.any "none") (fun fmt (r : Rope.t) ->
           Fmt.pf fmt "@[<h>(%a, %a)@]" pp_ref r.Rope.cos pp_ref r.Rope.sin))
      t.rope opt t.mask pp_params t.params

  (* The head-split shape a query, key or value takes: [1,1,B,heads,L,d]. *)
  let head_shape ~(x_shape : Vec6.shape) ~heads ~d =
    Vec6.shape ~n:1 ~t:1
      ~d:(Dim.to_int (Vec6.get x_shape Axis.D))
      ~h:heads
      ~w:(Dim.to_int (Vec6.get x_shape Axis.W))
      ~c:d

  (* Every operand's shape against the one [x] and [heads] determine. The
     output is [x]'s; the attention's own bounds (the mask's broadcast set,
     the total work) are [Sdpa.output_shape]'s, applied to the head-split
     shapes the decomposition gives it. *)
  let output_shape (t : t) ~(shape_of : Tensor_ref.t -> Vec6.shape) =
    let open Err.Syntax in
    let x_shape = shape_of t.x in
    let b = Dim.to_int (Vec6.get x_shape Axis.D)
    and l = Dim.to_int (Vec6.get x_shape Axis.W)
    and e = Vec6.get x_shape Axis.C in
    let ei = Dim.to_int e in
    let expect role expected actual =
      if
        List.for_all
          (fun a -> Dim.equal (Vec6.get expected a) (Vec6.get actual a))
          Axis.all
      then Err.return ()
      else
        Err.fail
          (`Operand_shape
             {
               Shape_error.Operand_shape.operand = `Attention_layer role;
               expected;
               actual;
             })
    in
    let module R = Shape_error.Attention_layer_role in
    let* () = expect R.X (Vec6.shape ~n:1 ~t:1 ~d:b ~h:1 ~w:l ~c:ei) x_shape in
    let heads = Op_config.Pos.to_int t.params.heads in
    let* () =
      if ei mod heads <> 0 then
        Err.fail
          (`Attention_layer
             (Shape_error.Attention_layer.Heads
                { width = e; heads = t.params.heads }))
      else Err.return ()
    in
    let d = ei / heads in
    let vector = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:ei
    and weight = Vec6.shape ~n:ei ~t:1 ~d:1 ~h:1 ~w:1 ~c:ei in
    let opt role expected = function
      | None -> Err.return ()
      | Some r -> expect role expected (shape_of r)
    in
    let proj ~w ~b (p : Projection.t) =
      let* () = expect w weight (shape_of p.Projection.weight) in
      opt b vector p.Projection.bias
    in
    let* () =
      match t.norm with
      | None -> Err.return ()
      | Some n ->
          let* () = opt R.Norm_weight vector n.Norm.weight in
          opt R.Norm_bias vector n.Norm.bias
    in
    let* () = proj ~w:R.Q_weight ~b:R.Q_bias t.q in
    let* () = proj ~w:R.K_weight ~b:R.K_bias t.k in
    let* () = proj ~w:R.V_weight ~b:R.V_bias t.v in
    let* () = proj ~w:R.O_weight ~b:R.O_bias t.o in
    let* () = opt R.Gamma vector t.layer_scale in
    let* () =
      match t.rope with
      | None -> Err.return ()
      | Some r ->
          let* () =
            if d mod 2 <> 0 then
              Err.fail
                (`Attention_layer
                   (Shape_error.Attention_layer.Rope_odd
                      { head_dim = Dim.extent d }))
            else Err.return ()
          in
          let table role tensor =
            let actual = shape_of tensor in
            let batch = Dim.to_int (Vec6.get actual Axis.D) in
            let batch = if batch = 1 then 1 else b in
            expect role (Vec6.shape ~n:1 ~t:1 ~d:batch ~h:1 ~w:l ~c:d) actual
          in
          let* () = table R.Cos r.Rope.cos in
          table R.Sin r.Rope.sin
    in
    let qkv = head_shape ~x_shape ~heads ~d in
    let* (_ : Vec6.shape) =
      Attention.Sdpa.output_shape ~query_shape:qkv ~key_shape:qkv
        ~value_shape:qkv
        ~mask_shape:(Option.map shape_of t.mask)
    in
    Err.return x_shape
end
