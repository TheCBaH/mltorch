(* Fold an inference batch norm into the convolution feeding it.
   See .ai/native_transform_design.md §12c.

   With s = gamma / sqrt(var + eps), batch norm is y = (z - mean) * s + beta on
   z = conv(x, W) + b, so

     y = conv(x, W * s) + (b - mean) * s + beta

   and the whole normalisation disappears into the conv's parameters. The
   arithmetic is emitted as ordinary graph nodes; [Fold_const] then collapses it
   to two constants. Nothing here computes on a payload except the scalar
   literals below, which is what keeps the two passes independent.

   Layouts decide the shape of the rewrite. beta, mean, var and the conv bias all
   live on C, so the new bias is plain pointwise; the weight is
   [Cout,1,1,Kh,Kw,Cin] with Cout on N, so the scale has to be permuted C -> N
   before it multiplies the weight. *)

open Graph_ir

(* The convolutions this can fold into, reduced to what the rewrite needs. Both
   [Conv2d] and a forward [Convolution] carry Cout on the weight's N — the latter
   delegates its shape and compute straight to the former — so one rewrite serves
   both and [rebuild] is all that distinguishes them.

   A TRANSPOSED convolution is not one of them: [Conv.Convolution.bias_shape]
   takes its channel count from the weight's C rather than its N, so the permuted
   scale would land on the wrong axis and broadcast against the wrong extent. *)
type conv = {
  bias : Tensor_ref.t option;
  rebuild : weight:Tensor_ref.t -> bias:Tensor_ref.t -> op;
  weight : Tensor_ref.t;
  x : Tensor_ref.t;
}

let as_conv = function
  | Conv2d { Conv.Conv2d.params; x; weight; bias } ->
      Some
        {
          bias;
          weight;
          x;
          rebuild =
            (fun ~weight ~bias ->
              Conv2d { Conv.Conv2d.params; x; weight; bias = Some bias });
        }
  | Convolution { Conv.Convolution.params; x; weight; bias }
    when not params.transposed ->
      Some
        {
          bias;
          weight;
          x;
          rebuild =
            (fun ~weight ~bias ->
              Convolution
                { Conv.Convolution.params; x; weight; bias = Some bias });
        }
  | _ -> None

type match_ = {
  bn : Norm.BatchNorm.t;
  bn_node : Node_id.t;
  conv : conv;
  conv_node : Node_id.t;
  out : Tensor_id.t; (* the batch norm's output, substituted away *)
  param : Tensor_sig.t; (* running_mean's: the shape every C vector has *)
  weight : Tensor_sig.t;
  y : Tensor_sig.t;
}

(* [out N] takes [in C] and vice versa, so a [1,…,Cout] channel vector becomes
   the [Cout,1,…] the weight broadcasts against. *)
let c_to_n =
  Permute.Permute.of_fn (function
    | Axis.C -> Axis.N
    | Axis.N -> Axis.C
    | a -> a)

let swap_nc shape =
  let n = Vec6.get shape Axis.N and c = Vec6.get shape Axis.C in
  Vec6.set (Vec6.set shape Axis.N c) Axis.C n

let scalar = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1

(* ---- matching ------------------------------------------------------------- *)

let all_constant ids =
  let open Pattern in
  List.fold_left
    (fun acc id ->
      let* () = acc in
      constant id)
    (return ()) ids

let pattern anchor =
  let open Pattern in
  let* (bn : Norm.BatchNorm.t), (bn_node : node) =
    def anchor (function Batch_norm b -> Some b | _ -> None)
  in
  (* The rewrite moves the scale onto the weight's N axis, which is only the
     channel axis when the norm is over C. *)
  let* () = guard (Axis.equal bn.params.channel Axis.C) in
  (* The conv output must not escape: folding removes the node that computes
     it, and any other consumer still wants the unnormalised value. *)
  let* () = interior bn.x in
  let* (conv : conv), (conv_node : node) = def bn.x as_conv in
  (* Every parameter has to be constant or this pessimises rather than
     optimises: the weight multiply would run on every inference, over the whole
     weight tensor, where before it was a per-channel scale of the output. *)
  let* () =
    all_constant
      ([ conv.weight; bn.running_mean; bn.running_var ]
      @ List.filter_map Fun.id [ conv.bias; bn.weight; bn.bias ])
  in
  let* (param : Tensor_sig.t) = sig_of bn.running_mean in
  let* (weight : Tensor_sig.t) = sig_of conv.weight in
  let* (y : Tensor_sig.t) = sig_of anchor in
  (* Cout on the weight's N must be the channel count the statistics describe,
     or the permuted scale would broadcast against the wrong axis extent. *)
  let+ () =
    guard
      (Dim.equal (Vec6.get weight.shape Axis.N) (Vec6.get param.shape Axis.C))
  in
  {
    bn;
    bn_node = bn_node.Node.id;
    conv;
    conv_node = conv_node.Node.id;
    out = anchor;
    param;
    weight;
    y;
  }

(* ---- building ------------------------------------------------------------- *)

(* An absent optional parameter becomes an explicit constant holding its
   identity value, so the arithmetic below is ONE path for all eight
   present/absent combinations rather than eight variants of it. They are scalar
   because they only ever broadcast, and [Fold_const] removes them again — the
   eps literal is unavoidable anyway, eps being a float parameter rather than an
   edge, so the identities cost nothing extra. *)
let literal value = Tensor.materialize scalar (fun _ -> value)

let supplied ref_ identity =
  let open Recipe in
  match ref_ with
  | Some id -> return (id, [])
  | None ->
      let+ id = fresh scalar in
      (id, [ (id, literal identity) ])

let build m _region =
  let open Recipe in
  let fmt = m.param.Tensor_sig.fmt in
  let pshape = m.param.Tensor_sig.shape in
  let* eps = fresh scalar in
  let* gamma, gamma_c = supplied m.bn.weight 1. in
  let* beta, beta_c = supplied m.bn.bias 0. in
  let* bias, bias_c = supplied m.conv.bias 0. in
  let* shifted = fresh ~fmt pshape in
  let* denom = fresh ~fmt pshape in
  let* scale = fresh ~fmt pshape in
  let* scale_n = fresh ~fmt (swap_nc pshape) in
  let* weight' = fresh ~fmt:m.weight.Tensor_sig.fmt m.weight.Tensor_sig.shape in
  let* centred = fresh ~fmt pshape in
  let* scaled = fresh ~fmt pshape in
  let* bias' = fresh ~fmt pshape in
  (* The folded conv is NOT the same tensor: equal in exact arithmetic, but it
     rounds differently, and §4 lets an id be kept only for the very same
     tensor. So it gets a fresh id and the recipe claims [Equivalent], which
     propagation then carries to everything downstream. *)
  let* y' = fresh ~fmt:m.y.Tensor_sig.fmt m.y.Tensor_sig.shape in
  let node op outputs from = { Recipe.op; outputs; from } in
  let from_bn = [ m.bn_node ] in
  emit
    {
      empty_replacement with
      remove = Node_id.Set.of_list [ m.conv_node; m.bn_node ];
      insert =
        [
          node (Add { a = m.bn.running_var; b = eps }) [ shifted ] from_bn;
          node (Sqrt { x = shifted }) [ denom ] from_bn;
          node (Div { a = gamma; b = denom }) [ scale ] from_bn;
          node (Permute { perm = c_to_n; x = scale }) [ scale_n ] from_bn;
          node (Mul { a = m.conv.weight; b = scale_n }) [ weight' ] from_bn;
          node (Sub { a = bias; b = m.bn.running_mean }) [ centred ] from_bn;
          node (Mul { a = centred; b = scale }) [ scaled ] from_bn;
          node (Add { a = scaled; b = beta }) [ bias' ] from_bn;
          node
            (m.conv.rebuild ~weight:weight' ~bias:bias')
            [ y' ] [ m.conv_node; m.bn_node ];
        ];
      subst = Tensor_id.Map.singleton m.out y';
      value_claims = [ (m.out, y', Correspondence.Equivalent) ];
      constants = ((eps, literal m.bn.params.eps) :: gamma_c) @ beta_c @ bias_c;
    }

let pass = Pass.of_pattern ~name:"fold_batch_norm" ~pattern ~build
