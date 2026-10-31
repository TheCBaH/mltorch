(* The attention-layer rewrite is [decompose]'s inverse: on each decomposition
   it leaves exactly one [Attention_layer] node, equal to the subgraph it
   replaced bit for bit, and it leaves alone a subgraph the node cannot stand
   for. *)

module F = Attention_layer_fixtures
module L = Attention_layer.Attention_layer

let rewritten g =
  match Rewrite.origin g with
  | Error e -> Error (Fmt.str "%a" Rewrite.pp_error (Err.Error.kind e))
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state [ Fuse_attention_layer.pass ] with
      | Error e -> Error (Fmt.str "%a" Pass.pp_error (Err.Error.kind e))
      | Ok (Rewrite.Step (final, _)) -> Ok (Rewrite.graph final))

let layers g =
  List.length
    (List.filter
       (fun (n : Graph_ir.node) ->
         match n.Graph_ir.Node.op with
         | Graph_ir.Attention_layer _ -> true
         | _ -> false)
       g.Graph_ir.Graph.nodes)

let output g =
  let env =
    Eval_direct.run g ~inputs:(F.inputs g)
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  let t = Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs) env in
  let out = ref [] in
  Vec6.iter (Tensor.packed_shape t) (fun c -> out := Tensor.read t c :: !out);
  List.rev !out

let configs =
  let base = F.small in
  [
    ("bare", base);
    ( "pre layer, all flags, masked",
      F.with_norm L.Norm.Kind.Layer L.Placement.Pre
        { (F.all_flags base) with F.Config.mask = true } );
    ( "post layer, all flags",
      F.with_norm L.Norm.Kind.Layer L.Placement.Post (F.all_flags base) );
    ( "pre rms, rope",
      F.with_norm L.Norm.Kind.Rms L.Placement.Pre
        { base with F.Config.rope = F.Rope_form.Shared; q_bias = true } );
    ( "post rms w-less, batched rope, masked",
      F.with_norm ~weight:false L.Norm.Kind.Rms L.Placement.Post
        {
          (F.all_flags base) with
          F.Config.rope = F.Rope_form.Batched;
          mask = true;
        } );
    ( "explicit scale",
      { base with F.Config.scale = Attention.Sdpa.Scale.Explicit 0.125 } );
  ]

let%expect_test "the rewrite inverts the decomposition, bit for bit" =
  List.iter
    (fun (name, c) ->
      let dec = F.decomposed c in
      match rewritten dec with
      | Error e -> Fmt.pr "%s: %s@." name e
      | Ok g ->
          let node = F.node c in
          Fmt.pr
            "%s: %d nodes -> %d (%d layer); equal to the decomposition: %b; to \
             the node built directly: %b@."
            name
            (List.length dec.Graph_ir.Graph.nodes)
            (List.length g.Graph_ir.Graph.nodes)
            (layers g)
            (List.for_all2 Core.Float_bits.equal_portable (output dec)
               (output g))
            (List.for_all2 Core.Float_bits.equal_portable (output node)
               (output g)))
    configs;
  [%expect
    {|
    bare: 14 nodes -> 1 (1 layer); equal to the decomposition: true; to the node built directly: true
    pre layer, all flags, masked: 16 nodes -> 1 (1 layer); equal to the decomposition: true; to the node built directly: true
    post layer, all flags: 16 nodes -> 1 (1 layer); equal to the decomposition: true; to the node built directly: true
    pre rms, rope: 29 nodes -> 1 (1 layer); equal to the decomposition: true; to the node built directly: true
    post rms w-less, batched rope, masked: 30 nodes -> 1 (1 layer); equal to the decomposition: true; to the node built directly: true
    explicit scale: 14 nodes -> 1 (1 layer); equal to the decomposition: true; to the node built directly: true |}]

(* An interior read twice, a bfloat16 weight nowhere, a rotation over part of
   the head: each subgraph stays as it was. *)
let%expect_test "what the rewrite leaves alone" =
  let show name g =
    match rewritten g with
    | Error e -> Fmt.pr "%s: %s@." name e
    | Ok g' ->
        Fmt.pr "%s: %d layer(s), %d nodes kept of %d@." name (layers g')
          (List.length g'.Graph_ir.Graph.nodes)
          (List.length g.Graph_ir.Graph.nodes)
  in
  (* the attention output read a second time: an extra graph output *)
  let c = F.all_flags F.small in
  let dec = F.decomposed c in
  let sdpa_out =
    List.find_map
      (fun (n : Graph_ir.node) ->
        match n.Graph_ir.Node.op with
        | Graph_ir.Sdpa _ -> Some (List.hd n.Graph_ir.Node.outputs)
        | _ -> None)
      dec.Graph_ir.Graph.nodes
    |> Option.get
  in
  show "an interior with a second consumer"
    {
      dec with
      Graph_ir.Graph.outputs = dec.Graph_ir.Graph.outputs @ [ sdpa_out ];
    };
  (* a norm over W and C is not the layer's *)
  show "a norm over two axes"
    (F.build "wide_norm"
       Graph_builder.(
         let* x = input ~shape:(F.s 1 1 2 1 5 8) () in
         layer_norm
           { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 1e-5 }
           ~x ()));
  show "the decomposition itself, for comparison" dec;
  [%expect
    {|
    an interior with a second consumer: 0 layer(s), 15 nodes kept of 15
    a norm over two axes: 0 layer(s), 1 nodes kept of 1
    the decomposition itself, for comparison: 1 layer(s), 1 nodes kept of 15 |}]
