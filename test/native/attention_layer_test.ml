(* The attention layer node: built, printed, serialized, refused on bad
   operands, evaluated equal to its decomposition written out by hand, and
   bundled only under a grant. *)

module F = Attention_layer_fixtures
module L = Attention_layer.Attention_layer

let run g =
  Eval_direct.run g ~inputs:(F.inputs g)
  |> Err.or_raise ~pp_error:Eval_direct.pp_error

let output g env = Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs) env

let cells t =
  let shape = Tensor.packed_shape t in
  let out = ref [] in
  Vec6.iter shape (fun c -> out := Tensor.read t c :: !out);
  Array.of_list (List.rev !out)

let bitwise a b =
  Array.length a = Array.length b
  && Array.for_all2 Core.Float_bits.equal_portable a b

let full =
  F.with_norm L.Norm.Kind.Layer L.Placement.Pre
    {
      (F.all_flags F.small) with
      F.Config.rope = F.Rope_form.Shared;
      mask = true;
    }

let%expect_test "the node prints and round-trips through JSON" =
  let g = F.node full in
  let node = List.hd g.Graph_ir.Graph.nodes in
  Fmt.pr "%a@." (Graph_ir.pp_op g) node.Graph_ir.Node.op;
  let json =
    Graph_json.encode_graph g |> Err.or_raise ~pp_error:Graph_json.pp_error
  in
  let back =
    Graph_json.decode_graph json |> Err.or_raise ~pp_error:Graph_json.pp_error
  in
  let again =
    Graph_json.encode_graph back |> Err.or_raise ~pp_error:Graph_json.pp_error
  in
  Fmt.pr "round trip: %b@." (String.equal json again);
  [%expect
    {|
    attention_layer
      x=t0
      norm=layer/pre(eps=1e-05, weight=t1, bias=t2)
      q=(t3, t4)
      k=(t5, t6)
      v=(t7, t8)
      o=(t9, t10)
      layer_scale=t11
      rope=(t12, t13)
      mask=t14
      params={heads=2; scale=default}
    round trip: true |}]

let refusal name m =
  match Graph_builder.build ~name ~outputs:(fun y -> [ y ]) m with
  | Ok _ -> Fmt.pr "%s: built@." name
  | Error e -> Fmt.pr "%s: %a@." name Graph_builder.pp_error (Err.Error.kind e)

let%expect_test "operand shapes the node refuses" =
  let open Graph_builder in
  let with_ c f =
    let* o = F.operands c in
    attention_layer (f (F.payload c o))
  in
  (* a weight of the wrong orientation's extent *)
  refusal "wide output weight"
    (let* o = F.operands F.small in
     let* w = input ~shape:(F.s 9 1 1 1 1 8) () in
     attention_layer
       {
         (F.payload F.small o) with
         L.o = { L.Projection.weight = w; bias = None };
       });
  refusal "three heads of eight"
    (with_ F.small (fun t ->
         {
           t with
           L.params = { t.L.params with L.heads = Op_config.Pos.of_int 3 };
         }));
  refusal "rope over odd heads"
    (with_
       { F.small with F.Config.width = 6; rope = F.Rope_form.Shared }
       Fun.id);
  refusal "a cos table of three sequences"
    (let c = { F.small with F.Config.rope = F.Rope_form.Shared } in
     let* o = F.operands c in
     let* cos = input ~shape:(F.s 1 1 3 1 5 4) () in
     attention_layer
       { (F.payload c o) with L.rope = Some { L.Rope.cos; sin = cos } });
  refusal "a mask over seven keys"
    (let* o = F.operands F.small in
     let* m = input ~shape:(F.s 1 1 2 1 5 7) () in
     attention_layer { (F.payload F.small o) with L.mask = Some m });
  refusal "an input with heads on H"
    (let* o = F.operands F.small in
     let* x = input ~shape:(F.s 1 1 2 2 5 8) () in
     attention_layer { (F.payload F.small o) with L.x });
  [%expect
    {|
    wide output weight: attention_layer output weight shape must be [N=8 T=1 D=1
                                                                     H=1 W=1 C=8], got
    [N=9 T=1 D=1 H=1 W=1 C=8]
    three heads of eight: attention_layer: width 8 is not a multiple of 3 heads
    rope over odd heads: attention_layer: a rotary embedding needs an even head dimension, got 3
    a cos table of three sequences: attention_layer cos shape must be [D=2 H=1
                                                                       W=5 C=4], got
    [D=3 H=1 W=5 C=4]
    a mask over seven keys: sdpa: mask extent on C must be 5 (the score extent) or 1, got 7
    an input with heads on H: attention_layer input shape must be [D=2 H=1 W=5
                                                                   C=8], got
    [D=2 H=2 W=5 C=8] |}]

let configs =
  let base = F.small in
  [
    ("bare", base);
    ("all flags", F.all_flags base);
    ( "pre layer, all flags, masked",
      F.with_norm L.Norm.Kind.Layer L.Placement.Pre
        { (F.all_flags base) with F.Config.mask = true } );
    ( "post layer, all flags",
      F.with_norm L.Norm.Kind.Layer L.Placement.Post (F.all_flags base) );
    ( "pre rms, shared rope",
      F.with_norm L.Norm.Kind.Rms L.Placement.Pre
        { base with F.Config.rope = F.Rope_form.Shared; q_bias = true } );
    ( "post rms w-less, batched rope, masked",
      F.with_norm ~weight:false L.Norm.Kind.Rms L.Placement.Post
        {
          (F.all_flags base) with
          F.Config.rope = F.Rope_form.Batched;
          mask = true;
        } );
    ( "pre layer without affine, explicit scale",
      F.with_norm ~weight:false ~bias:false L.Norm.Kind.Layer L.Placement.Pre
        { base with F.Config.scale = Attention.Sdpa.Scale.Explicit 0.125 } );
    ( "one head of 12, 7 tokens, all flags, rope",
      {
        (F.all_flags base) with
        F.Config.width = 12;
        heads = 1;
        tokens = 7;
        rope = F.Rope_form.Shared;
      } );
  ]

let%expect_test "the node equals its decomposition, bit for bit" =
  List.iter
    (fun (name, c) ->
      let node = F.node c and dec = F.decomposed c in
      let a = cells (output node (run node))
      and b = cells (output dec (run dec)) in
      let finite =
        Array.fold_left (fun n x -> if Float.is_finite x then n + 1 else n) 0 a
      in
      Fmt.pr "%s: %s (%d nodes decomposed, %d of %d finite)@." name
        (if bitwise a b then "bitwise" else "DIFFERS")
        (List.length dec.Graph_ir.Graph.nodes)
        finite (Array.length a))
    configs;
  [%expect
    {|
    bare: bitwise (14 nodes decomposed, 80 of 80 finite)
    all flags: bitwise (15 nodes decomposed, 80 of 80 finite)
    pre layer, all flags, masked: bitwise (16 nodes decomposed, 80 of 80 finite)
    post layer, all flags: bitwise (16 nodes decomposed, 80 of 80 finite)
    pre rms, shared rope: bitwise (29 nodes decomposed, 80 of 80 finite)
    post rms w-less, batched rope, masked: bitwise (30 nodes decomposed, 80 of 80 finite)
    pre layer without affine, explicit scale: bitwise (15 nodes decomposed, 80 of 80 finite)
    one head of 12, 7 tokens, all flags, rope: bitwise (29 nodes decomposed, 168 of 168 finite) |}]
