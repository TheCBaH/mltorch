(* Non-default operation parameters surviving a graph-level round trip
   (Group 2/3/7 and op8's sdpa): conv2d, conv2d_padding, max_pool2d,
   layer_norm/rms_norm's optional operands, sdpa, linear, sub, reshape, pad,
   slice, select, and stack. Split from graph_json_test.ml. *)

open Graph_ir
open Graph_json_fixtures

(* ---- Group-2 non-default parameters through the codecs ------------------ *)

(* Every op below already has a [params_jsont], so no new codec is expected --
   what is unproven is that a NON-DEFAULT value survives one. A round trip over
   default parameters cannot tell a codec that carries a field from one that
   drops it and re-derives the default, and defaults are exactly what the
   existing blocks above use: unit stride, zero pad, unit dilation, one group.

   So each configuration here is asymmetric in H/W and non-unit wherever it can
   be. The decoded graph is printed rather than compared, because a comparison
   would pass on two identically-wrong values. *)
let round_trip name build =
  let result =
    let open Err.Syntax in
    let* g = lift_build build in
    let* json = encode_graph g in
    let+ g2 = decode_graph json in
    g2
  in
  Format.printf "%s: %a@." name (pp_result Graph_ir.pp) result

let%expect_test "Group 2: non-default conv2d params survive a round trip" =
  round_trip "conv2d"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 8 8 4) ~name:"x" () in
      let* w = input ~shape:(s 8 1 1 3 2 2) ~name:"w" () in
      let* b = input ~shape:(s1c 8) ~name:"b" () in
      conv2d ~name:"y"
        Conv.Conv2d.
          {
            h =
              {
                kernel = Dim.extent 3;
                stride = Op_config.Pos.of_int 2;
                pad_before = Op_config.Nonneg.of_int 1;
                pad_after = Op_config.Nonneg.of_int 1;
                dilation = Op_config.Pos.of_int 1;
              };
            w =
              {
                kernel = Dim.extent 2;
                stride = Op_config.Pos.of_int 1;
                pad_before = Op_config.Nonneg.of_int 0;
                pad_after = Op_config.Nonneg.of_int 0;
                dilation = Op_config.Pos.of_int 2;
              };
            in_channels = Dim.extent 4;
            groups = Op_config.Pos.of_int 2;
          }
        ~x ~weight:w ~bias:b ());
  [%expect
    {|
    conv2d: graph
            inputs:
              [t0 f32 [H=8 W=8 C=4] ->[n0],
               t1 f32 [N=8 T=1 D=1 H=3 W=2 C=2] ->[n0], t2 f32 [C=8] ->[n0]]
            nodes:
              n0: [t3 f32 [H=4 W=6 C=8]] =
                conv2d
                  x=t0
                  weight=t1
                  bias=t2
                  params={h={kernel=3;
                            stride=2;
                            pad_before=1;
                            pad_after=1;
                            dilation=1};
                         w={kernel=2;
                           stride=1;
                           pad_before=0;
                           pad_after=0;
                           dilation=2};
                         in_channels=4;
                         groups=2}
            outputs: [t3 f32 [H=4 W=6 C=8] <-n0] |}]

let%expect_test "Group 2: conv2d_padding carries the MODE, unresolved" =
  round_trip "conv2d_padding same"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 8 8 4) ~name:"x" () in
      (* per-group input extent 2 x 2 groups = the activation's 4 channels *)
      let* w = input ~shape:(s 8 1 1 3 3 2) ~name:"w" () in
      conv2d_padding ~name:"y"
        Conv.Conv2d_padding.
          {
            stride =
              Op_config.Hw.
                { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
            padding = Conv.Conv2d_padding.Same;
            dilation =
              Op_config.Hw.
                { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 3 };
            groups = Op_config.Pos.of_int 2;
          }
        ~x ~weight:w ());
  [%expect
    {|
    conv2d_padding same: graph
                         inputs:
                           [t0 f32 [H=8 W=8 C=4] ->[n0],
                            t1 f32 [N=8 T=1 D=1 H=3 W=3 C=2] ->[n0]]
                         nodes:
                           n0: [t2 f32 [H=8 W=8 C=8]] =
                             conv2d_padding
                               x=t0
                               weight=t1
                               bias=none
                               params={stride={h=1; w=1};
                                      padding=same;
                                      dilation={h=2; w=3};
                                      groups=2}
                         outputs: [t2 f32 [H=8 W=8 C=8] <-n0] |}]

let%expect_test "Group 2: rectangular max_pool2d params survive a round trip" =
  round_trip "max_pool2d"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 9 7 3) ~name:"x" () in
      max_pool2d ~name:"y"
        Pool.MaxPool2d.
          {
            (* [true], not the default [false]: proves the codec actually
               carries the field rather than always taking [dec_absent]'s
               default. *)
            ceil_mode = true;
            kernel = Op_config.Hw.{ h = Dim.extent 3; w = Dim.extent 2 };
            stride =
              Op_config.Hw.
                { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 1 };
            pad =
              Op_config.Hw.
                { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 0 };
          }
        x);
  [%expect
    {|
    max_pool2d: graph
                inputs: [t0 f32 [H=9 W=7 C=3] ->[n0]]
                nodes:
                  n0: [t1 f32 [H=5 W=6 C=3]] =
                    max_pool2d
                      x=t0
                      params={kernel={h=3; w=2};
                             stride={h=2; w=1};
                             pad={h=1; w=0};
                             ceil_mode=true}
                outputs: [t1 f32 [H=5 W=6 C=3] <-n0] |}]

(* layer_norm's payload is the first with TWO independently optional operands,
   which is a JSON shape no other op has: the codec has to distinguish four
   states, not two, and "weight present, bias absent" is the one an encoder that
   pairs them would silently get wrong. All four are round-tripped.

   [eps] is 0.1 rather than either corpus value on purpose: 0.1 is not f32-exact,
   so a codec that failed to narrow through [Json_util.f32_jsont] would print a
   different number here, which printing alone cannot show. *)
let%expect_test
    "Group 7: layer_norm's two optional operands survive a round trip" =
  let g ?weight ?bias () =
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 8 2 5) ~name:"x" () in
      let* w =
        match weight with
        | None -> return None
        | Some () ->
            let* w = input ~shape:(s 1 1 1 1 2 5) ~name:"w" () in
            return (Some w)
      in
      let* b =
        match bias with
        | None -> return None
        | Some () ->
            let* b = input ~shape:(s 1 1 1 1 2 5) ~name:"b" () in
            return (Some b)
      in
      layer_norm ~name:"y"
        { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 0.1 }
        ~x ?weight:w ?bias:b ())
  in
  round_trip "both" (g ~weight:() ~bias:() ());
  round_trip "weight only" (g ~weight:() ());
  round_trip "bias only" (g ~bias:() ());
  round_trip "neither" (g ());
  [%expect
    {|
    both: graph
          inputs:
            [t0 f32 [H=8 W=2 C=5] ->[n0], t1 f32 [W=2 C=5] ->[n0],
             t2 f32 [W=2 C=5] ->[n0]]
          nodes:
            n0: [t3 f32 [H=8 W=2 C=5]] =
              layer_norm x=t0 weight=t1 bias=t2 params={dims=[W, C]; eps=0.1}
          outputs: [t3 f32 [H=8 W=2 C=5] <-n0]
    weight only: graph
                 inputs: [t0 f32 [H=8 W=2 C=5] ->[n0], t1 f32 [W=2 C=5] ->[n0]]
                 nodes:
                   n0: [t2 f32 [H=8 W=2 C=5]] =
                     layer_norm
                       x=t0
                       weight=t1
                       bias=none
                       params={dims=[W, C]; eps=0.1}
                 outputs: [t2 f32 [H=8 W=2 C=5] <-n0]
    bias only: graph
               inputs: [t0 f32 [H=8 W=2 C=5] ->[n0], t1 f32 [W=2 C=5] ->[n0]]
               nodes:
                 n0: [t2 f32 [H=8 W=2 C=5]] =
                   layer_norm
                     x=t0
                     weight=none
                     bias=t1
                     params={dims=[W, C]; eps=0.1}
               outputs: [t2 f32 [H=8 W=2 C=5] <-n0]
    neither: graph
             inputs: [t0 f32 [H=8 W=2 C=5] ->[n0]]
             nodes:
               n0: [t1 f32 [H=8 W=2 C=5]] =
                 layer_norm
                   x=t0
                   weight=none
                   bias=none
                   params={dims=[W, C]; eps=0.1}
             outputs: [t1 f32 [H=8 W=2 C=5] <-n0] |}]

(* A multi-axis [dims] and an OPTIONAL weight, in both states. The absent case is
   the one the bridge used to make unreachable by materializing a ones tensor,
   so a codec that could not express it would have gone unnoticed. *)
let%expect_test
    "Group 2: rms_norm dims and optional weight survive a round trip" =
  round_trip "rms_norm with weight"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 8 2 5) ~name:"x" () in
      let* w = input ~shape:(s 1 1 1 1 2 5) ~name:"w" () in
      rms_norm ~name:"y"
        { Norm.RmsNorm.dims = [ Axis.W; Axis.C ]; eps = 1e-5 }
        ~x ~weight:w ());
  round_trip "rms_norm no weight"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 8 2 5) ~name:"x" () in
      rms_norm ~name:"y"
        { Norm.RmsNorm.dims = [ Axis.W; Axis.C ]; eps = 1e-5 }
        ~x ());
  [%expect
    {|
    rms_norm with weight: graph
                          inputs:
                            [t0 f32 [H=8 W=2 C=5] ->[n0],
                             t1 f32 [W=2 C=5] ->[n0]]
                          nodes:
                            n0: [t2 f32 [H=8 W=2 C=5]] =
                              rms_norm
                                x=t0
                                weight=t1
                                params={dims=[W, C]; eps=1e-05}
                          outputs: [t2 f32 [H=8 W=2 C=5] <-n0]
    rms_norm no weight: graph
                        inputs: [t0 f32 [H=8 W=2 C=5] ->[n0]]
                        nodes:
                          n0: [t1 f32 [H=8 W=2 C=5]] =
                            rms_norm
                              x=t0
                              weight=none
                              params={dims=[W, C]; eps=1e-05}
                        outputs: [t1 f32 [H=8 W=2 C=5] <-n0] |}]

(* Mask present/absent (an independently-optional operand, [rms_norm]'s
   precedent) AND [Scale.Default] vs [Scale.Explicit] (a bare-number JSON
   shape none of the other ops have): [0.1] is not f32-exact, so a narrowed
   scalar is distinguishable from an unnarrowed one by printing alone. *)
let%expect_test
    "op8: sdpa mask presence and Default/Explicit scale survive a round trip" =
  round_trip "sdpa, mask present, explicit scale"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* q = input ~shape:(s 1 1 2 3 4 5) ~name:"q" () in
      let* k = input ~shape:(s 1 1 2 3 6 5) ~name:"k" () in
      let* v = input ~shape:(s 1 1 2 3 6 5) ~name:"v" () in
      let* m = input ~shape:(s 1 1 2 3 4 6) ~name:"m" () in
      sdpa ~name:"y"
        { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 0.1 }
        ~query:q ~key:k ~value:v ~mask:m ());
  round_trip "sdpa, mask absent, default scale"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* q = input ~shape:(s 1 1 2 3 4 5) ~name:"q" () in
      let* k = input ~shape:(s 1 1 2 3 6 5) ~name:"k" () in
      let* v = input ~shape:(s 1 1 2 3 6 5) ~name:"v" () in
      sdpa ~name:"y"
        { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
        ~query:q ~key:k ~value:v ());
  [%expect
    {|
    sdpa, mask present, explicit scale: graph
                                        inputs:
                                          [t0 f32 [D=2 H=3 W=4 C=5] ->[n0],
                                           t1 f32 [D=2 H=3 W=6 C=5] ->[n0],
                                           t2 f32 [D=2 H=3 W=6 C=5] ->[n0],
                                           t3 f32 [D=2 H=3 W=4 C=6] ->[n0]]
                                        nodes:
                                          n0: [t4 f32 [D=2 H=3 W=4 C=5]] =
                                            sdpa
                                              query=t0
                                              key=t1
                                              value=t2
                                              mask=t3
                                              params={scale=explicit(0.1)}
                                        outputs: [t4 f32 [D=2 H=3 W=4 C=5] <-n0]
    sdpa, mask absent, default scale: graph
                                      inputs:
                                        [t0 f32 [D=2 H=3 W=4 C=5] ->[n0],
                                         t1 f32 [D=2 H=3 W=6 C=5] ->[n0],
                                         t2 f32 [D=2 H=3 W=6 C=5] ->[n0]]
                                      nodes:
                                        n0: [t3 f32 [D=2 H=3 W=4 C=5]] =
                                          sdpa
                                            query=t0
                                            key=t1
                                            value=t2
                                            mask=none
                                            params={scale=default}
                                      outputs: [t3 f32 [D=2 H=3 W=4 C=5] <-n0] |}]

let%expect_test "Group 2: linear with a non-square weight survives a round trip"
    =
  round_trip "linear"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 8 2 5) ~name:"x" () in
      let* w = input ~shape:(s 3 1 1 1 1 5) ~name:"w" () in
      let* b = input ~shape:(s1c 3) ~name:"b" () in
      linear ~name:"y"
        { Linear.Linear.in_features = Dim.extent 5 }
        ~x ~weight:w ~bias:b ());
  [%expect
    {|
    linear: graph
            inputs:
              [t0 f32 [H=8 W=2 C=5] ->[n0],
               t1 f32 [N=3 T=1 D=1 H=1 W=1 C=5] ->[n0], t2 f32 [C=3] ->[n0]]
            nodes:
              n0: [t3 f32 [H=8 W=2 C=3]] =
                linear x=t0 weight=t1 bias=t2 params={in_features=5}
            outputs: [t3 f32 [H=8 W=2 C=3] <-n0] |}]

(* ---- op3-impl.md commit 8: Group 3 round trips --------------------------- *)

(* Sub has no [params_jsont] of its own -- it carries the same [Pointwise.Bin]
   payload [Add]/[Mul] do -- so this is evidence that the JSON case tag for
   THIS op decodes to the right compute, not a new codec. *)
let%expect_test "Sub: encode -> decode" =
  round_trip "sub"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* a = input ~shape:(s 1 1 1 2 3 4) ~name:"a" () in
      let* b = input ~shape:(s 1 1 1 2 3 4) ~name:"b" () in
      sub ~name:"y" a b);
  [%expect
    {|
    sub: graph
         inputs: [t0 f32 [H=2 W=3 C=4] ->[n0], t1 f32 [H=2 W=3 C=4] ->[n0]]
         nodes:
           n0: [t2 f32 [H=2 W=3 C=4]] = sub a=t0 b=t1
         outputs: [t2 f32 [H=2 W=3 C=4] <-n0] |}]

(* A non-trivial Reshape target: rank-changing (rank 3 -> rank 2) and not
   flattening to a single axis, so the codec's [Vec6.shape_jsont] has more than
   one non-default extent to lose. *)
let%expect_test "Reshape: encode -> decode, a rank-changing non-default target"
    =
  round_trip "reshape"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 2 3 4) ~name:"x" () in
      reshape ~name:"y" { Reshape.Reshape.shape = s 1 1 1 1 4 6 } x);
  [%expect
    {|
    reshape: graph
             inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
             nodes:
               n0: [t1 f32 [W=4 C=6]] = reshape x=t0 params={shape=[W=4 C=6]}
             outputs: [t1 f32 [W=4 C=6] <-n0] |}]

(* The non-identity Permute round trip is already pinned above ("op Permute:
   encode -> decode", a 3-cycle over H/W/C) -- op3-impl.md's exit condition is
   satisfied by that test, not restated here. *)

(* Pad introduces two JSON shapes no other op has: a per-axis SIGNED int pair,
   and a mode carried as a single-key union whose constant case holds a bare
   float. Both are exactly the kind [.ai/native_add_op.md] says earns a round
   trip.

   The fill is 0.1, which is not f32-exact: printing alone cannot tell a codec
   that narrowed it from one that did not, and the encoded form has to survive
   the trip as the SAME f32. A negative [before] rides along, since a codec that
   read the pair as unsigned would silently lose the crop. *)
let%expect_test "Pad: encode -> decode, signed entries and a non-exact fill" =
  round_trip "pad"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 4 5 2) ~name:"x" () in
      pad ~name:"y"
        {
          Pad.Pad.pads =
            [
              (Axis.H, { Pad.Pad.before = 2; after = 1 });
              (Axis.W, { Pad.Pad.before = -1; after = 0 });
            ];
          mode = Pad.Pad.Constant 0.1;
        }
        x);
  [%expect
    {|
    pad: graph
         inputs: [t0 f32 [H=4 W=5 C=2] ->[n0]]
         nodes:
           n0: [t1 f32 [H=7 W=4 C=2]] =
             pad x=t0 params={pads=[H:2,1, W:-1,0] mode=constant(0.1)}
         outputs: [t1 f32 [H=7 W=4 C=2] <-n0] |}]

let%expect_test "Pad: encode -> decode, the reflect mode has no fill to carry" =
  round_trip "pad_reflect"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 4 5 2) ~name:"x" () in
      pad ~name:"y"
        {
          Pad.Pad.pads = [ (Axis.W, { Pad.Pad.before = 3; after = 2 }) ];
          mode = Pad.Pad.Reflect;
        }
        x);
  [%expect
    {|
    pad_reflect: graph
                 inputs: [t0 f32 [H=4 W=5 C=2] ->[n0]]
                 nodes:
                   n0: [t1 f32 [H=4 W=10 C=2]] =
                     pad x=t0 params={pads=[W:3,2] mode=reflect}
                 outputs: [t1 f32 [H=4 W=10 C=2] <-n0] |}]

(* The round trip above cannot see the narrowing, and saying so is the point:
   [Json_util.f32_jsont] narrows on ENCODE, so an unnarrowed payload would still
   produce a narrowed JSON number and decode back to the same value. What the
   trip proves is the codec; what it cannot prove is the BUILDER.

   So read the fill straight off the built node, at full precision. 0.1 has no
   f32 representation, so a builder that skipped [f32_scalar] would print
   0.10000000000000001 -- and would then compute in a precision the payload
   cannot store. *)
let%expect_test "Pad: the builder narrows a Constant fill to f32" =
  let result =
    let open Err.Syntax in
    let+ g =
      lift_build
        Graph_builder.(
          build ~name:"g" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 4) ~name:"x" () in
          pad ~name:"y"
            {
              Pad.Pad.pads = [ (Axis.C, { Pad.Pad.before = 1; after = 0 }) ];
              mode = Pad.Pad.Constant 0.1;
            }
            x)
    in
    match g.Graph.nodes with
    | [ { Node.op = Graph_ir.Pad { Pad.Pad.params; _ }; _ } ] -> (
        match params.Pad.Pad.mode with
        | Pad.Pad.Constant v -> Printf.sprintf "%.17g" v
        | Pad.Pad.Reflect -> "reflect")
    | _ -> "unexpected graph"
  in
  Format.printf "%a@." (pp_result Format.pp_print_string) result;
  [%expect {| 0.10000000149011612 |}]

(* Slice's payload is three ints and an axis, which is not a new SHAPE of JSON
   the way Pad's signed pair and mode union were. It earns a round trip for a
   different reason: [step] is an [Op_config.Pos.t] on the OCaml side and a bare
   int on the wire, so the codec narrows on the way in, and the three bounds are
   easy to permute in an encoder without any of them failing to decode. The
   values are deliberately all different from one another. *)
let%expect_test "Slice: encode -> decode, bounds and a narrowed step" =
  round_trip "slice"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 4 9 2) ~name:"x" () in
      slice ~name:"y"
        { Split.Slice.axis = Axis.W; start = 1; stop = 8; step = pos 3 }
        x);
  [%expect
    {|
    slice: graph
           inputs: [t0 f32 [H=4 W=9 C=2] ->[n0]]
           nodes:
             n0: [t1 f32 [H=4 W=3 C=2]] =
               slice x=t0 params={axis=W start=1 stop=8 step=3}
           outputs: [t1 f32 [H=4 W=3 C=2] <-n0] |}]

(* [Select]'s payload is an axis and a bare int [index] -- the new SHAPE here
   (relative to [Slice]'s three-int/axis payload) is that a single int is
   easy to confuse with the AXIS position it's judged against on a decode
   with the fields permuted, so the two are given different values. *)
let%expect_test "Select: encode -> decode, axis and index" =
  round_trip "select"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 4 9 2) ~name:"x" () in
      select ~name:"y" { Split.Select.axis = Axis.W; index = 5 } x);
  [%expect
    {|
    select: graph
            inputs: [t0 f32 [H=4 W=9 C=2] ->[n0]]
            nodes:
              n0: [t1 f32 [W=4 C=2]] = select x=t0 params={axis=W index=5}
            outputs: [t1 f32 [W=4 C=2] <-n0] |}]

(* [Stack]'s payload is variadic like [Concat]'s, but the operand LIST itself
   is the only field -- no per-op axis/shape metadata to permute -- so the
   round trip that matters is arity: three operands, not silently truncated
   or reordered by the codec. *)
let%expect_test "Stack: encode -> decode, three operands" =
  round_trip "stack"
    Graph_builder.(
      build ~name:"g" ~outputs:(fun r -> [ r ])
      @@
      let* a = input ~shape:(s1c 3) ~name:"a" () in
      let* b = input ~shape:(s1c 3) ~name:"b" () in
      let* c = input ~shape:(s1c 3) ~name:"c" () in
      stack ~name:"y" { Concat.Stack.axis = Axis.W } [ a; b; c ]);
  [%expect
    {|
    stack: graph
           inputs:
             [t0 f32 [C=3] ->[n0], t1 f32 [C=3] ->[n0], t2 f32 [C=3] ->[n0]]
           nodes:
             n0: [t3 f32 [W=3 C=3]] = stack xs=[t0, t1, t2] params={axis=W}
           outputs: [t3 f32 [W=3 C=3] <-n0] |}]
