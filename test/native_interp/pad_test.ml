(* `aten.pad.default` through the serialized path (op6.md row 6.1). No model
   reachable from this checkout serializes the target at all -- not the 23
   core-ATen graphs in modules/pytorch.models.pt2, not the five downloadable
   release models (op6-impl F1) -- so these fixtures are the only place it
   reaches [Native_interp], and there is no cram that will regress if it breaks.

   What is asserted here is the LOWERING: which frame axes the serialized pad
   list lands on, and which configurations are refused. Numeric agreement with
   real ATen is test/native_bridge_test.ml's job, whose harness runs both
   backends. *)

open Programs

let pad_node ?mode ?value ~pad () =
  let arg name a = jstr {|{"name":"%s","arg":%s,"kind":1}|} name a in
  let inputs =
    [ arg "self" (as_tensor "x"); arg "pad" (jstr "{\"as_ints\":[%s]}" pad) ]
    @ (match mode with
      | None -> []
      | Some m -> [ arg "mode" (jstr {|{"as_string":"%s"}|} m) ])
    @
    match value with
    | None -> []
    | Some v -> [ arg "value" (jstr {|{"as_float":%s}|} v) ]
  in
  jstr
    {|{"target":"torch.ops.aten.pad.default","inputs":[%s],"outputs":[%s],"metadata":{}}|}
    (String.concat "," inputs) (as_tensor "y")

let dump label ~x_sizes node =
  Format.printf "%s@." label;
  match
    lower (program ~x_sizes ~nodes:[ node ] ~graph_outputs:[ as_tensor "y" ] ())
  with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

(* THE ordering test. PyTorch's list is innermost-first, so on a rank-3 input
   [1,2, 3,4] means "W by (1,2), H by (3,4)" -- H's amounts are the LAST pair,
   not the first. Every amount is distinct so a reversal, a swap, or an
   off-by-one pairing each produce a different printed payload. *)
let%expect_test "pad.default: the serialized list is innermost-first" =
  dump "one pair (innermost dim only):" ~x_sizes:[ 2; 3; 4 ]
    (pad_node ~pad:"1,2" ());
  dump "two pairs:" ~x_sizes:[ 2; 3; 4 ] (pad_node ~pad:"1,2,3,4" ());
  dump "three pairs (every dim):" ~x_sizes:[ 2; 3; 4 ]
    (pad_node ~pad:"1,2,3,4,5,6" ());
  [%expect
    {|
    one pair (innermost dim only):
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=7]] =
        pad x=t0 params={pads=[C:1,2] mode=constant(0)}
    outputs: [t1 f32 [H=2 W=3 C=7] <-n0]
    two pairs:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=10 C=7]] =
        pad x=t0 params={pads=[W:3,4, C:1,2] mode=constant(0)}
    outputs: [t1 f32 [H=2 W=10 C=7] <-n0]
    three pairs (every dim):
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=13 W=10 C=7]] =
        pad x=t0 params={pads=[H:5,6, W:3,4, C:1,2] mode=constant(0)}
    outputs: [t1 f32 [H=13 W=10 C=7] <-n0] |}]

let%expect_test "pad.default: absent mode and value mean constant zero" =
  dump "no mode, no value:" ~x_sizes:[ 2; 3 ] (pad_node ~pad:"1,1" ());
  dump "explicit constant, no value:" ~x_sizes:[ 2; 3 ]
    (pad_node ~mode:"constant" ~pad:"1,1" ());
  dump "explicit value:" ~x_sizes:[ 2; 3 ]
    (pad_node ~mode:"constant" ~value:"-2.5" ~pad:"1,1" ());
  [%expect
    {|
    no mode, no value:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=5]] = pad x=t0 params={pads=[C:1,1] mode=constant(0)}
    outputs: [t1 f32 [W=2 C=5] <-n0]
    explicit constant, no value:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=5]] = pad x=t0 params={pads=[C:1,1] mode=constant(0)}
    outputs: [t1 f32 [W=2 C=5] <-n0]
    explicit value:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=5]] = pad x=t0 params={pads=[C:1,1] mode=constant(-2.5)}
    outputs: [t1 f32 [W=2 C=5] <-n0] |}]

(* Cropping: ATen accepts negative amounts and so does this engine. The only
   boundary is the resulting extent, and it belongs to the shape rule -- which
   is why the refusal below reads as a Native shape error rather than as an
   importer one. *)
let%expect_test "pad.default: negative amounts crop" =
  dump "crop both sides of W:" ~x_sizes:[ 4; 5 ] (pad_node ~pad:"-1,-1" ());
  dump "crop W, pad H:" ~x_sizes:[ 4; 5 ] (pad_node ~pad:"-1,0,2,2" ());
  dump "crop consumes the axis:" ~x_sizes:[ 4; 5 ] (pad_node ~pad:"-3,-2" ());
  [%expect
    {|
    crop both sides of W:
    graph
    inputs: [t0 f32 [W=4 C=5] ->[n0]]
    nodes:
      n0: [t1 f32 [W=4 C=3]] = pad x=t0 params={pads=[C:-1,-1] mode=constant(0)}
    outputs: [t1 f32 [W=4 C=3] <-n0]
    crop W, pad H:
    graph
    inputs: [t0 f32 [W=4 C=5] ->[n0]]
    nodes:
      n0: [t1 f32 [W=8 C=4]] =
        pad x=t0 params={pads=[W:2,2, C:-1,0] mode=constant(0)}
    outputs: [t1 f32 [W=8 C=4] <-n0]
    crop consumes the axis:
      pad of axis C by (-3, -2) over extent 5 leaves 0 elements; the engine has no empty extent |}]

let%expect_test "pad.default: reflect, and the ranks ATen defines it for" =
  dump "2 pads on rank 3:" ~x_sizes:[ 2; 3; 4 ]
    (pad_node ~mode:"reflect" ~pad:"1,1" ());
  dump "4 pads on rank 4:" ~x_sizes:[ 1; 2; 5; 5 ]
    (pad_node ~mode:"reflect" ~pad:"1,1,2,2" ());
  (* ATen has no reflect kernel for this (pad size, rank) pair: 2 pads need a
     rank-2 or rank-3 input. Accepting it would turn a refusal into a walk
     MISMATCH, which reads as a wrong answer rather than as a boundary. *)
  dump "2 pads on rank 4:" ~x_sizes:[ 1; 2; 5; 5 ]
    (pad_node ~mode:"reflect" ~pad:"1,1" ());
  (* Reflect mirrors about the boundary element, so a pad as wide as the axis
     has nothing to mirror. PyTorch refuses the same configuration. *)
  dump "reflect wider than the axis:" ~x_sizes:[ 2; 3 ]
    (pad_node ~mode:"reflect" ~pad:"3,0" ());
  [%expect
    {|
    2 pads on rank 3:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=6]] = pad x=t0 params={pads=[C:1,1] mode=reflect}
    outputs: [t1 f32 [H=2 W=3 C=6] <-n0]
    4 pads on rank 4:
    graph
    inputs: [t0 f32 [H=2 W=5 C=5] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=9 C=7]] =
        pad x=t0 params={pads=[W:2,2, C:1,1] mode=reflect}
    outputs: [t1 f32 [H=2 W=9 C=7] <-n0]
    2 pads on rank 4:
      malformed PT2 graph: pad mode "reflect" over 1 dimension is not defined for a rank-4 input
    reflect wider than the axis:
      reflect pad of axis C by (3, 0) needs each side below the extent 3 |}]

let%expect_test "pad.default: the refusals" =
  dump "odd list length:" ~x_sizes:[ 2; 3 ] (pad_node ~pad:"1,2,3" ());
  dump "more pairs than dims:" ~x_sizes:[ 2; 3 ]
    (pad_node ~pad:"1,1,1,1,1,1" ());
  dump "replicate:" ~x_sizes:[ 2; 3 ] (pad_node ~mode:"replicate" ~pad:"1,1" ());
  dump "circular:" ~x_sizes:[ 2; 3 ] (pad_node ~mode:"circular" ~pad:"1,1" ());
  dump "unknown mode:" ~x_sizes:[ 2; 3 ] (pad_node ~mode:"mirror" ~pad:"1,1" ());
  (* ATen's rule is [!value.has_value() || *value == 0], so an explicit zero is
     accepted with reflect and a non-zero one is not. *)
  dump "reflect with value 0:" ~x_sizes:[ 2; 3 ]
    (pad_node ~mode:"reflect" ~value:"0.0" ~pad:"1,1" ());
  dump "reflect with a non-zero value:" ~x_sizes:[ 2; 3 ]
    (pad_node ~mode:"reflect" ~value:"1.5" ~pad:"1,1" ());
  [%expect
    {|
    odd list length:
      malformed PT2 graph: pad list has 3 entries; a pair per padded dimension is required
    more pairs than dims:
      malformed PT2 graph: pad list covers 3 dimensions of a rank-2 input
    replicate:
      malformed PT2 graph: pad mode "replicate" is outside the Native domain (constant and reflect are supported)
    circular:
      malformed PT2 graph: pad mode "circular" is outside the Native domain (constant and reflect are supported)
    unknown mode:
      malformed PT2 graph: pad mode "mirror" is outside the Native domain (constant and reflect are supported)
    reflect with value 0:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=5]] = pad x=t0 params={pads=[C:1,1] mode=reflect}
    outputs: [t1 f32 [W=2 C=5] <-n0]
    reflect with a non-zero value:
      malformed PT2 graph: pad mode "reflect" takes no non-zero value argument |}]
