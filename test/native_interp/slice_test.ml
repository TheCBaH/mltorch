(* `aten.slice.Tensor` through the serialized path (op6.md row 6.2). Like
   [pad.default], no model reachable from this checkout serializes the target
   (op6-impl F1), so these fixtures are the only place it reaches
   [Native_interp] and no cram will regress if it breaks.

   What is asserted here is the LOWERING: how each serialized spelling of the
   bounds resolves against the metadata extent, and which requests are refused.
   The resolution itself is [Aten_shape.resolve_slice], shared with the bridge
   arm, and pinned directly in test/native/aten_shape_test.ml; what these
   fixtures add is that this path reaches it with the right extent and the right
   axis. Numeric agreement with real ATen is test/native_bridge_test.ml's job. *)

open Programs

(* [start]/[end] are optional in the schema, and the three spellings are
   different arguments as far as a decoder is concerned: ABSENT, an explicit
   null, and a value. All three appear below. *)
let slice_node ?dim ?start ?stop ?step () =
  let arg name a = jstr {|{"name":"%s","arg":%s,"kind":1}|} name a in
  let opt name = function None -> [] | Some v -> [ arg name v ] in
  let inputs =
    [ arg "self" (as_tensor "x") ]
    @ opt "dim" (Option.map (jstr "{\"as_int\":%s}") dim)
    @ opt "start" start @ opt "end" stop
    @ opt "step" (Option.map (jstr "{\"as_int\":%s}") step)
  in
  jstr
    {|{"target":"torch.ops.aten.slice.Tensor","inputs":[%s],"outputs":[%s],"metadata":{}}|}
    (String.concat "," inputs) (as_tensor "y")

let as_int n = jstr "{\"as_int\":%s}" n
let as_null = "{\"as_none\":true}"
let as_sym_int n = jstr "{\"as_sym_int\":{\"as_int\":%s}}" n
let as_sym_name n = jstr "{\"as_sym_int\":{\"as_name\":\"%s\"}}" n

let dump label ~x_sizes node =
  Format.printf "%s@." label;
  match
    lower (program ~x_sizes ~nodes:[ node ] ~graph_outputs:[ as_tensor "y" ] ())
  with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

(* The defaults are the whole axis, and they have to come from the EXTENT rather
   than from a constant: absent [end] on a rank-3 input sliced at dim 1 is 3,
   not 0 and not some fixed cap. *)
let%expect_test "slice.Tensor: absent bounds select the whole axis" =
  dump "no bounds at all:" ~x_sizes:[ 2; 3; 4 ] (slice_node ());
  dump "explicit nulls:" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"1" ~start:as_null ~stop:as_null ());
  [%expect
    {|
    no bounds at all:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=4]] =
        slice x=t0 params={axis=H start=0 stop=2 step=1}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0]
    explicit nulls:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=4]] =
        slice x=t0 params={axis=W start=0 stop=3 step=1}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0] |}]

(* Negative bounds normalize against the extent, and clamping is ACCEPT rather
   than reject: ATen clamps, so a node ATen runs has to lower rather than be
   refused. Only the resulting emptiness is a Native limitation. *)
let%expect_test "slice.Tensor: negative bounds normalize, wide bounds clamp" =
  dump "start=-2:" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"2" ~start:(as_int "-2") ());
  dump "end=-1:" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"2" ~stop:(as_int "-1") ());
  dump "end past the extent:" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"2" ~stop:(as_int "99") ());
  dump "start past the extent (empty, and refused):" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"2" ~start:(as_int "99") ());
  dump "start below -extent:" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"2" ~start:(as_int "-99") ());
  [%expect
    {|
    start=-2:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=2]] =
        slice x=t0 params={axis=C start=2 stop=4 step=1}
    outputs: [t1 f32 [H=2 W=3 C=2] <-n0]
    end=-1:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=3]] =
        slice x=t0 params={axis=C start=0 stop=3 step=1}
    outputs: [t1 f32 [H=2 W=3 C=3] <-n0]
    end past the extent:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=4]] =
        slice x=t0 params={axis=C start=0 stop=4 step=1}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0]
    start past the extent (empty, and refused):
      slice of axis C [4, 4) step 1 over extent 4 selects 0 elements; the engine has no empty extent
    start below -extent:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=4]] =
        slice x=t0 params={axis=C start=0 stop=4 step=1}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0] |}]

let%expect_test "slice.Tensor: step, exact and inexact" =
  dump "step 2 over 4 (exact):" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"2" ~step:"2" ());
  dump "step 3 over 4 (ceiling):" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"2" ~step:"3" ());
  dump "step 0:" ~x_sizes:[ 2; 3; 4 ] (slice_node ~dim:"2" ~step:"0" ());
  dump "step -1:" ~x_sizes:[ 2; 3; 4 ] (slice_node ~dim:"2" ~step:"-1" ());
  [%expect
    {|
    step 2 over 4 (exact):
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=2]] =
        slice x=t0 params={axis=C start=0 stop=4 step=2}
    outputs: [t1 f32 [H=2 W=3 C=2] <-n0]
    step 3 over 4 (ceiling):
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=2]] =
        slice x=t0 params={axis=C start=0 stop=4 step=3}
    outputs: [t1 f32 [H=2 W=3 C=2] <-n0]
    step 0:
      malformed PT2 graph: slice [none, none) step 0: slice step must be >= 1, got 0
    step -1:
      malformed PT2 graph: slice [none, none) step -1: slice step must be >= 1, got -1 |}]

(* [dim] goes through the same checked [axes_for_rank] every other arm uses, so
   a negative spelling names the same axis and an out-of-range one is refused by
   the shared rule rather than by anything written here. *)
let%expect_test "slice.Tensor: dim spellings" =
  dump "dim=-1:" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"-1" ~stop:(as_int "2") ());
  dump "dim=-3:" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"-3" ~stop:(as_int "1") ());
  dump "dim=3 (out of range):" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"3" ~stop:(as_int "1") ());
  [%expect
    {|
    dim=-1:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=2]] =
        slice x=t0 params={axis=C start=0 stop=2 step=1}
    outputs: [t1 f32 [H=2 W=3 C=2] <-n0]
    dim=-3:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [W=3 C=4]] = slice x=t0 params={axis=H start=0 stop=1 step=1}
    outputs: [t1 f32 [W=3 C=4] <-n0]
    dim=3 (out of range):
      malformed PT2 graph: invalid dimension 3 for rank 3 |}]

(* op6-impl decision 3, on this decoder. A RESOLVED SymInt is a value spelled
   differently and is accepted; a NAMED one is an unresolved symbol and is
   refused with the symbol in the message -- the same distinction tensor
   METADATA already drew, applied to an argument. Before this op no bound target
   had a SymInt argument, so both spellings used to reach [`Wrong_arg_kind]. *)
let%expect_test "slice.Tensor: resolved SymInt accepted, named SymInt refused" =
  dump "as_sym_int(1):" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"2" ~start:(as_sym_int "1") ());
  dump "as_sym_int named:" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"2" ~start:(as_sym_name "s0") ());
  dump "named on end:" ~x_sizes:[ 2; 3; 4 ]
    (slice_node ~dim:"2" ~stop:(as_sym_name "s17") ());
  [%expect
    {|
    as_sym_int(1):
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=3]] =
        slice x=t0 params={axis=C start=1 stop=4 step=1}
    outputs: [t1 f32 [H=2 W=3 C=3] <-n0]
    as_sym_int named:
      malformed PT2 graph: torch.ops.aten.slice.Tensor.start is the unresolved symbol "s0"
    named on end:
      malformed PT2 graph: torch.ops.aten.slice.Tensor.end is the unresolved symbol "s17" |}]
