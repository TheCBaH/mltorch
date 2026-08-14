(* The exact `rms_norm.default` overload through the serialized path.

   As with the rest of Group 2, nothing else reaches this arm: no downloadable
   model serialises it (vit_b_32, the only transformer in the corpus, carries
   `native_layer_norm.default`).

   The property that makes this row different from the others is that almost
   nothing here changes the output SHAPE. Normalizing over the leading axes
   instead of the trailing ones, dividing by the wrong count, or reading the
   weight on the wrong axis all produce a correctly-shaped wrong answer -- so the
   structural fixtures below pin the AXES the arm chose, and the numeric
   agreement with ATen lives in test/native_bridge_test.ml. *)

open Programs

let rms ?(normalized = "[4]") ?(weight = `Absent) ?eps () =
  let weight_arg =
    match weight with
    | `Absent -> ""
    | `None -> {|,{"name":"weight","arg":{"as_none":true},"kind":1}|}
    | `Tensor -> jstr {|,{"name":"weight","arg":%s,"kind":1}|} (as_tensor "w")
  in
  let eps_arg =
    match eps with
    | None -> ""
    | Some `None -> {|,{"name":"eps","arg":{"as_none":true},"kind":1}|}
    | Some (`Float e) ->
        jstr {|,{"name":"eps","arg":{"as_float":%.12g},"kind":1}|} e
    | Some (`Raw v) -> jstr {|,{"name":"eps","arg":%s,"kind":1}|} v
  in
  jstr
    {|{"target":"torch.ops.aten.rms_norm.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"normalized_shape","arg":{"as_ints":%s},"kind":1}%s%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") normalized weight_arg eps_arg (as_tensor "y")

let prog ?(x_sizes = [ 2; 3; 4 ]) ?w_sizes node =
  let captured = match w_sizes with None -> [] | Some s -> [ ("w", s) ] in
  program ~x_sizes
    ~extra_tensor_values:(List.map (fun (n, s) -> (n, tensor_meta s)) captured)
    ~params:(List.map fst captured) ~nodes:[ node ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

(* The single-axis case, which is what a per-channel rms_norm looks like. The
   frame axes matter: a rank-3 input is right-aligned onto D/H/W... C, so the
   trailing axis is C and the arm must pick that one rather than the first. *)
let%expect_test "rms_norm.default normalizes the trailing axis" =
  dump "one axis:" (prog (rms ()));
  [%expect
    {|
    one axis:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.rms_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          rms_norm x=t0 weight=none params={dims=[C]; eps=1.19209e-07}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0] |}]

let%expect_test "a multi-axis normalized_shape takes the trailing k axes" =
  dump "two axes:" (prog (rms ~normalized:"[3,4]" ()));
  dump "all three:" (prog (rms ~normalized:"[2,3,4]" ()));
  [%expect
    {|
    two axes:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.rms_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          rms_norm x=t0 weight=none params={dims=[W, C]; eps=1.19209e-07}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0]
    all three:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.rms_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          rms_norm x=t0 weight=none params={dims=[H, W, C]; eps=1.19209e-07}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0] |}]

(* The absent weight must reach [Graph_builder.rms_norm]'s [?weight] as None,
   not as a synthesized ones tensor. Native4D's optional-weight arm
   (lower.ml:293-299) is only reachable if it does. *)
let%expect_test "an absent weight stays absent" =
  dump "absent:" (prog (rms ()));
  dump "explicit none:" (prog (rms ~weight:`None ()));
  [%expect
    {|
    absent:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.rms_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          rms_norm x=t0 weight=none params={dims=[C]; eps=1.19209e-07}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0]
    explicit none:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.rms_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          rms_norm x=t0 weight=none params={dims=[C]; eps=1.19209e-07}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0] |}]

let%expect_test "a present weight is wired in unrelayouted" =
  dump "weight [4]:" (prog ~w_sizes:[ 4 ] (rms ~weight:`Tensor ()));
  [%expect
    {|
    weight [4]:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0], t1 f32 [C=4] ->[n0] constant]
    nodes:
      group g1 torch.ops.aten.rms_norm.default:
        n0: [t2 f32 [H=2 W=3 C=4]] =
          rms_norm x=t0 weight=t1 params={dims=[C]; eps=1.19209e-07}
    outputs: [t2 f32 [H=2 W=3 C=4] <-n0] |}]

(* [dump], not [show]: a node count says nothing about eps, which is the whole
   claim. The default is [Norm.RmsNorm.default_eps], the one spelling both
   importers now read. *)
(* Three spellings of "use the default epsilon", not two. [eps] is a [float?] and
   the generated op-spec path serialises [Float_opt None] as an explicit
   [Argument.None], so a node that means the default arrives written out --
   which [float_arg] used to reject as a wrong argument kind. *)
let%expect_test "eps defaults to float32 epsilon and is otherwise carried" =
  dump "omitted:" (prog (rms ()));
  dump "explicit none:" (prog (rms ~eps:`None ()));
  dump "explicit:" (prog (rms ~eps:(`Float 1e-5) ()));
  [%expect
    {|
    omitted:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.rms_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          rms_norm x=t0 weight=none params={dims=[C]; eps=1.19209e-07}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0]
    explicit none:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.rms_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          rms_norm x=t0 weight=none params={dims=[C]; eps=1.19209e-07}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0]
    explicit:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.rms_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          rms_norm x=t0 weight=none params={dims=[C]; eps=1e-05}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0] |}]

(* An eps that is neither a float nor a none. [Op_bridge] used to read every
   non-float value as the default, so a malformed node was accepted there and
   refused here -- the acceptance divergence this group exists to close. Both
   now reject. *)
let%expect_test "a non-float, non-none eps is refused" =
  show "eps as_int:" (prog (rms ~eps:(`Raw {|{"as_int":1}|}) ()));
  show "eps as_bool:" (prog (rms ~eps:(`Raw {|{"as_bool":true}|}) ()));
  show "eps as_string:" (prog (rms ~eps:(`Raw {|{"as_string":"1e-5"}|}) ()));
  [%expect
    {|
    eps as_int:                malformed PT2 graph: torch.ops.aten.rms_norm.default.eps is not a float
    eps as_bool:               malformed PT2 graph: torch.ops.aten.rms_norm.default.eps is not a float
    eps as_string:             malformed PT2 graph: torch.ops.aten.rms_norm.default.eps is not a float |}]

(* ---- rejections --------------------------------------------------------- *)

let%expect_test "normalized_shape must fit inside the rank" =
  show "empty:" (prog (rms ~normalized:"[]" ()));
  show "longer than rank:" (prog (rms ~normalized:"[2,3,4,5]" ()));
  [%expect
    {|
    empty:                     malformed PT2 graph: normalized_shape has 0 entries, outside [1, 3] for this rank
    longer than rank:          malformed PT2 graph: normalized_shape has 4 entries, outside [1, 3] for this rank |}]

(* The check the bridge does not do. Its arm reads the LENGTH of
   normalized_shape and nothing else, so each of these silently normalizes over
   axes whose extents are not the ones the model named. *)
let%expect_test "normalized_shape must match the input's trailing extents" =
  show "wrong extent:" (prog (rms ~normalized:"[5]" ()));
  show "right length, wrong values:" (prog (rms ~normalized:"[4,3]" ()));
  [%expect
    {|
    wrong extent:              malformed PT2 graph: normalized_shape [5] does not match the input's trailing extents [4]
    right length, wrong values: malformed PT2 graph: normalized_shape [4,3] does not match the input's trailing extents [3,4] |}]

(* The weight carries the input's extent on each NORMALIZED axis and is
   extent-1 elsewhere, and nothing checked it. A weight of the wrong extent --
   or of the right extent on the wrong axis -- normalized a correctly-shaped
   graph and then read out of bounds during evaluation. *)
let%expect_test "an rms_norm weight of the wrong shape is refused" =
  show "weight [3], normalized [4]:"
    (prog ~w_sizes:[ 3 ] (rms ~weight:`Tensor ()));
  show "weight [4], normalized [3,4]:"
    (prog ~w_sizes:[ 4 ] (rms ~normalized:"[3,4]" ~weight:`Tensor ()));
  show "weight [3,4], normalized [3,4]:"
    (prog ~w_sizes:[ 3; 4 ] (rms ~normalized:"[3,4]" ~weight:`Tensor ()));
  [%expect
    {|
    weight [3], normalized [4]: rms_norm weight shape must be [C=4], got
    [C=3]
    weight [4], normalized [3,4]: malformed PT2 graph: w is rank 1, expected 2
    weight [3,4], normalized [3,4]: lowered, nodes=1 |}]

(* Same rank erasure on the norm weight: [1,4] right-aligns onto the extents of
   [4], so only the declared rank separates them. Here the required rank is [k],
   the length of normalized_shape, not 1. *)
let%expect_test "a leading-singleton rms_norm weight is refused on rank" =
  show "weight [1,4], normalized [4]:"
    (prog ~w_sizes:[ 1; 4 ] (rms ~weight:`Tensor ()));
  show "weight [4], normalized [4]:"
    (prog ~w_sizes:[ 4 ] (rms ~weight:`Tensor ()));
  [%expect
    {|
    weight [1,4], normalized [4]: malformed PT2 graph: w is rank 2, expected 1
    weight [4], normalized [4]: lowered, nodes=1 |}]

let%expect_test "missing input metadata names the rms_norm role" =
  show "no input meta:"
    (jstr
       {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[{"user_input":{"arg":%s}}],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
       (as_tensor "x") (as_tensor "y") (rms ()) (as_tensor "x") (as_tensor "y"));
  [%expect
    {| no input meta:             malformed PT2 graph: no tensor metadata for "x" |}]
