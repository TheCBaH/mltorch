(* The exact `layer_norm.default` overload through the serialized path.

   Nothing in this repository's reach serialises it. Every ExportedProgram in
   the corpus lowers `aten.layer_norm.default` to `native_layer_norm.default`
   before writing model.json -- the 148 ViT nodes carry the decomposed target
   and record the functional one only in `metadata.from_node`. So these
   fixtures ARE the coverage for this arm: no cram, no downloaded model and no
   inference run will ever reach it.

   As with rms_norm, almost nothing here changes the output SHAPE. Normalizing
   over the leading axes instead of the trailing ones, or reading an affine
   operand on the wrong axis, produces a correctly-shaped wrong answer -- so
   these fixtures pin the AXES and the operand wiring the arm chose, and the
   numeric agreement with ATen lives in test/native_bridge_test.ml. *)

open Programs

let ln ?(normalized = "[4]") ?(weight = `Absent) ?(bias = `Absent) ?eps ?cudnn
    () =
  let opt_tensor name = function
    | `Absent -> ""
    | `None -> jstr {|,{"name":"%s","arg":{"as_none":true},"kind":1}|} name
    | `Tensor t -> jstr {|,{"name":"%s","arg":%s,"kind":1}|} name (as_tensor t)
  in
  let eps_arg =
    match eps with
    | None -> ""
    | Some (`Float e) ->
        jstr {|,{"name":"eps","arg":{"as_float":%.12g},"kind":1}|} e
    | Some (`Raw v) -> jstr {|,{"name":"eps","arg":%s,"kind":1}|} v
  in
  let cudnn_arg =
    match cudnn with
    | None -> ""
    | Some (`Bool b) ->
        jstr {|,{"name":"cudnn_enable","arg":{"as_bool":%b},"kind":1}|} b
    | Some (`Raw v) -> jstr {|,{"name":"cudnn_enable","arg":%s,"kind":1}|} v
  in
  jstr
    {|{"target":"torch.ops.aten.layer_norm.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"normalized_shape","arg":{"as_ints":%s},"kind":1}%s%s%s%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") normalized
    (opt_tensor "weight" weight)
    (opt_tensor "bias" bias) eps_arg cudnn_arg (as_tensor "y")

let prog ?(x_sizes = [ 2; 3; 4 ]) ?(captured = []) node =
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

let show label json =
  Format.printf "%-28s" label;
  match lower json with
  | Error e -> Format.printf "%a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok _ -> Format.printf "ok@."

(* A rank-3 input right-aligns onto H/W/C, so the single trailing axis is C --
   the arm must pick that one rather than the first. *)
let%expect_test "layer_norm.default normalizes the trailing axes" =
  dump "one axis:" (prog (ln ()));
  dump "two axes:" (prog (ln ~normalized:"[3,4]" ()));
  dump "all three:" (prog (ln ~normalized:"[2,3,4]" ()));
  [%expect
    {|
    one axis:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.layer_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=none bias=none params={dims=[C]; eps=1e-05}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0]
    two axes:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.layer_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=none bias=none params={dims=[W, C]; eps=1e-05}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0]
    all three:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.layer_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          layer_norm
            x=t0
            weight=none
            bias=none
            params={dims=[H, W, C]; eps=1e-05}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0] |}]

(* All four affine states, which are four structurally different graphs:
   [Graph_ir]'s [Layer_norm] carries weight and bias as INDEPENDENT options and
   [Eval_op] fills each absent one separately. "bias but no weight" is legal in
   the schema and is the state a paired encoding would get wrong. *)
let%expect_test "layer_norm.default wires each affine operand independently" =
  let cap = [ ("w", [ 4 ]); ("b", [ 4 ]) ] in
  dump "both:"
    (prog ~captured:cap (ln ~weight:(`Tensor "w") ~bias:(`Tensor "b") ()));
  dump "weight only:" (prog ~captured:cap (ln ~weight:(`Tensor "w") ()));
  dump "bias only:" (prog ~captured:cap (ln ~bias:(`Tensor "b") ()));
  dump "explicit nones:" (prog ~captured:cap (ln ~weight:`None ~bias:`None ()));
  [%expect
    {|
    both:
    graph
    inputs:
      [t0 f32 [H=2 W=3 C=4] ->[n0], t1 f32 [C=4] ->[n0] constant,
       t2 f32 [C=4] ->[n0] constant]
    nodes:
      group g1 torch.ops.aten.layer_norm.default:
        n0: [t3 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=t1 bias=t2 params={dims=[C]; eps=1e-05}
    outputs: [t3 f32 [H=2 W=3 C=4] <-n0]
    weight only:
    graph
    inputs:
      [t0 f32 [H=2 W=3 C=4] ->[n0], t1 f32 [C=4] ->[n0] constant,
       t2 f32 [C=4] constant]
    nodes:
      group g1 torch.ops.aten.layer_norm.default:
        n0: [t3 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=t1 bias=none params={dims=[C]; eps=1e-05}
    outputs: [t3 f32 [H=2 W=3 C=4] <-n0]
    bias only:
    graph
    inputs:
      [t0 f32 [H=2 W=3 C=4] ->[n0], t1 f32 [C=4] constant,
       t2 f32 [C=4] ->[n0] constant]
    nodes:
      group g1 torch.ops.aten.layer_norm.default:
        n0: [t3 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=none bias=t2 params={dims=[C]; eps=1e-05}
    outputs: [t3 f32 [H=2 W=3 C=4] <-n0]
    explicit nones:
    graph
    inputs:
      [t0 f32 [H=2 W=3 C=4] ->[n0], t1 f32 [C=4] constant, t2 f32 [C=4] constant]
    nodes:
      group g1 torch.ops.aten.layer_norm.default:
        n0: [t3 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=none bias=none params={dims=[C]; eps=1e-05}
    outputs: [t3 f32 [H=2 W=3 C=4] <-n0] |}]

(* eps is a REQUIRED float with a schema default of 1e-05 here, unlike
   rms_norm's [float?] whose absence means "ATen picks the machine epsilon".
   Both corpus values are pinned, and so is the defaulted case. *)
let%expect_test "layer_norm.default defaults eps to the schema's 1e-05" =
  dump "absent:" (prog (ln ()));
  dump "1e-5:" (prog (ln ~eps:(`Float 1e-5) ()));
  dump "1e-6:" (prog (ln ~eps:(`Float 1e-6) ()));
  [%expect
    {|
    absent:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.layer_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=none bias=none params={dims=[C]; eps=1e-05}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0]
    1e-5:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.layer_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=none bias=none params={dims=[C]; eps=1e-05}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0]
    1e-6:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.layer_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=none bias=none params={dims=[C]; eps=1e-06}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0] |}]

(* cudnn_enable is an implementation-selection hint that ATen's own composite
   discards -- [layer_norm_symint (..., bool /* cudnn_enable, deprecated */)].
   Both values are accepted with the same result, which is the documented
   backend-independent contract; rejecting [true] would reject the schema's own
   default. It is still DECODED, so a non-boolean is a malformed node rather
   than a silently ignored one. *)
let%expect_test "layer_norm.default accepts both cudnn_enable values" =
  show "absent:" (prog (ln ()));
  show "true:" (prog (ln ~cudnn:(`Bool true) ()));
  show "false:" (prog (ln ~cudnn:(`Bool false) ()));
  show "not a bool:" (prog (ln ~cudnn:(`Raw {|{"as_int":1}|}) ()));
  [%expect
    {|
    absent:                     ok
    true:                       ok
    false:                      ok
    not a bool:                 malformed PT2 graph: torch.ops.aten.layer_norm.default.cudnn_enable is not a bool |}]

(* normalized_shape is checked against the input's trailing extents, not merely
   for its length: a shape naming extents the input does not have would
   normalize over the wrong axes and return a plausible wrong answer. *)
let%expect_test "layer_norm.default validates normalized_shape, not its length"
    =
  show "empty:" (prog (ln ~normalized:"[]" ()));
  show "longer than rank:" (prog (ln ~normalized:"[2,3,4,5]" ()));
  show "wrong extent:" (prog (ln ~normalized:"[5]" ()));
  show "right length, wrong values:" (prog (ln ~normalized:"[4,3]" ()));
  [%expect
    {|
    empty:                      malformed PT2 graph: layer_norm: normalized_shape has 0 entries, outside [1, 3] for this rank
    longer than rank:           malformed PT2 graph: layer_norm: normalized_shape has 4 entries, outside [1, 3] for this rank
    wrong extent:               malformed PT2 graph: layer_norm: normalized_shape [5] does not match the input's trailing extents [4]
    right length, wrong values: malformed PT2 graph: layer_norm: normalized_shape [4,3] does not match the input's trailing extents [3,4] |}]

(* Each affine operand carries the whole normalized shape, so its RANK is k --
   not 1, and not the input's. Checked separately for each, so the diagnostic
   says which one disagreed. *)
let%expect_test "layer_norm.default checks each affine operand's rank" =
  let cap =
    [ ("w", [ 4 ]); ("b", [ 4 ]); ("w2", [ 3; 4 ]); ("b2", [ 3; 4 ]) ]
  in
  show "k=2, rank-2 affine:"
    (prog ~captured:cap
       (ln ~normalized:"[3,4]" ~weight:(`Tensor "w2") ~bias:(`Tensor "b2") ()));
  show "k=2, rank-1 weight:"
    (prog ~captured:cap
       (ln ~normalized:"[3,4]" ~weight:(`Tensor "w") ~bias:(`Tensor "b2") ()));
  show "k=2, rank-1 bias:"
    (prog ~captured:cap
       (ln ~normalized:"[3,4]" ~weight:(`Tensor "w2") ~bias:(`Tensor "b") ()));
  [%expect
    {|
    k=2, rank-2 affine:         ok
    k=2, rank-1 weight:         malformed PT2 graph: w is rank 1, expected 2
    k=2, rank-1 bias:           malformed PT2 graph: b is rank 1, expected 2 |}]

(* eps decoded with the wrong kind is a malformed node, not a default. *)
let%expect_test "layer_norm.default rejects a malformed eps" =
  show "as_int:" (prog (ln ~eps:(`Raw {|{"as_int":1}|}) ()));
  show "as_bool:" (prog (ln ~eps:(`Raw {|{"as_bool":true}|}) ()));
  show "as_none:" (prog (ln ~eps:(`Raw {|{"as_none":true}|}) ()));
  [%expect
    {|
    as_int:                     malformed PT2 graph: torch.ops.aten.layer_norm.default.eps is not a float
    as_bool:                    malformed PT2 graph: torch.ops.aten.layer_norm.default.eps is not a float
    as_none:                    malformed PT2 graph: torch.ops.aten.layer_norm.default.eps is not a float |}]

(* ---- native_layer_norm.default ------------------------------------------ *)

(* The DECOMPOSED target, and the one that actually carries model coverage: 148
   nodes across four ViT models, of which vit_b_32 is downloadable. It shares
   this arm's whole body with the functional overload and differs in three
   things -- [eps] is required with no schema default, there is no
   [cudnn_enable], and it returns [(out, mean, rstd)] where Native's
   [Layer_norm] has one output. *)
let nln ?(normalized = "[4]") ?(weight = `Absent) ?(bias = `Absent)
    ?(eps = `Float 1e-06) ?(outs = [ "y"; "mean"; "rstd" ]) () =
  let opt_tensor name = function
    | `Absent -> ""
    | `None -> jstr {|,{"name":"%s","arg":{"as_none":true},"kind":1}|} name
    | `Tensor t -> jstr {|,{"name":"%s","arg":%s,"kind":1}|} name (as_tensor t)
  in
  let eps_arg =
    match eps with
    | `Absent -> ""
    | `Float e -> jstr {|,{"name":"eps","arg":{"as_float":%.12g},"kind":1}|} e
    | `Raw v -> jstr {|,{"name":"eps","arg":%s,"kind":1}|} v
  in
  jstr
    {|{"target":"torch.ops.aten.native_layer_norm.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"normalized_shape","arg":{"as_ints":%s},"kind":1}%s%s%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") normalized
    (opt_tensor "weight" weight)
    (opt_tensor "bias" bias) eps_arg
    (String.concat "," (List.map as_tensor outs))

(* The shape of every corpus occurrence: rank 3, one normalized axis, both
   affine operands present, eps spelled 1e-06, and the two statistics dead. Only
   the first output is bound -- the graph has one node with one result. *)
let%expect_test "native_layer_norm.default drops mean and rstd" =
  dump "corpus shape:"
    (prog
       ~captured:[ ("w", [ 4 ]); ("b", [ 4 ]) ]
       (nln ~weight:(`Tensor "w") ~bias:(`Tensor "b") ()));
  dump "no affine:" (prog (nln ()));
  [%expect
    {|
    corpus shape:
    graph
    inputs:
      [t0 f32 [H=2 W=3 C=4] ->[n0], t1 f32 [C=4] ->[n0] constant,
       t2 f32 [C=4] ->[n0] constant]
    nodes:
      group g1 torch.ops.aten.native_layer_norm.default:
        n0: [t3 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=t1 bias=t2 params={dims=[C]; eps=1e-06}
    outputs: [t3 f32 [H=2 W=3 C=4] <-n0]
    no affine:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      group g1 torch.ops.aten.native_layer_norm.default:
        n0: [t1 f32 [H=2 W=3 C=4]] =
          layer_norm x=t0 weight=none bias=none params={dims=[C]; eps=1e-06}
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0] |}]

(* [eps] has NO default in this schema, unlike the functional overload's 1e-05.
   Reading it with a default would silently substitute a number the model never
   supplied -- and 1e-05 is not even the value the corpus uses. *)
let%expect_test "native_layer_norm.default requires eps" =
  show "omitted:" (prog (nln ~eps:`Absent ()));
  show "1e-06:" (prog (nln ~eps:(`Float 1e-06) ()));
  show "1e-05:" (prog (nln ~eps:(`Float 1e-05) ()));
  show "not a float:" (prog (nln ~eps:(`Raw {|{"as_none":true}|}) ()));
  [%expect
    {|
    omitted:                    malformed PT2 graph: torch.ops.aten.native_layer_norm.default: missing argument "eps"
    1e-06:                      ok
    1e-05:                      ok
    not a float:                malformed PT2 graph: torch.ops.aten.native_layer_norm.default.eps is not a float |}]

(* Dropping [mean] and [rstd] is sound only while nothing reads them. This is
   the check that says so, and it must be seen failing to be worth anything --
   no corpus node reaches it, so the fixture is the only evidence it can have.

   The diagnostic names the op that cannot provide the statistic. Without the
   check the dropped name is simply never bound and the failure surfaces at the
   CONSUMER, as [`Undefined_ssa "mean"] on a node that did nothing wrong. *)
let%expect_test "native_layer_norm.default refuses a live mean or rstd" =
  let consumer src =
    jstr
      {|{"target":"torch.ops.aten.relu.default","inputs":[{"name":"self","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
      (as_tensor src) (as_tensor "r")
  in
  let with_consumer src =
    program ~x_sizes:[ 2; 3; 4 ]
      ~nodes:[ nln (); consumer src ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "mean read by a node:" (with_consumer "mean");
  show "rstd read by a node:" (with_consumer "rstd");
  (* A graph OUTPUT is a read too, and it is not a node input -- collecting
     only node inputs would let this one through. *)
  show "mean is a graph output:"
    (program ~x_sizes:[ 2; 3; 4 ]
       ~nodes:[ nln () ]
       ~graph_outputs:[ as_tensor "y"; as_tensor "mean" ]
       ());
  (* The output itself being read is not a stat read. *)
  show "only y is read:" (prog (nln ()));
  [%expect
    {|
    mean read by a node:        malformed PT2 graph: torch.ops.aten.native_layer_norm.default: mean output "mean" is read, and this graph does not have it
    rstd read by a node:        malformed PT2 graph: torch.ops.aten.native_layer_norm.default: rstd output "rstd" is read, and this graph does not have it
    mean is a graph output:     malformed PT2 graph: torch.ops.aten.native_layer_norm.default: mean output "mean" is read, and this graph does not have it
    only y is read:             ok |}]

(* [materialized_output_names] keeps only the head, so a node declaring the
   wrong number of outputs would pass the generic arity check that compares
   names against ids -- one against one -- while dropping a different number of
   results than the op has. Checked in the arm for that reason. *)
let%expect_test "native_layer_norm.default checks its output arity" =
  show "two outputs:" (prog (nln ~outs:[ "y"; "mean" ] ()));
  show "one output:" (prog (nln ~outs:[ "y" ] ()));
  show "four outputs:" (prog (nln ~outs:[ "y"; "mean"; "rstd"; "extra" ] ()));
  [%expect
    {|
    two outputs:                malformed PT2 graph: torch.ops.aten.native_layer_norm.default declares 2 outputs but produces 3
    one output:                 malformed PT2 graph: torch.ops.aten.native_layer_norm.default declares 1 outputs but produces 3
    four outputs:               malformed PT2 graph: torch.ops.aten.native_layer_norm.default declares 4 outputs but produces 3 |}]

(* The validation shared with the functional overload still runs, and its
   diagnostics name THIS target rather than the one the body is shared with. *)
let%expect_test "native_layer_norm.default validates normalized_shape" =
  show "empty:" (prog (nln ~normalized:"[]" ()));
  show "longer than the rank:" (prog (nln ~normalized:"[2,3,4,5]" ()));
  show "wrong extent:" (prog (nln ~normalized:"[5]" ()));
  show "right length, wrong values:" (prog (nln ~normalized:"[2,4]" ()));
  show "wrong-rank weight:"
    (prog
       ~captured:[ ("w", [ 4 ]) ]
       (nln ~normalized:"[3,4]" ~weight:(`Tensor "w") ()));
  [%expect
    {|
    empty:                      malformed PT2 graph: native_layer_norm: normalized_shape has 0 entries, outside [1, 3] for this rank
    longer than the rank:       malformed PT2 graph: native_layer_norm: normalized_shape has 4 entries, outside [1, 3] for this rank
    wrong extent:               malformed PT2 graph: native_layer_norm: normalized_shape [5] does not match the input's trailing extents [4]
    right length, wrong values: malformed PT2 graph: native_layer_norm: normalized_shape [2,4] does not match the input's trailing extents [3,4]
    wrong-rank weight:          malformed PT2 graph: w is rank 1, expected 2 |}]
