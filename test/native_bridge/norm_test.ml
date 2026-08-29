(* layer_norm/native_layer_norm dispatch arms, and transpose.int. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/norm_test.ml]. *)

open Helpers

(* ---- the layer_norm.default arm ----------------------------------------- *)

(* Both affine operands are optional in the schema and in [Graph_ir], so all
   four states are dispatched, and the GRAPH is printed for each. An arm that
   materialised a ones/zeros tensor for an absent operand would build a
   structurally different graph from [Native_interp]'s for the same node and
   leave that arm unreachable -- the bug the rms_norm block below records at
   length. "bias but no weight" is the state no exported model produces and the
   one a paired encoding gets wrong, so it is dispatched rather than assumed.

   The values are the claim beneath the structure: with weight all ones and bias
   all zeros the four states must agree elementwise, and they do. *)
let%expect_test "dispatch: layer_norm.default builds only the operands present"
    =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let ones = float_tensor [ 3 ] [ 1.; 1.; 1. ] in
  let zeros = float_tensor [ 3 ] [ 0.; 0.; 0. ] in
  let state label affine =
    Format.printf "%s@." label;
    dispatch_print_with_graph ~print_graph:true
      ~target:"torch.ops.aten.layer_norm.default"
      ~bindings:
        ([ ("input", x) ]
        @ List.map (fun n -> (n, if n = "weight" then ones else zeros)) affine)
      ~inputs:
        ([ in_tensor "input"; in_ints "normalized_shape" [ 3 ] ]
        @ List.map in_tensor affine
        @ [ in_float "eps" 1e-5 ])
      ~noutputs:1
  in
  state "neither:" [];
  state "weight:" [ "weight" ];
  state "bias:" [ "bias" ];
  state "both:" [ "weight"; "bias" ];
  [%expect
    {|
    neither:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] =
        layer_norm x=t0 weight=none bias=none params={dims=[C]; eps=1e-05}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    weight:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [C=3] ->[n0]]
    nodes:
      n0: [t2 f32 [W=2 C=3]] =
        layer_norm x=t0 weight=t1 bias=none params={dims=[C]; eps=1e-05}
    outputs: [t2 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    bias:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [C=3] ->[n0]]
    nodes:
      n0: [t2 f32 [W=2 C=3]] =
        layer_norm x=t0 weight=none bias=t1 params={dims=[C]; eps=1e-05}
    outputs: [t2 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    both:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [C=3] ->[n0], t2 f32 [C=3] ->[n0]]
    nodes:
      n0: [t3 f32 [W=2 C=3]] =
        layer_norm x=t0 weight=t1 bias=t2 params={dims=[C]; eps=1e-05}
    outputs: [t3 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474} |}]

(* A weight that is not all ones and a bias that is not all zeros, both
   negative-valued somewhere: [* weight] then [+ bias] and the reverse order
   give different answers only when both are non-trivial. *)
let%expect_test "dispatch: layer_norm.default applies weight then bias" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 3 ] [ 2.; -1.; 0.5 ] in
  let b = float_tensor [ 3 ] [ 0.25; 0.; -1. ] in
  dispatch_print ~target:"torch.ops.aten.layer_norm.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", b) ]
    ~inputs:
      [
        in_tensor "input";
        in_ints "normalized_shape" [ 3 ];
        in_tensor "weight";
        in_tensor "bias";
        in_float "eps" 1e-5;
      ]
    ~noutputs:1;
  [%expect
    {| tensor f32 [W=2 C=3] {-2.19947, 0, -0.387632, -2.19947, 0, -0.387632} |}]

(* Same validation as rms_norm -- [normalized_dims] is shared -- but the
   diagnostics now name the op that failed, which is what the [op] field added
   to [Normalized_rank]/[Normalized_shape] buys. *)
let%expect_test "dispatch: layer_norm validates normalized_shape" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let bad normalized_shape =
    dispatch_print ~target:"torch.ops.aten.layer_norm.default"
      ~bindings:[ ("input", x) ]
      ~inputs:[ in_tensor "input"; in_ints "normalized_shape" normalized_shape ]
      ~noutputs:1
  in
  bad [ 2 ];
  bad [ 3; 2 ];
  bad [ 2; 3; 3 ];
  bad [];
  (* and the shapes that DO match still lower, with eps defaulted to 1e-5 *)
  bad [ 3 ];
  bad [ 2; 3 ];
  [%expect
    {|
    error: layer_norm: normalized_shape [2] does not match the input's trailing extents [3]
    error: layer_norm: normalized_shape [3, 2] does not match the input's trailing extents [2, 3]
    error: layer_norm: normalized_shape has 3 entries, outside [1, 2] for this rank
    error: layer_norm: normalized_shape has 0 entries, outside [1, 2] for this rank
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    tensor f32 [W=2 C=3] {-1.46385, -0.878309, -0.29277, 0.29277, 0.878309, 1.46385} |}]

(* [cudnn_enable] is accepted at BOTH values and when omitted, and all three
   produce the same tensor: ATen's own composite names it
   [bool /* cudnn_enable, deprecated */] and drops it, so the native contract is
   backend-independent by construction. It is still DECODED -- a non-boolean
   there is a malformed node, and the fourth line is what says so. *)
let%expect_test "dispatch: layer_norm accepts cudnn_enable at both values" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let run extra =
    dispatch_print ~target:"torch.ops.aten.layer_norm.default"
      ~bindings:[ ("input", x) ]
      ~inputs:([ in_tensor "input"; in_ints "normalized_shape" [ 3 ] ] @ extra)
      ~noutputs:1
  in
  run [];
  run [ in_bool "cudnn_enable" true ];
  run [ in_bool "cudnn_enable" false ];
  run [ in_int "cudnn_enable" 1 ];
  [%expect
    {|
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    error: argument "cudnn_enable": expected bool, got Int |}]

(* Both affine operands carry the whole normalized_shape, so both are rank [k].
   A rank-1 weight against a two-axis normalization right-aligns into the frame
   indistinguishably from a correct one-axis weight, which is why the rank is
   read on the ATen tensor before the bridge crosses. *)
let%expect_test "dispatch: layer_norm rejects a wrong-rank weight or bias" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let flat = float_tensor [ 3 ] [ 1.; 1.; 1. ] in
  let full = float_tensor [ 2; 3 ] [ 1.; 1.; 1.; 1.; 1.; 1. ] in
  let run name t =
    dispatch_print ~target:"torch.ops.aten.layer_norm.default"
      ~bindings:[ ("input", x); (name, t) ]
      ~inputs:
        [
          in_tensor "input"; in_ints "normalized_shape" [ 2; 3 ]; in_tensor name;
        ]
      ~noutputs:1
  in
  run "weight" flat;
  run "bias" flat;
  (* the rank-2 operands the two-axis normalization actually wants *)
  run "weight" full;
  run "bias" full;
  [%expect
    {|
    error: layer_norm weight must be rank-2, got rank-1
    error: layer_norm bias must be rank-2, got rank-1
    tensor f32 [W=2 C=3] {-1.46385, -0.878309, -0.29277, 0.29277, 0.878309, 1.46385}
    tensor f32 [W=2 C=3] {-0.463848, 0.121691, 0.707231, 1.29277, 1.87831, 2.46385} |}]

(* [eps] is added INSIDE the sqrt, so on data whose variance is comparable to it
   the two corpus values are far apart rather than indistinguishable. The
   extents are powers of two so the input, its mean and their differences are
   exact in f32 and the separation is the epsilon alone. *)
let%expect_test "dispatch: layer_norm eps defaults to 1e-5 and is inside sqrt" =
  let x = float_tensor [ 3 ] [ 0.; 0.001953125; 0.00390625 ] in
  let run extra =
    dispatch_print ~target:"torch.ops.aten.layer_norm.default"
      ~bindings:[ ("input", x) ]
      ~inputs:([ in_tensor "input"; in_ints "normalized_shape" [ 3 ] ] @ extra)
      ~noutputs:1
  in
  run [];
  run [ in_float "eps" 1e-5 ];
  run [ in_float "eps" 1e-6 ];
  [%expect
    {|
    tensor f32 [C=3] {-0.551477, 0, 0.551477}
    tensor f32 [C=3] {-0.551477, 0, 0.551477}
    tensor f32 [C=3] {-1.03762, 0, 1.03762} |}]

(* ---- the native_layer_norm.default arm ---------------------------------- *)

(* The DECOMPOSED target shares the whole body above, so what is dispatched here
   is only what differs: the 3-tuple return, the required [eps], and the absent
   [cudnn_enable].

   The node declares three outputs and the bridge builds ONE. That is legal
   rather than an arity bug -- [Verify.requires_exact_outputs] is true only for
   a dynamic [Argument.Tensors] return, so a fixed tuple falls under the
   leading-outputs rule and is verified against the first ATen result alone. The
   graph is printed to show that the missing two leave nothing behind. *)
let%expect_test "dispatch: native_layer_norm.default exposes one output" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 3 ] [ 2.; -1.; 0.5 ] in
  let b = float_tensor [ 3 ] [ 0.25; 0.; -1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.native_layer_norm.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", b) ]
    ~inputs:
      [
        in_tensor "input";
        in_ints "normalized_shape" [ 3 ];
        in_tensor "weight";
        in_tensor "bias";
        in_float "eps" 1e-6;
      ]
    ~noutputs:3;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [C=3] ->[n0], t2 f32 [C=3] ->[n0]]
    nodes:
      n0: [t3 f32 [W=2 C=3]] =
        layer_norm x=t0 weight=t1 bias=t2 params={dims=[C]; eps=1e-06}
    outputs: [t3 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {-2.19949, 0, -0.387628, -2.19949, 0, -0.387628} |}]

(* [eps] has NO schema default here, unlike the functional overload's 1e-05, so
   its absence is a malformed node. Reading it with a default would substitute a
   number the model never supplied -- and not even the one the corpus uses,
   which is 1e-06 in all 148 occurrences. *)
let%expect_test "dispatch: native_layer_norm requires eps" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let run ?(normalized = [ 3 ]) extra =
    dispatch_print ~target:"torch.ops.aten.native_layer_norm.default"
      ~bindings:[ ("input", x) ]
      ~inputs:
        ([ in_tensor "input"; in_ints "normalized_shape" normalized ] @ extra)
      ~noutputs:3
  in
  run [];
  run [ in_float "eps" 1e-6 ];
  run [ in_int "eps" 0 ];
  (* The shared validation still runs, and names THIS target rather than the
     one the body is shared with. *)
  run ~normalized:[ 2 ] [ in_float "eps" 1e-6 ];
  run ~normalized:[ 2; 3; 3 ] [ in_float "eps" 1e-6 ];
  [%expect
    {|
    error: missing required argument "eps"
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    error: argument "eps": expected float, got Int
    error: native_layer_norm: normalized_shape [2] does not match the input's trailing extents [3]
    error: native_layer_norm: normalized_shape has 3 entries, outside [1, 2] for this rank |}]

let%expect_test "dispatch: rms_norm normalized_shape=[3] with weight" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 3 ] [ 1.; 1.; 1. ] in
  dispatch_print ~target:"torch.ops.aten.rms_norm.default"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_ints "normalized_shape" [ 3 ];
        in_tensor "weight";
        in_float "eps" 1e-5;
      ]
    ~noutputs:1;
  [%expect
    {| tensor f32 [W=2 C=3] {0.46291, 0.925819, 1.38873, 0.789542, 0.986927, 1.18431} |}]

(* The GRAPH is printed here and not above, because this is where the arm used
   to materialize a ones tensor and pass it as a required operand. It no longer
   does: [Graph_ir]'s [Rms_norm] carries [weight : Tensor_ref.t option] and
   Native4D reads the option (lower.ml:293-299), so synthesizing a constant made
   this path build a structurally different graph from [Native_interp]'s for the
   same node and left that arm unreachable from the bridge.

   That is a visible behaviour change, and the values beneath it are the claim
   it must not hide: they are byte-identical to the weighted case above, whose
   weight is all ones -- which is what an absent weight means. *)
let%expect_test "dispatch: rms_norm with no weight builds no weight operand" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.rms_norm.default"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [
        in_tensor "input"; in_ints "normalized_shape" [ 3 ]; in_float "eps" 1e-5;
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] =
        rms_norm x=t0 weight=none params={dims=[C]; eps=1e-05}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {0.46291, 0.925819, 1.38873, 0.789542, 0.986927, 1.18431} |}]

(* Neither of these was refused before. The first normalized over axes whose
   extents are not the ones the model named; the second reached [trailing_axes]
   with [k > rank], where [List.filteri]'s negative lower bound keeps every
   element, and normalized over the whole tensor. *)
let%expect_test
    "dispatch: rms_norm validates normalized_shape, not just its length" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let bad normalized_shape =
    dispatch_print ~target:"torch.ops.aten.rms_norm.default"
      ~bindings:[ ("input", x) ]
      ~inputs:[ in_tensor "input"; in_ints "normalized_shape" normalized_shape ]
      ~noutputs:1
  in
  bad [ 2 ];
  bad [ 3; 2 ];
  bad [ 2; 3; 3 ];
  bad [];
  (* and the shapes that DO match still lower *)
  bad [ 3 ];
  bad [ 2; 3 ];
  [%expect
    {|
    error: rms_norm: normalized_shape [2] does not match the input's trailing extents [3]
    error: rms_norm: normalized_shape [3, 2] does not match the input's trailing extents [2, 3]
    error: rms_norm: normalized_shape has 3 entries, outside [1, 2] for this rank
    error: rms_norm: normalized_shape has 0 entries, outside [1, 2] for this rank
    tensor f32 [W=2 C=3] {0.46291, 0.92582, 1.38873, 0.789542, 0.986928, 1.18431}
    tensor f32 [W=2 C=3] {0.256776, 0.513553, 0.770329, 1.02711, 1.28388, 1.54066} |}]

let%expect_test
    "dispatch: permute.default rank-2 dims=[1,0] — transposes W and C" =
  (* ATen [2,3] right-aligns to native [W=2 C=3]; permute([1,0]) swaps dims ->
     output [W=3 C=2], i.e. the transpose [[0,3],[1,4],[2,5]] row-major. *)
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.permute.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" [ 1; 0 ] ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=2] {0, 3, 1, 4, 2, 5} |}]

let%expect_test "dispatch: permute.default rank-3 dims=[2,0,1] — CHW cycle" =
  (* ATen [2,3,4] -> native [H=2 W=3 C=4]; permute([2,0,1]) cycles dims:
     output ATen dim 0 <- input dim 2, dim 1 <- dim 0, dim 2 <- dim 1.
     Frame: output H <- input C, output W <- input H, output C <- input W.
     Output shape: H=C_in=4, W=H_in=2, C=W_in=3. *)
  let vals = List.init 24 float_of_int in
  let x = float_tensor [ 2; 3; 4 ] vals in
  dispatch_print ~target:"torch.ops.aten.permute.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" [ 2; 0; 1 ] ]
    ~noutputs:1;
  (* output[h,w,c] = input[w, c, h] (inverse of the cycle applied to indices):
     (h=0,w=0,c=0): input[0,0,0]=0; (h=0,w=0,c=1): input[0,1,0]=4 *)
  [%expect {| tensor f32 [H=4 W=2 C=3] {0, 4, 8, 12, 16, 20, 1, 5, ...} |}]

let%expect_test "dispatch: permute.default identity — output equals input" =
  let x = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.permute.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" [ 0; 1 ] ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=4] {0, 1, 2, 3, 4, 5, 6, 7, ...} |}]

(* Same hole as unbind.int, on the other arm that resolves a decoded dim: before
   commit 0, [dims.(1) = 5] escaped as an uncaught [Invalid_argument] rather
   than the typed row. *)
let%expect_test "dispatch: permute.default rejects an out-of-range dim" =
  let x = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.permute.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" [ 0; 5 ] ]
    ~noutputs:1;
  [%expect {| error: permute.default: invalid dimension 5 for rank 2 |}]

(* A [dims] list whose length disagrees with the operand's rank is a distinct
   fault from any single entry being out of range. *)
let%expect_test "dispatch: permute.default rejects a wrong-length dims list" =
  let x = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.permute.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" [ 0 ] ]
    ~noutputs:1;
  [%expect {| error: permute.default: expected 2 dims, got 1 |}]

(* ---- transpose.int --------------------------------------------------------- *)

(* Coordinate-coded values (v = 100*i + j, generalized per rank below) so a
   wrong swap fails on the printed VALUES, not merely on the output shape --
   the mutation table's "wrong pair swapped" case needs asymmetric extents for
   the same reason. *)
let%expect_test "dispatch: transpose.int rank-2 (0,1)" =
  (* [2,3], v[i,j] = 100*i+j. transpose(0,1) -> [3,2], out[j,i] = in[i,j]. *)
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 100.; 101.; 102. ] in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 0; in_int "dim1" 1 ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=2] {0, 100, 1, 101, 2, 102} |}]

let%expect_test "dispatch: transpose.int rank-3 (1,2)" =
  (* [2,3,4], v[i,j,k] = 100*i+10*j+k. transpose(1,2) swaps the middle two
     axes: out[i,k,j] = in[i,j,k], shape [2,4,3]. *)
  let vals =
    List.concat_map
      (fun i ->
        List.concat_map
          (fun j -> List.init 4 (fun k -> (100 * i) + (10 * j) + k))
          [ 0; 1; 2 ])
      [ 0; 1 ]
  in
  let x = float_tensor [ 2; 3; 4 ] (List.map float_of_int vals) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 1; in_int "dim1" 2 ]
    ~noutputs:1;
  [%expect
    {|
    tensor f32 [H=2 W=4 C=3] {0, 10, 20, 1, 11, 21, 2, 12, ...} |}]

let%expect_test "dispatch: transpose.int rank-4 negative dims (-1,-2)" =
  (* [1,2,3,4] (D,H,W,C), asymmetric extents 3x4 on the swapped axes, v =
     100*h + w. dims -1,-2 normalize to rank-1=3 (C) and rank-2=2 (W), the
     same pair positive dims 2,3 would name. *)
  let vals =
    List.concat_map (fun h -> List.init 4 (fun w -> (100 * h) + w)) [ 0; 1; 2 ]
  in
  let x = float_tensor [ 1; 2; 3; 4 ] (List.map float_of_int vals) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" (-1); in_int "dim1" (-2) ]
    ~noutputs:1;
  [%expect
    {|
    tensor f32 [H=2 W=4 C=3] {0, 100, 200, 1, 101, 201, 2, 102, ...} |}]

let%expect_test "dispatch: transpose.int rank-4 mixed dims (0,-1)" =
  let x = float_tensor [ 2; 3; 4; 5 ] (List.init 120 float_of_int) in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 0; in_int "dim1" (-1) ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [D=2 H=3 W=4 C=5] ->[n0]]
    nodes:
      n0: [t1 f32 [D=5 H=3 W=4 C=2]] = permute x=t0 perm=[D<-C, C<-D]
    outputs: [t1 f32 [D=5 H=3 W=4 C=2] <-n0]
    tensor f32 [D=5 H=3 W=4 C=2] {0, 60, 5, 65, 10, 70, 15, 75, ...} |}]

(* Equal dims: a real identity transpose, not special-cased away. *)
let%expect_test "dispatch: transpose.int equal dims is the identity" =
  let x = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 1; in_int "dim1" 1 ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=4] {0, 1, 2, 3, 4, 5, 6, 7, ...} |}]

(* Duplicate dims after normalization -- (1, -3) on a rank-4 tensor both name
   axis 1 -- is the same case as literally-equal dims: a well-defined identity
   swap, not a rejection. *)
let%expect_test
    "dispatch: transpose.int duplicate dims after normalization is the identity"
    =
  let x = float_tensor [ 1; 3; 4; 5 ] (List.init 60 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 1; in_int "dim1" (-3) ]
    ~noutputs:1;
  [%expect {| tensor f32 [H=3 W=4 C=5] {0, 1, 2, 3, 4, 5, 6, 7, ...} |}]

let%expect_test "dispatch: transpose.int rejects an out-of-range dim" =
  let x = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 0; in_int "dim1" 5 ]
    ~noutputs:1;
  [%expect {| error: transpose.int: invalid dimension 5 for rank 2 |}]

(* Rank 6, the frame's full width: transpose the outermost pair. *)
let%expect_test "dispatch: transpose.int rank-6 (0,1)" =
  let x = float_tensor [ 2; 3; 1; 1; 1; 1 ] (List.init 6 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 0; in_int "dim1" 1 ]
    ~noutputs:1;
  [%expect {| tensor f32 [N=3 T=2 D=1 H=1 W=1 C=1] {0, 3, 1, 4, 2, 5} |}]
