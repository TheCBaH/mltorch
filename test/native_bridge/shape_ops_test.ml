(* SymInt-spelled arguments, and pad/slice/select/stack/unsqueeze against ATen. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/shape_ops_test.ml]. *)

open Helpers

(* ---- SymInt-spelled arguments on the ATen path -------------------------- *)

(* A schema [SymInt] slot crosses as [Argument.Int] in every graph in the
   corpus, but the export schema also admits [Argument.Sym_int], which carries
   EITHER a resolved value ([as_int]) or an unresolved symbol ([as_name]). No
   decoder in the tree accepted that carrier at all until now, so both spellings
   were refused as the wrong kind and neither the acceptance nor the refusal had
   a stated rule.

   The rule, applied by every SymInt-shaped decoder in [Interp_decode]: a
   resolved value decodes as the int it is; a symbol is [`Unresolved_sym_arg],
   its own row rather than a wrong-kind, because the argument IS the kind the
   schema declares and what is missing is a shape environment to evaluate it in.
   Same distinction [Native_interp] already draws for a symbolic tensor
   dimension.

   [select.int]'s [SymInt index] covers the scalar decoder and [view.default]'s
   [SymInt[] size] the list one — the two shapes slice.Tensor's bounds and
   pad.default's pad list will arrive in.

   [Op_bridge] inherits the rule rather than restating it: its [int_arg] /
   [ints_arg] are [decode_result] wrappers over the [Interp_decode] ones
   (op_bridge.ml:169-174), so the two importers that share a decoder cannot
   disagree about a spelling. [Native_interp], which has its own hand-written
   decoders, is the third and is NOT covered here.

   [select.int] now has an [Op_bridge] arm (a single [Select] node, see the
   "aten.select.int" section below), so "aten and native agree" here means what
   it usually means: a native kernel ran and matched ATen, not merely that the
   dispatch succeeded with nothing to compare. *)

let%expect_test "sym_int: a resolved SymInt argument decodes as its value" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  (* index spelled as_int and as_sym_int(as_int) must reach the same ATen call. *)
  verify_print ~target:"torch.ops.aten.select.int"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 0; in_int "index" 1 ];
  verify_print ~target:"torch.ops.aten.select.int"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 0; in_sym_int "index" 1 ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "sym_int: an unresolved SymInt argument is refused by name" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  verify_print ~target:"torch.ops.aten.select.int"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 0; in_sym_name "index" "s3" ];
  [%expect
    {| dispatch error: argument "index": unresolved symbolic value "s3" |}]

let%expect_test "sym_ints: a resolved SymInt[] decodes elementwise" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  verify_print ~target:"torch.ops.aten.view.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_sym_ints "size" [ 3; 2 ] ];
  [%expect {| aten and native agree |}]

let%expect_test "sym_ints: one unresolved entry refuses the whole list" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  verify_print ~target:"torch.ops.aten.view.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_sym_ints_named "size" [ `I 3; `N "s0" ] ];
  [%expect
    {| dispatch error: argument "size": unresolved symbolic value "s0" |}]

(* ---- aten.pad.default: ATen as the oracle ------------------------------- *)

(* [verify_print] runs the node through real ATen (Interp_dispatch) AND through
   Op_bridge + Eval_direct, then compares element-wise. Silence means agreement,
   so this is the independent oracle the Direct-vs-Symbolic walk cannot be: both
   sides of that walk instantiate the same Compute functor.

   It is what pins the two things the native side derives rather than reads: the
   pad list's innermost-first ORDER, and the reflect mirror. A reversal or a
   replicate-style clamp changes values that ATen does not, so it shows up here
   as a numeric disagreement rather than as a shape mismatch. *)

let pad_verify ~sizes ~pad ?mode ?value () =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  verify_print ~target:"torch.ops.aten.pad.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      ([ in_tensor "self"; in_ints "pad" pad ]
      @ (match mode with None -> [] | Some m -> [ in_string "mode" m ])
      @ match value with None -> [] | Some v -> [ in_float "value" v ])

let%expect_test "verify: pad constant, one and two pairs" =
  (* Asymmetric amounts on both axes, and DIFFERENT amounts per axis: a
     reversal of the pair list swaps H's amounts with W's, which changes the
     output shape as well as the values. *)
  pad_verify ~sizes:[ 2; 3 ] ~pad:[ 1; 2 ] ();
  pad_verify ~sizes:[ 2; 3 ] ~pad:[ 1; 2; 3; 0 ] ();
  pad_verify ~sizes:[ 2; 3; 4 ] ~pad:[ 1; 0; 0; 2 ] ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: pad constant fill, present and absent" =
  pad_verify ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ~value:0.5 ();
  pad_verify ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ~value:(-7.25) ();
  (* Absent means 0.0, which the source values (1..6) never take, so a dropped
     fill would be visible rather than accidentally right. *)
  pad_verify ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: pad reflect at every boundary" =
  (* rank 2 with one pair: ATen's reflection_pad1d. *)
  pad_verify ~sizes:[ 2; 5 ] ~pad:[ 2; 2 ] ~mode:"reflect" ();
  pad_verify ~sizes:[ 2; 5 ] ~pad:[ 3; 0 ] ~mode:"reflect" ();
  pad_verify ~sizes:[ 2; 5 ] ~pad:[ 0; 3 ] ~mode:"reflect" ();
  (* rank 3 with two pairs: reflection_pad2d, both spatial axes mirrored. *)
  pad_verify ~sizes:[ 2; 4; 5 ] ~pad:[ 1; 2; 2; 1 ] ~mode:"reflect" ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: negative pads crop, and mix with padding" =
  pad_verify ~sizes:[ 3; 5 ] ~pad:[ -1; -1 ] ();
  pad_verify ~sizes:[ 3; 5 ] ~pad:[ -2; 0 ] ();
  pad_verify ~sizes:[ 3; 5 ] ~pad:[ -1; 0; 2; 2 ] ();
  pad_verify ~sizes:[ 3; 5 ] ~pad:[ 2; 2; -1; 0 ] ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

(* The refusals, through [dispatch_print] rather than [verify_print]: these are
   nodes the native side declines, so there is nothing to compare against. Each
   asserts the typed row rather than an exception. *)
let pad_dispatch ~sizes ~pad ?mode ?value () =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  dispatch_print ~target:"torch.ops.aten.pad.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      ([ in_tensor "self"; in_ints "pad" pad ]
      @ (match mode with None -> [] | Some m -> [ in_string "mode" m ])
      @ match value with None -> [] | Some v -> [ in_float "value" v ])
    ~noutputs:1

let%expect_test "dispatch: pad.default refusals carry a typed row" =
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 1; 2; 3 ] ();
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 1; 1; 1; 1; 1; 1 ] ();
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ~mode:"replicate" ();
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ~mode:"circular" ();
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 3; 0 ] ~mode:"reflect" ();
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ~mode:"reflect" ~value:1.5 ();
  (* A crop that consumes the axis: ATen returns a size-0 tensor, which the
     engine has no representation for. Refused by the SHAPE rule, so the row is
     a Native shape error and not an importer one. *)
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ -2; -2 ] ();
  [%expect
    {|
    error: pad list has 3 entries; a pair per padded dimension is required
    error: pad list covers 3 dimensions of a rank-2 input
    error: pad mode "replicate" is outside the Native domain (constant and reflect are supported)
    error: pad mode "circular" is outside the Native domain (constant and reflect are supported)
    error: reflect pad of axis C by (3, 0) needs each side below the extent 3
    error: pad mode "reflect" takes no non-zero value argument
    error: pad of axis C by (-2, -2) over extent 3 leaves -1 elements; the engine has no empty extent |}]

(* ---- aten.slice.Tensor: ATen as the oracle ------------------------------ *)

(* The independent oracle for row 6.2. What it pins that nothing else can: the
   bound RESOLUTION -- defaulting, negative normalization, clamping -- against
   ATen's own, on the same tensor. test/native/aten_shape_test.ml checks
   [resolve_slice] against hand values; this checks that the values it was
   given are the ones ATen would have used. A rule that is self-consistently
   wrong passes the first and fails here. *)
let slice_verify ~sizes ?dim ?start ?stop ?step () =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  verify_print ~target:"torch.ops.aten.slice.Tensor"
    ~bindings:[ ("self", x) ]
    ~inputs:
      ([ in_tensor "self" ]
      @ (match dim with None -> [] | Some d -> [ in_int "dim" d ])
      @ (match start with None -> [] | Some s -> [ s ])
      @ (match stop with None -> [] | Some s -> [ s ])
      @ match step with None -> [] | Some s -> [ in_int "step" s ])

let%expect_test "verify: slice bounds, absent and explicit" =
  (* Every argument at its schema default: the identity slice on dim 0. Kept
     because it is the configuration a generated Default-tier walk would draw,
     and it has to be RIGHT even though it proves nothing on its own. *)
  slice_verify ~sizes:[ 4; 5 ] ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" 1) ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~stop:(in_int "end" 3) ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" 1)
    ~stop:(in_int "end" 4) ();
  (* An explicit null is a different ARGUMENT from an absent one, and both have
     to mean the schema default. *)
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_none "start")
    ~stop:(in_none "end") ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: slice negative bounds and clamping" =
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" (-2)) ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~stop:(in_int "end" (-1)) ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" (-4))
    ~stop:(in_int "end" (-1)) ();
  (* Both clamps. ATen clamps rather than rejecting, so a node ATen runs has to
     lower -- these would be refusals if the native side had chosen to reject
     out-of-range bounds instead of narrowing them. *)
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~stop:(in_int "end" 99) ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" (-99)) ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: slice step, exact and inexact division" =
  slice_verify ~sizes:[ 4; 6 ] ~dim:1 ~step:2 ();
  slice_verify ~sizes:[ 4; 6 ] ~dim:1 ~step:3 ();
  (* Span 5 over step 2 and span 5 over step 3: the ceiling is what makes these
     3 and 2 rather than 2 and 1, and ATen is the authority on which. *)
  slice_verify ~sizes:[ 4; 6 ] ~dim:1 ~start:(in_int "start" 1) ~step:2 ();
  slice_verify ~sizes:[ 4; 6 ] ~dim:1 ~start:(in_int "start" 1) ~step:3 ();
  slice_verify ~sizes:[ 4; 6 ] ~dim:1 ~start:(in_int "start" 1)
    ~stop:(in_int "end" 4) ~step:2 ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: slice dim spellings name the same axis" =
  slice_verify ~sizes:[ 4; 5; 6 ] ~dim:0 ~stop:(in_int "end" 2) ();
  slice_verify ~sizes:[ 4; 5; 6 ] ~dim:(-3) ~stop:(in_int "end" 2) ();
  slice_verify ~sizes:[ 4; 5; 6 ] ~dim:2 ~stop:(in_int "end" 2) ();
  slice_verify ~sizes:[ 4; 5; 6 ] ~dim:(-1) ~stop:(in_int "end" 2) ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

(* op6-impl decision 3, on the BRIDGE decoder. The same pair is asserted on
   [Interp_decode] (test/native_bridge/shape_ops_test.ml's sym_int rows above, through
   [select.int]) and on [Native_interp] (test/native_interp/slice_test.ml).
   Three separate code paths, one rule -- which is the point, since before this
   they agreed only by all three refusing every [Sym_int] alike. *)
let%expect_test "verify: slice bounds spelled as resolved SymInt" =
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_sym_int "start" 1)
    ~stop:(in_sym_int "end" 4) ();
  [%expect {| aten and native agree |}]

let slice_dispatch ~sizes ?dim ?start ?stop ?step () =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  dispatch_print ~target:"torch.ops.aten.slice.Tensor"
    ~bindings:[ ("self", x) ]
    ~inputs:
      ([ in_tensor "self" ]
      @ (match dim with None -> [] | Some d -> [ in_int "dim" d ])
      @ (match start with None -> [] | Some s -> [ s ])
      @ (match stop with None -> [] | Some s -> [ s ])
      @ match step with None -> [] | Some s -> [ in_int "step" s ])
    ~noutputs:1

let%expect_test "dispatch: slice.Tensor refusals carry a typed row" =
  (* Empty: ATen returns a size-0 tensor and the engine has no representation
     for one. Refused by the SHAPE rule, so the row is a Native shape error
     rather than an importer one -- the same layering pad's crop follows. *)
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" 2)
    ~stop:(in_int "end" 2) ();
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" 99) ();
  (* A non-positive step. ATen refuses it too, and the row comes from
     [Aten_shape.resolve_slice] rather than from the op. *)
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:1 ~step:0 ();
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:1 ~step:(-1) ();
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:2 ();
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_sym_name "start" "s4") ();
  [%expect
    {|
    error: slice of axis C [2, 2) step 1 over extent 5 selects 0 elements; the engine has no empty extent
    error: slice of axis C [5, 5) step 1 over extent 5 selects 0 elements; the engine has no empty extent
    error: slice step must be >= 1, got 0
    error: slice step must be >= 1, got -1
    error: slice.Tensor: invalid dimension 2 for rank 2
    error: argument "start": unresolved symbolic value "s4" |}]

(* ---- aten.select.int: ATen as the oracle --------------------------------- *)

(* One [Select] node -- real ATen is what proves it computes the
   RIGHT tensor, not merely one of the right shape. *)
let select_verify ~sizes ~dim ~index =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  verify_print ~target:"torch.ops.aten.select.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" dim; in_int "index" index ]

let%expect_test "verify: select.int on each axis of a rank-3 tensor" =
  select_verify ~sizes:[ 3; 4; 5 ] ~dim:0 ~index:1;
  select_verify ~sizes:[ 3; 4; 5 ] ~dim:1 ~index:2;
  select_verify ~sizes:[ 3; 4; 5 ] ~dim:2 ~index:4;
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: select.int negative dim and negative index" =
  select_verify ~sizes:[ 3; 4; 5 ] ~dim:(-1) ~index:4;
  select_verify ~sizes:[ 3; 4; 5 ] ~dim:1 ~index:(-1);
  select_verify ~sizes:[ 3; 4; 5 ] ~dim:(-3) ~index:(-3);
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: select.int on a rank-1 tensor collapses to rank 0" =
  select_verify ~sizes:[ 5 ] ~dim:0 ~index:2;
  [%expect {| aten and native agree |}]

let select_dispatch ~sizes ~dim ~index =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  dispatch_print ~target:"torch.ops.aten.select.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" dim; in_int "index" index ]
    ~noutputs:1

let%expect_test "dispatch: select.int rejects an out-of-range index" =
  (* Unlike slice.Tensor, ATen does not clamp an out-of-range index: 5 and -6
     are both refused on a size-5 axis. *)
  select_dispatch ~sizes:[ 3; 5 ] ~dim:1 ~index:5;
  select_dispatch ~sizes:[ 3; 5 ] ~dim:1 ~index:(-6);
  select_dispatch ~sizes:[ 3; 5 ] ~dim:2 ~index:0;
  [%expect
    {|
    error: index 5 is out of bounds for dimension with size 5
    error: index -6 is out of bounds for dimension with size 5
    error: select.int: invalid dimension 2 for rank 2 |}]

(* One [Select] node, not a [Slice]+[Reshape] pair (the design-goal fix) --
   the graph shape is the fact this test pins, not just the value
   [select_verify]'s ATen comparison already covers. *)
let%expect_test "dispatch: select.int builds a single Select node" =
  let x =
    float_tensor [ 3; 4 ] (List.init 12 (fun i -> float_of_int (i + 1)))
  in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.select.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 1; in_int "index" 2 ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [C=3]] = select x=t0 params={axis=C index=2}
    outputs: [t1 f32 [C=3] <-n0]
    tensor f32 [C=3] {3, 7, 11} |}]

(* ---- aten.select_scatter.default: ATen as the oracle ---------------------- *)

(* One [Select_scatter] node -- real ATen is what proves it computes the
   RIGHT tensor (self elsewhere, src at the written position), not merely one
   of the right shape. [src] is filled from a disjoint range (100+) so a value
   read from the wrong operand is legible rather than merely unequal. *)
let select_scatter_verify ~self_sizes ~src_sizes ~dim ~index =
  let self_n = List.fold_left ( * ) 1 self_sizes in
  let src_n = List.fold_left ( * ) 1 src_sizes in
  let self =
    float_tensor self_sizes (List.init self_n (fun i -> float_of_int (i + 1)))
  in
  let src =
    float_tensor src_sizes
      (List.init src_n (fun i -> float_of_int (100 + i + 1)))
  in
  verify_print ~target:"torch.ops.aten.select_scatter.default"
    ~bindings:[ ("self", self); ("src", src) ]
    ~inputs:
      [
        in_tensor "self";
        in_tensor "src";
        in_int "dim" dim;
        in_int "index" index;
      ]

let%expect_test "verify: select_scatter on each axis of a rank-3 tensor" =
  select_scatter_verify ~self_sizes:[ 3; 4; 5 ] ~src_sizes:[ 4; 5 ] ~dim:0
    ~index:1;
  select_scatter_verify ~self_sizes:[ 3; 4; 5 ] ~src_sizes:[ 3; 5 ] ~dim:1
    ~index:2;
  select_scatter_verify ~self_sizes:[ 3; 4; 5 ] ~src_sizes:[ 3; 4 ] ~dim:2
    ~index:4;
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: select_scatter negative dim and negative index" =
  select_scatter_verify ~self_sizes:[ 3; 4; 5 ] ~src_sizes:[ 3; 4 ] ~dim:(-1)
    ~index:4;
  select_scatter_verify ~self_sizes:[ 3; 4; 5 ] ~src_sizes:[ 3; 5 ] ~dim:1
    ~index:(-1);
  select_scatter_verify ~self_sizes:[ 3; 4; 5 ] ~src_sizes:[ 4; 5 ] ~dim:(-3)
    ~index:(-3);
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let select_scatter_dispatch ~self_sizes ~src_sizes ~dim ~index =
  let self_n = List.fold_left ( * ) 1 self_sizes in
  let src_n = List.fold_left ( * ) 1 src_sizes in
  let self =
    float_tensor self_sizes (List.init self_n (fun i -> float_of_int (i + 1)))
  in
  let src =
    float_tensor src_sizes
      (List.init src_n (fun i -> float_of_int (100 + i + 1)))
  in
  dispatch_print ~target:"torch.ops.aten.select_scatter.default"
    ~bindings:[ ("self", self); ("src", src) ]
    ~inputs:
      [
        in_tensor "self";
        in_tensor "src";
        in_int "dim" dim;
        in_int "index" index;
      ]
    ~noutputs:1

let%expect_test "dispatch: select_scatter rejects an out-of-range index" =
  (* Same discipline as select.int: ATen does not clamp an out-of-range
     index. *)
  select_scatter_dispatch ~self_sizes:[ 3; 5 ] ~src_sizes:[ 3 ] ~dim:1 ~index:5;
  select_scatter_dispatch ~self_sizes:[ 3; 5 ] ~src_sizes:[ 3 ] ~dim:1
    ~index:(-6);
  select_scatter_dispatch ~self_sizes:[ 3; 5 ] ~src_sizes:[ 5 ] ~dim:2 ~index:0;
  [%expect
    {|
    error: index 5 is out of bounds for dimension with size 5
    error: index -6 is out of bounds for dimension with size 5
    error: select_scatter.default: invalid dimension 2 for rank 2 |}]

(* [src]'s shape is untrusted model data, not re-derived from [self]/[axis]/
   [index] -- this is the NATIVE shape check ([Select_scatter.output_shape]),
   distinct from the ATen-level index rejection above. *)
let%expect_test "dispatch: select_scatter rejects a src of the wrong shape" =
  select_scatter_dispatch ~self_sizes:[ 3; 5 ] ~src_sizes:[ 4 ] ~dim:1 ~index:2;
  [%expect
    {| error: select_scatter src shape must be [C=3] (axis=C index=2), got [C=4] |}]

(* One [Select_scatter] node, not a multi-node stand-in built from other
   ops -- the graph shape is the fact this test pins, not just the value
   [select_scatter_verify]'s ATen comparison already covers. *)
let%expect_test "dispatch: select_scatter builds a single Select_scatter node" =
  let self =
    float_tensor [ 3; 4 ] (List.init 12 (fun i -> float_of_int (i + 1)))
  in
  let src = float_tensor [ 4 ] [ 900.; 901.; 902.; 903. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.select_scatter.default"
    ~bindings:[ ("self", self); ("src", src) ]
    ~inputs:
      [ in_tensor "self"; in_tensor "src"; in_int "dim" 0; in_int "index" 1 ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=3 C=4] ->[n0], t1 f32 [C=4] ->[n0]]
    nodes:
      n0: [t2 f32 [W=3 C=4]] =
        select_scatter self=t0 src=t1 params={axis=W index=1}
    outputs: [t2 f32 [W=3 C=4] <-n0]
    tensor f32 [W=3 C=4] {1, 2, 3, 4, 900, 901, 902, 903, ...} |}]

(* ---- aten.stack.default: ATen as the oracle ------------------------------ *)

let in_tensors name names =
  PT.NamedArgument.make name
    (PT.Argument.Tensors (List.map PT.TensorArgument.make names))
    None

(* One [Stack] node, not N [Reshape]s plus [Concat] (the design-goal fix) --
   real ATen is what proves it computes the RIGHT tensor, not merely
   one of the right shape. *)
let stack_verify ~sizes ~n ~dim =
  let elems = List.fold_left ( * ) 1 sizes in
  let xs =
    List.init n (fun k ->
        float_tensor sizes
          (List.init elems (fun i -> float_of_int ((k * 1000) + i + 1))))
  in
  let names = List.mapi (fun i _ -> Printf.sprintf "x%d" i) xs in
  verify_print ~target:"torch.ops.aten.stack.default"
    ~bindings:(List.combine names xs)
    ~inputs:[ in_tensors "tensors" names; in_int "dim" dim ]

let%expect_test "verify: stack.default at the outermost and an inner dim" =
  stack_verify ~sizes:[ 3; 4 ] ~n:2 ~dim:0;
  (* The non-outermost, non-innermost case: exactly what caught the
     shape/coordinate direction bug during development -- [dim]
     relabels which native axis carries each operand's real extent, rather
     than leaving it where the operand's own storage has it. *)
  stack_verify ~sizes:[ 3; 4 ] ~n:2 ~dim:1;
  stack_verify ~sizes:[ 3; 4 ] ~n:3 ~dim:2;
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: stack.default negative dim" =
  stack_verify ~sizes:[ 3; 4 ] ~n:2 ~dim:(-1);
  [%expect {| aten and native agree |}]

let%expect_test "dispatch: stack.default builds a single Stack node" =
  let a = float_tensor [ 3 ] [ 1.; 2.; 3. ] in
  let b = float_tensor [ 3 ] [ 10.; 20.; 30. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.stack.default"
    ~bindings:[ ("a", a); ("b", b) ]
    ~inputs:[ in_tensors "tensors" [ "a"; "b" ]; in_int "dim" 1 ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [C=3] ->[n0], t1 f32 [C=3] ->[n0]]
    nodes:
      n0: [t2 f32 [W=3 C=2]] = stack xs=[t0, t1] params={axis=C}
    outputs: [t2 f32 [W=3 C=2] <-n0]
    tensor f32 [W=3 C=2] {1, 10, 2, 20, 3, 30} |}]

let%expect_test "dispatch: stack.default refusals carry a typed row" =
  let a = float_tensor [ 3 ] [ 1.; 2.; 3. ] in
  let b =
    float_tensor [ 3; 4 ] (List.init 12 (fun i -> float_of_int (i + 1)))
  in
  dispatch_print ~target:"torch.ops.aten.stack.default" ~bindings:[]
    ~inputs:[ in_tensors "tensors" []; in_int "dim" 0 ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.stack.default"
    ~bindings:[ ("a", a); ("b", b) ]
    ~inputs:[ in_tensors "tensors" [ "a"; "b" ]; in_int "dim" 0 ]
    ~noutputs:1;
  [%expect
    {|
    error: stack.default: at least one tensor is required
    error: stack.default: every tensor must have the same rank: 1 vs 2 |}]

(* ---- aten.unsqueeze.default: ATen as the oracle -------------------------- *)

(* Legalized to [Reshape] alone: inserting a size-1 axis never changes the
   linearized data order, unlike [select.int]'s axis removal, so no [Slice]
   is needed. *)
let unsqueeze_verify ~sizes ~dim =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  verify_print ~target:"torch.ops.aten.unsqueeze.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" dim ]

let%expect_test "verify: unsqueeze.default at every insertion position" =
  unsqueeze_verify ~sizes:[ 3; 4 ] ~dim:0;
  unsqueeze_verify ~sizes:[ 3; 4 ] ~dim:1;
  unsqueeze_verify ~sizes:[ 3; 4 ] ~dim:2;
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: unsqueeze.default negative dim" =
  unsqueeze_verify ~sizes:[ 3; 4 ] ~dim:(-1);
  unsqueeze_verify ~sizes:[ 3; 4 ] ~dim:(-3);
  [%expect {|
    aten and native agree
    aten and native agree |}]

let unsqueeze_dispatch ~sizes ~dim =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  dispatch_print ~target:"torch.ops.aten.unsqueeze.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" dim ]
    ~noutputs:1

let%expect_test "dispatch: unsqueeze.default rejects a dim outside [-(r+1), r]"
    =
  unsqueeze_dispatch ~sizes:[ 3; 4 ] ~dim:3;
  unsqueeze_dispatch ~sizes:[ 3; 4 ] ~dim:(-4);
  [%expect
    {|
    error: unsqueeze.default: invalid dimension 3 for rank 2
    error: unsqueeze.default: invalid dimension -4 for rank 2 |}]

let%expect_test "dispatch: unsqueeze.default rejects pushing rank past six" =
  let x = float_tensor [ 1; 1; 2; 2; 2; 2 ] (List.init 16 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.unsqueeze.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 0 ]
    ~noutputs:1;
  [%expect {| error: rank 7 out of [0, 6] |}]

(* ---- aten.alias.default: ATen as the oracle ------------------------------ *)

(* [alias(self) -> Tensor]: a pure identity view, ATen's own schema takes no
   argument beyond [self]. Legalized to the *existing* [Clone] node --
   paramless, shape-preserving, and already this repo's "same value, no
   change" node (see [clone.default]'s own arm) -- rather than a same-shape
   [Reshape], to keep the printed graph's node name matching what actually
   happened: nothing was reshaped. *)
let alias_verify ~sizes =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  verify_print ~target:"torch.ops.aten.alias.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self" ]

let%expect_test "verify: alias.default at several ranks" =
  alias_verify ~sizes:[ 3; 4 ];
  alias_verify ~sizes:[ 2; 3; 4 ];
  alias_verify ~sizes:[ 5 ];
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "dispatch: alias.default builds a single Clone node" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.alias.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = clone x=t0
    outputs: [t1 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {1, 2, 3, 4, 5, 6} |}]

(* ---- aten._assert_tensor_metadata.default: routed to Discard -------------- *)

(* [_assert_tensor_metadata(Tensor a, SymInt[]? size=None, SymInt[]?
   stride=None, ScalarType? dtype=None, *, Device? device=None, Layout?
   layout=None) -> ()]: ATen's own schema returns no [Tensor] at all, so there
   is no ATen output to verify against ([verify_print] hardcodes exactly one
   node output and has no counterpart here) -- the only meaningful check is
   that the bridge routes [a] into the same [Discard] sink
   [max_pool2d_with_indices.default]'s dead indices edge uses, and that the
   built graph has zero outputs, matching the node's own empty serialized
   [outputs] list. *)
(* ---- aten.unfold.default: ATen as the oracle ----------------------------- *)

(* [dimension] is read against [self]'s OWN rank, same as [select.int]'s
   [dim] above; [Unfold.Unfold.dest_of] then maps it onto the axis the
   window COUNT lands on in the OUTPUT (op_bridge_shape.ml's own comment). *)
let unfold_verify ~sizes ~dimension ~size ~step =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  verify_print ~target:"torch.ops.aten.unfold.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_int "dimension" dimension;
        in_int "size" size;
        in_int "step" step;
      ]

let%expect_test "verify: unfold.default overlapping and non-overlapping windows"
    =
  (* A plain vector: the appended window axis is the only "extra" one, so
     this exercises the simplest [source_of]/[dest_of] case (dim 0 of a
     rank-1 tensor). *)
  unfold_verify ~sizes:[ 8 ] ~dimension:0 ~size:3 ~step:2;
  unfold_verify ~sizes:[ 8 ] ~dimension:0 ~size:2 ~step:2;
  (* The corpus's own HaloAttn shape (eca_halonext26ts): unfolding each
     spatial axis of a rank-4 [N,C,H,W] tensor in turn. *)
  unfold_verify ~sizes:[ 1; 8; 20; 20 ] ~dimension:2 ~size:12 ~step:8;
  unfold_verify ~sizes:[ 1; 8; 20; 20 ] ~dimension:3 ~size:12 ~step:8;
  (* A rank-5 input, as the corpus's own SECOND (chained) unfold call sees:
     the output is rank 6, using every frame axis -- the demanding case
     [output_shape]'s N-must-be-unit precondition exists to gate. *)
  unfold_verify ~sizes:[ 1; 4; 2; 10; 6 ] ~dimension:3 ~size:3 ~step:2;
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: unfold.default negative dimension" =
  unfold_verify ~sizes:[ 3; 8 ] ~dimension:(-1) ~size:3 ~step:2;
  [%expect {| aten and native agree |}]

let%expect_test
    "dispatch: _assert_tensor_metadata.default builds a single Discard node" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten._assert_tensor_metadata.default"
    ~bindings:[ ("a", x) ]
    ~inputs:[ in_tensor "a" ]
    ~noutputs:0;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [] = discard x=t0
    outputs: [] |}]
