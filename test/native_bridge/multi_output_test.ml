(* multi-output (Tensor[]) dispatch: unbind, split_with_sizes. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/multi_output_test.ml]. *)

open Helpers

(* --- Interp_decode.output_names ---

   Five call sites used to open-code a [Argument.Tensor]-only filter, which
   yields ZERO names for a Tensor[]-returning node -- a report with no output
   lines reads as "nothing to say" rather than as the omission it is. This pins
   the helper's own behaviour (flattening order, None handling) independently of
   any CLI formatting that consumes it. *)

let names_of outputs =
  let node = PT.Node.make "t" [] outputs Sm.empty None (Some "test") in
  print_endline (String.concat "," (Interp_decode.output_names node))

let%expect_test "output_names: flattens Tensor[] in place, skips None" =
  (* single tensor: one name *)
  names_of [ targ "a" ];
  [%expect "a"];
  (* fixed tuple: N separate outputs, in order *)
  names_of [ targ "a"; targ "b"; targ "c" ];
  [%expect "a,b,c"];
  (* Tensor[]: ONE output carrying N names -- the shape unbind serializes as *)
  names_of
    [
      PT.Argument.Tensors
        [
          PT.TensorArgument.make "x";
          PT.TensorArgument.make "y";
          PT.TensorArgument.make "z";
        ];
    ];
  [%expect "x,y,z"];
  (* a dead output stays skipped, and flattening happens in position *)
  names_of
    [
      targ "a";
      PT.Argument.None false;
      PT.Argument.Tensors
        [ PT.TensorArgument.make "x"; PT.TensorArgument.make "y" ];
      targ "b";
    ];
  [%expect "a,x,y,b"];
  (* an empty Tensor[] contributes nothing, but is not an error *)
  names_of [ PT.Argument.Tensors [] ];
  [%expect ""]

(* --- Interp handle ownership ---

   [Interp.top_predictions] and [Interp.argmax] read their results through
   [Aten_tensor.data] / [item_int], neither of which manages a handle:
   [data]'s finaliser only anchors an already-registered handle for the
   Bigarray's lifetime, it never calls atc_free. So every op result these two
   allocate has to be piped through [Aten_tensor.manage] at the call site or it
   leaks for the process's lifetime. Counting handles is the only way to see
   that -- the values are correct either way. *)

let collect () =
  Gc.full_major ();
  Gc.full_major ()

let%expect_test "top_predictions and argmax leave no live tensors" =
  collect ();
  let base = T.live_count () in
  let logits = float_tensor [ 1; 4 ] [ 0.5; 3.0; 1.0; 2.0 ] in
  (* Run in an inner scope so nothing but [logits] is still referenced when the
     collection below runs. *)
  (match
     ( Interp.top_predictions logits 2 |> Err.payload,
       Interp.argmax logits |> Err.payload )
   with
  | Ok top, Ok am ->
      Printf.printf "argmax=%d top=%s\n" am
        (String.concat ","
           (List.map (fun (i, p) -> Printf.sprintf "%d:%.3f" i p) top))
  | _ -> print_endline "unexpected error");
  collect ();
  (* [logits] is still live and still managed: +1, not +0. *)
  Printf.printf "after gc: +%d\n" (T.live_count () - base);
  ignore (Sys.opaque_identity logits);
  [%expect {|
    argmax=1 top=1:0.631,3:0.232
    after gc: +1 |}]

(* --- Interp_dispatch: the Tensor[] output shape --------------------------

   A Tensor[] return serializes as ONE output of kind Tensor[] carrying every
   result name, not as N separate outputs, so [bind_tensor_list] handles it
   instead of the positional [bind_many]. These drive the generated arm through
   hand-built nodes, because the fixture format cannot express a wrong output
   shape (Aten_spec_run synthesizes outputs, so the counts always agree). *)

let pp_dispatch_error ppf = function
  | #Interp_decode.error as e -> Interp_decode.pp_error ppf e
  | `Aten_runtime_failure (op, st) ->
      Fmt.pf ppf "ATen op %s failed with status %d" op st
  | `Unhandled_op target -> Fmt.pf ppf "unhandled op %S" target

(* Dispatch a one-node unbind graph and print each bound name's values. Reads go
   through [materialize_for_raw_read]: unbind returns views. *)
let unbind_dispatch ~inputs ~outputs ~self =
  let env = Sm.add "self" self Sm.empty in
  let node =
    PT.Node.make "torch.ops.aten.unbind.int"
      (PT.NamedArgument.make "self" (targ "self") None :: inputs)
      outputs Sm.empty None (Some "test")
  in
  match Interp_dispatch.dispatch env node |> Err.payload with
  | Error e -> Format.printf "Error: %a@." pp_dispatch_error e
  | Ok env' ->
      List.iter
        (fun name ->
          Format.printf "%s = %a@." name
            (Core.Pretty.option_or ~none:"<unbound>" (fun ppf t ->
                 Aten_tensor.pp_float32 ppf
                   (Option.get
                      (Aten_tensor.as_float32
                         (Aten_tensor.materialize_for_raw_read t)))))
            (Sm.find_opt name env'))
        (Interp_decode.output_names node)

let tensors names =
  [ PT.Argument.Tensors (List.map PT.TensorArgument.make names) ]

(* The real exported node omits [dim] entirely — the schema default 0 is applied
   by the generated decoder, so this is the shape that actually has to work. *)
let%expect_test "dispatch: unbind.int with dim absent binds every name" =
  unbind_dispatch ~inputs:[]
    ~outputs:(tensors [ "a"; "b"; "c" ])
    ~self:(float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect {|
    a = [0; 1]
    b = [2; 3]
    c = [4; 5] |}]

let%expect_test "dispatch: unbind.int with a negative dim" =
  unbind_dispatch
    ~inputs:[ in_int "dim" (-1) ]
    ~outputs:(tensors [ "a"; "b"; "c" ])
    ~self:(float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect {|
    a = [0; 3]
    b = [1; 4]
    c = [2; 5] |}]

let%expect_test "dispatch: unbind.int of a zero-length dim binds nothing" =
  unbind_dispatch
    ~inputs:[ in_int "dim" 0 ]
    ~outputs:(tensors [])
    ~self:(Aten_tensor.create [ 0; 2 ]);
  [%expect {| |}]

(* A fixed tuple's output shape must not be accepted here: zipping [Tensors]
   against separate [Tensor] outputs would leave SSA names unbound. *)
let%expect_test "dispatch: unbind.int rejects the fixed-tuple output shape" =
  unbind_dispatch ~inputs:[]
    ~outputs:[ targ "a"; targ "b"; targ "c" ]
    ~self:(float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect
    {| Error: expected one Tensor[] output, got [Tensor; Tensor; Tensor] |}]

(* The arity check rides on the pairing it guards (Err.List.map2), so it cannot
   drift from it. Drop a name and it fires with both counts. *)
let%expect_test "dispatch: unbind.int reports a name/tensor count mismatch" =
  unbind_dispatch ~inputs:[]
    ~outputs:(tensors [ "a"; "b" ])
    ~self:(float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect
    {| Error: tensor-list output arity: 2 serialized names, 3 tensors returned |}]

let%expect_test "pp: unbind.int config" =
  Aten_op_config.find "torch.ops.aten.unbind.int"
  |> Format.printf "%a@."
       (Core.Pretty.option_or ~none:"not found" Aten_op_config.pp);
  [%expect {| torch.ops.aten.unbind.int (Tensor self, Int dim=0) -> T[] |}]

(* --- the native side of unbind ------------------------------------------- *)

(* [dispatch_print] is reusable here even though it synthesises a fixed-tuple
   output shape: the bridge arm reads [self] and [dim] only, never the node's
   outputs. The values are hand-derived, so this does not lean on ATen. *)
let%expect_test "dispatch: unbind.int slices along the leading dim" =
  (* [3,2] right-aligns to [W=3 C=2]; dim 0 is W, so each slice drops W and the
     survivors re-pack to [C=2]. *)
  dispatch_print ~target:"torch.ops.aten.unbind.int"
    ~bindings:[ ("self", float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:0;
  [%expect
    {|
    tensor f32 [C=2] {0, 1}
    tensor f32 [C=2] {2, 3}
    tensor f32 [C=2] {4, 5} |}]

(* dim=-1 on a rank-2 input is C. Each slice reads a COLUMN, so the values are
   strided rather than contiguous — the case a naive flat read gets wrong. *)
let%expect_test "dispatch: unbind.int with a negative dim" =
  dispatch_print ~target:"torch.ops.aten.unbind.int"
    ~bindings:[ ("self", float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]) ]
    ~inputs:[ in_tensor "self"; in_int "dim" (-1) ]
    ~noutputs:0;
  [%expect
    {|
    tensor f32 [C=2] {0, 3}
    tensor f32 [C=2] {1, 4}
    tensor f32 [C=2] {2, 5} |}]

(* [Aten_shape.axis_of_dim] asserts its range and raises, so the arm checks
   first and reports a typed row. The ORIGINAL dim is echoed, not the
   normalised one. *)
let%expect_test "dispatch: unbind.int rejects an out-of-range dim" =
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.unbind.int"
        ~bindings:[ ("self", float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]) ]
        ~inputs:[ in_tensor "self"; in_int "dim" d ]
        ~noutputs:0)
    [ 2; -3 ];
  [%expect
    {|
    error: unbind.int: invalid dimension 2 for rank 2
    error: unbind.int: invalid dimension -3 for rank 2 |}]

(* Unbind is a view operation: it copies storage cells and keeps the source
   format, rather than taking the arithmetic f32 materialization path.  Values
   around 2^53 and the signed endpoints prove the result never went through an
   OCaml float. *)
let%expect_test "dispatch: unbind.int preserves int64 storage exactly" =
  dispatch_print ~target:"torch.ops.aten.unbind.int"
    ~bindings:
      [
        ( "self",
          i64_tensor [ 3; 2 ]
            [
              Int64.min_int;
              -1L;
              9_007_199_254_740_993L;
              9_007_199_254_740_995L;
              Int64.max_int;
              0L;
            ] );
      ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:0;
  [%expect
    {|
    tensor i64 [C=2] {-9223372036854775808, -1}
    tensor i64 [C=2] {9007199254740993, 9007199254740995}
    tensor i64 [C=2] {9223372036854775807, 0} |}]

(* The real oracle: ATen runs the op, the native side runs [Graph_ir.Unbind]
   through [Eval_direct], and [Verify.verify_node] compares EVERY slice —
   exactly, because a single [Argument.Tensors] output has no dead-output story
   and a bridge returning fewer would otherwise report a match.

   [verify_print] cannot drive this: it synthesises one [Argument.Tensor]
   output, which [bind_tensor_list] rejects. Silence is agreement. *)
let verify_unbind ~inputs ~outputs ~self =
  let env = Sm.add "self" self Sm.empty in
  let node =
    PT.Node.make "torch.ops.aten.unbind.int"
      (PT.NamedArgument.make "self" (targ "self") None :: inputs)
      outputs Sm.empty None (Some "test")
  in
  match
    Interp_verify.dispatch ~verify:true ~ppf:Format.std_formatter env node
  with
  | Error e ->
      Format.printf "dispatch error: %a@." Interp_verify.pp_interp_error
        (Err.Error.kind e)
  | Ok _ -> print_string "aten and native agree\n"

let%expect_test "verify: unbind.int agrees with ATen on every slice" =
  let self = float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  verify_unbind ~inputs:[] ~outputs:(tensors [ "a"; "b"; "c" ]) ~self;
  (* And along the strided axis, where each slice is a column. *)
  verify_unbind
    ~inputs:[ in_int "dim" (-1) ]
    ~outputs:(tensors [ "a"; "b" ])
    ~self:(float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: unbind.int agrees with ATen for int64" =
  verify_unbind ~inputs:[]
    ~outputs:(tensors [ "a"; "b"; "c" ])
    ~self:
      (i64_tensor [ 3; 2 ]
         [
           Int64.min_int;
           -1L;
           9_007_199_254_740_993L;
           9_007_199_254_740_995L;
           Int64.max_int;
           0L;
         ]);
  [%expect {| aten and native agree |}]

(* --- split_with_sizes.default ---------------------------------------------

   Same two-layer structure as unbind.int just above: the generated
   [Interp_dispatch] arm (real ATen, a [Tensor[]] output) first, then the
   [Op_bridge] arm (hand-derived Native values), then [Interp_verify] as the
   real oracle comparing the two. *)

let split_with_sizes_dispatch ~inputs ~outputs ~self =
  let env = Sm.add "self" self Sm.empty in
  let node =
    PT.Node.make "torch.ops.aten.split_with_sizes.default"
      (PT.NamedArgument.make "self" (targ "self") None :: inputs)
      outputs Sm.empty None (Some "test")
  in
  match Interp_dispatch.dispatch env node |> Err.payload with
  | Error e -> Format.printf "Error: %a@." pp_dispatch_error e
  | Ok env' ->
      List.iter
        (fun name ->
          Format.printf "%s = %a@." name
            (Core.Pretty.option_or ~none:"<unbound>" (fun ppf t ->
                 Aten_tensor.pp_float32 ppf
                   (Option.get
                      (Aten_tensor.as_float32
                         (Aten_tensor.materialize_for_raw_read t)))))
            (Sm.find_opt name env'))
        (Interp_decode.output_names node)

let%expect_test
    "dispatch: split_with_sizes.default with dim absent binds every name" =
  split_with_sizes_dispatch
    ~inputs:[ in_ints "split_sizes" [ 1; 2 ] ]
    ~outputs:(tensors [ "a"; "b" ])
    ~self:(float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect {|
    a = [0; 1]
    b = [2; 3; 4; 5] |}]

let%expect_test "pp: split_with_sizes.default config" =
  Aten_op_config.find "torch.ops.aten.split_with_sizes.default"
  |> Format.printf "%a@."
       (Core.Pretty.option_or ~none:"not found" Aten_op_config.pp);
  [%expect
    {|
    torch.ops.aten.split_with_sizes.default (Tensor self, Int[] split_sizes,
      Int dim=0) -> T[] |}]

(* --- the native side of split_with_sizes ----------------------------------- *)

(* Two DIFFERENT window sizes on the leading dim (W after right-alignment), so
   a wrong per-piece offset shows up as wrong values, not just a wrong shape. *)
let%expect_test
    "dispatch: split_with_sizes.default divides the leading dim into windows" =
  dispatch_print ~target:"torch.ops.aten.split_with_sizes.default"
    ~bindings:[ ("self", float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]) ]
    ~inputs:[ in_tensor "self"; in_ints "split_sizes" [ 1; 2 ] ]
    ~noutputs:0;
  [%expect
    {|
    tensor f32 [C=2] {0, 1}
    tensor f32 [W=2 C=2] {2, 3, 4, 5} |}]

(* dim=-1 on a rank-2 input is C, so each piece is a COLUMN RANGE rather than a
   row range -- strided, not contiguous, the case a naive flat read gets
   wrong. *)
let%expect_test "dispatch: split_with_sizes.default with a negative dim" =
  dispatch_print ~target:"torch.ops.aten.split_with_sizes.default"
    ~bindings:[ ("self", float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]) ]
    ~inputs:
      [ in_tensor "self"; in_ints "split_sizes" [ 2; 1 ]; in_int "dim" (-1) ]
    ~noutputs:0;
  [%expect
    {|
    tensor f32 [W=2 C=2] {0, 1, 3, 4}
    tensor f32 [W=2 C=1] {2, 5} |}]

let%expect_test "dispatch: split_with_sizes.default rejects an out-of-range dim"
    =
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.split_with_sizes.default"
        ~bindings:[ ("self", float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]) ]
        ~inputs:
          [ in_tensor "self"; in_ints "split_sizes" [ 3 ]; in_int "dim" d ]
        ~noutputs:0)
    [ 2; -3 ];
  [%expect
    {|
    error: split_with_sizes.default: invalid dimension 2 for rank 2
    error: split_with_sizes.default: invalid dimension -3 for rank 2 |}]

(* Sizes that do not sum to the axis's extent are a shape error, not an
   [Invalid_dim] -- [Split.Split_with_sizes.output_shapes]'s own check, not
   anything the bridge itself validates. *)
let%expect_test
    "dispatch: split_with_sizes.default rejects sizes that do not sum to the \
     extent" =
  dispatch_print ~target:"torch.ops.aten.split_with_sizes.default"
    ~bindings:[ ("self", float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]) ]
    ~inputs:[ in_tensor "self"; in_ints "split_sizes" [ 1; 3 ] ]
    ~noutputs:0;
  [%expect
    {| error: split_with_sizes of axis W: sizes sum to 4, not the axis's extent 3 |}]

(* Split_with_sizes is a view operation like unbind: it copies storage cells
   and keeps the source format rather than taking the arithmetic f32
   materialization path. Values around 2^53 and the signed endpoints prove
   the result never went through an OCaml float. *)
let%expect_test
    "dispatch: split_with_sizes.default preserves int64 storage exactly" =
  dispatch_print ~target:"torch.ops.aten.split_with_sizes.default"
    ~bindings:
      [
        ( "self",
          i64_tensor [ 3; 2 ]
            [
              Int64.min_int;
              -1L;
              9_007_199_254_740_993L;
              9_007_199_254_740_995L;
              Int64.max_int;
              0L;
            ] );
      ]
    ~inputs:[ in_tensor "self"; in_ints "split_sizes" [ 2; 1 ] ]
    ~noutputs:0;
  [%expect
    {|
    tensor i64 [W=2 C=2] {-9223372036854775808, -1, 9007199254740993, 9007199254740995}
    tensor i64 [C=2] {9223372036854775807, 0} |}]

(* The real oracle: ATen runs the op, the native side runs
   [Graph_ir.Split_with_sizes] through [Eval_direct], and
   [Verify.verify_node] compares EVERY piece, the same reason
   [verify_unbind] does. *)
let verify_split_with_sizes ~inputs ~outputs ~self =
  let env = Sm.add "self" self Sm.empty in
  let node =
    PT.Node.make "torch.ops.aten.split_with_sizes.default"
      (PT.NamedArgument.make "self" (targ "self") None :: inputs)
      outputs Sm.empty None (Some "test")
  in
  match
    Interp_verify.dispatch ~verify:true ~ppf:Format.std_formatter env node
  with
  | Error e ->
      Format.printf "dispatch error: %a@." Interp_verify.pp_interp_error
        (Err.Error.kind e)
  | Ok _ -> print_string "aten and native agree\n"

let%expect_test "verify: split_with_sizes.default agrees with ATen" =
  let self = float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  verify_split_with_sizes
    ~inputs:[ in_ints "split_sizes" [ 1; 2 ] ]
    ~outputs:(tensors [ "a"; "b" ])
    ~self;
  (* And along the strided axis, where each piece is a column range. Dim -1
     has extent 2 here, so the sizes are [1;1] rather than [2;1]. *)
  verify_split_with_sizes
    ~inputs:[ in_ints "split_sizes" [ 1; 1 ]; in_int "dim" (-1) ]
    ~outputs:(tensors [ "a"; "b" ])
    ~self:(float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: split_with_sizes.default agrees with ATen for int64" =
  verify_split_with_sizes
    ~inputs:[ in_ints "split_sizes" [ 1; 2 ] ]
    ~outputs:(tensors [ "a"; "b" ])
    ~self:
      (i64_tensor [ 3; 2 ]
         [
           Int64.min_int;
           -1L;
           9_007_199_254_740_993L;
           9_007_199_254_740_995L;
           Int64.max_int;
           0L;
         ]);
  [%expect {| aten and native agree |}]

(* --- split.Tensor -----------------------------------------------------------

   The equal-chunk-size sibling of split_with_sizes.default just above:
   legalizes onto the *existing* [Split_with_sizes] node ([chunk_sizes]
   derives the sizes list ATen itself would split into), so this is
   "translate params, still one node", not a decomposition. Correctness of
   the NATIVE kernel itself is already pinned by split_with_sizes.default's
   own fixtures above; what this section verifies is that [chunk_sizes]'s
   derived sizes list is the one ATen itself produces, including the SMALLER
   FINAL chunk when [split_size] does not evenly divide the axis extent --
   the one behavior split_with_sizes.default itself cannot exercise (its
   sizes list is given directly, not derived). *)
let verify_split_tensor ~inputs ~outputs ~self =
  let env = Sm.add "self" self Sm.empty in
  let node =
    PT.Node.make "torch.ops.aten.split.Tensor"
      (PT.NamedArgument.make "self" (targ "self") None :: inputs)
      outputs Sm.empty None (Some "test")
  in
  match
    Interp_verify.dispatch ~verify:true ~ppf:Format.std_formatter env node
  with
  | Error e ->
      Format.printf "dispatch error: %a@." Interp_verify.pp_interp_error
        (Err.Error.kind e)
  | Ok _ -> print_string "aten and native agree\n"

let%expect_test "verify: split.Tensor agrees with ATen, evenly-dividing" =
  (* Leading dim (W after right-alignment) has extent 6, split_size 2: three
     equal 2-row chunks, no remainder. *)
  verify_split_tensor
    ~inputs:[ in_int "split_size" 2 ]
    ~outputs:(tensors [ "a"; "b"; "c" ])
    ~self:(float_tensor [ 6; 2 ] (List.init 12 float_of_int));
  [%expect {| aten and native agree |}]

(* The case split_with_sizes.default's own fixtures cannot exercise: a
   smaller FINAL chunk, since its sizes list is given directly rather than
   derived from a single [split_size]. Extent 5 / split_size 2 = chunks of
   [2; 2; 1], not [2; 2; 2] -- a wrong [chunk_sizes] (e.g. rounding up, or
   dropping the remainder) would either crash the shape check or disagree
   with ATen's own output arity/values here. *)
let%expect_test "verify: split.Tensor agrees with ATen, remainder chunk" =
  verify_split_tensor
    ~inputs:[ in_int "split_size" 2 ]
    ~outputs:(tensors [ "a"; "b"; "c" ])
    ~self:(float_tensor [ 5; 2 ] (List.init 10 float_of_int));
  (* And along the strided axis (dim=-1, extent 2): split_size 5 exceeds the
     extent, so ATen's own rule ("one chunk of the whole axis") applies --
     exactly [chunk_sizes]'s [remaining <= split_size] base case. *)
  verify_split_tensor
    ~inputs:[ in_int "split_size" 5; in_int "dim" (-1) ]
    ~outputs:(tensors [ "a" ])
    ~self:(float_tensor [ 5; 2 ] (List.init 10 float_of_int));
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "dispatch: split.Tensor rejects a non-positive split_size" =
  List.iter
    (fun split_size ->
      dispatch_print ~target:"torch.ops.aten.split.Tensor"
        ~bindings:[ ("self", float_tensor [ 4; 2 ] (List.init 8 float_of_int)) ]
        ~inputs:[ in_tensor "self"; in_int "split_size" split_size ]
        ~noutputs:0)
    [ 0; -1 ];
  [%expect
    {|
    error: split.Tensor: split_size must be positive, got 0
    error: split.Tensor: split_size must be positive, got -1 |}]
