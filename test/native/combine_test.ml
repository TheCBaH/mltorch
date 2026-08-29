(* The axis-combining/splitting family: unbind, split_with_sizes, concat,
   and stack. Split from compute_test.ml. *)

open Compute_fixtures

(* --- Unbind ---------------------------------------------------------------

   Hand-computed, not compared against another instantiation of the same
   functor: agreement between Direct and Symbolic would prove staging, not
   arithmetic. Value at (h,w,c) is h*100 + w*10 + c, so each printed slice can
   be read straight off its coordinates. *)

(* Unbind DROPS the axis it selects along, so the survivors re-pack
   right-aligned exactly as mean(keepdim=false) does. [H2 W3 C2] unbound on H
   gives [W3 C2]; on W gives [W2 C2] with H's data now on W; on C gives [W2 C3]
   with H on W and W on C. Every ordinal is printed, because a bug that only
   gets output 0 right is exactly what a singleton test cannot see. *)
let%expect_test "Direct: unbind along an outer, a middle and an inner axis" =
  let module U = Split.Unbind.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int ((row c * 100) + (col c * 10) + chan c))
  in
  let run axis =
    let p = { Split.Unbind.axis } in
    let shapes =
      Split.Unbind.output_shapes ~x_shape p
      |> Err.or_raise ~pp_error:Shape_error.pp
    in
    Format.printf "unbind %a -> %d outputs@." Axis.pp axis (List.length shapes);
    List.iteri
      (fun i sh ->
        Format.printf "  out%d %a %a@." i Vec6.pp_shape sh Tensor.pp
          (Schedule.evaluate sh (U.pixel p ~output:i ~x)))
      shapes
  in
  run Axis.H;
  run Axis.W;
  run Axis.C;
  [%expect
    {|
    unbind H -> 2 outputs
      out0 [W=3 C=2] tensor f32 [W=3 C=2] {0, 1, 10, 11, 20, 21}
      out1 [W=3 C=2] tensor f32 [W=3 C=2] {100, 101, 110, 111, 120, 121}
    unbind W -> 3 outputs
      out0 [W=2 C=2] tensor f32 [W=2 C=2] {0, 1, 100, 101}
      out1 [W=2 C=2] tensor f32 [W=2 C=2] {10, 11, 110, 111}
      out2 [W=2 C=2] tensor f32 [W=2 C=2] {20, 21, 120, 121}
    unbind C -> 2 outputs
      out0 [W=2 C=3] tensor f32 [W=2 C=3] {0, 10, 20, 100, 110, 120}
      out1 [W=2 C=3] tensor f32 [W=2 C=3] {1, 11, 21, 101, 111, 121} |}]

(* --- Split_with_sizes -------------------------------------------------

   Hand-computed, same attribution discipline as the Unbind block above.
   Unlike [Unbind], the axis is KEPT in every output, so each piece is
   exactly what [Slice] would give for its window -- windows of DIFFERENT
   width here, so a wrong per-output offset is visible in which values land
   in which piece, not only in the piece count. *)
let%expect_test "Direct: split_with_sizes divides W into windows of 2 and 3" =
  let module Sw = Split.Split_with_sizes.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:5 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 10) + col c))
  in
  let params = { Split.Split_with_sizes.axis = Axis.W; sizes = [ 2; 3 ] } in
  let shapes =
    Split.Split_with_sizes.output_shapes ~x_shape params
    |> Err.or_raise ~pp_error:Shape_error.pp
  in
  let _, offsets =
    List.fold_left
      (fun (acc, os) size -> (acc + size, os @ [ acc ]))
      (0, []) params.Split.Split_with_sizes.sizes
  in
  List.iteri
    (fun i (sh, offset) ->
      Format.printf "out%d %a %a@." i Vec6.pp_shape sh Tensor.pp
        (Schedule.evaluate sh (Sw.pixel ~offset params ~x)))
    (List.combine shapes offsets);
  [%expect
    {|
    out0 [H=2 W=2 C=1] tensor f32 [H=2 W=2 C=1] {0, 1, 10, 11}
    out1 [H=2 W=3 C=1] tensor f32 [H=2 W=3 C=1] {2, 3, 4, 12, 13, 14} |}]

(* The two shape faults, checked at [output_shapes] rather than
   [Compute.pixel] (which trusts the shapes it is given, like every other
   op): a non-positive size (Native has no empty extent, [Slice]'s [Empty]
   rule) and a sizes list that does not sum to the axis's extent. *)
let%expect_test
    "split_with_sizes output_shapes: a non-positive size and a bad sum" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:5 ~c:1 in
  let at sizes =
    Format.printf "%a@."
      (pp_result (fun ppf shapes ->
           Format.fprintf ppf "%d outputs" (List.length shapes)))
      (Split.Split_with_sizes.output_shapes ~x_shape
         { Split.Split_with_sizes.axis = Axis.W; sizes })
  in
  at [ 2; 0; 3 ];
  at [ 2; -1; 4 ];
  at [ 2; 2 ];
  at [ 2; 4 ];
  at [ 2; 3 ];
  [%expect
    {|
    split_with_sizes of axis W over extent 5: size 0 at index 1 is not positive; the engine has no empty extent
    split_with_sizes of axis W over extent 5: size -1 at index 1 is not positive; the engine has no empty extent
    split_with_sizes of axis W: sizes sum to 4, not the axis's extent 5
    split_with_sizes of axis W: sizes sum to 6, not the axis's extent 5
    2 outputs |}]

(* Same ceiling [Unbind.output_shapes] checks, on the LIST LENGTH rather than
   a derived count. *)
let%expect_test
    "split_with_sizes output_shapes: the output-count ceiling is exclusive" =
  let limit = Kernel.Limits.Hard.outputs in
  let at n =
    let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:n ~w:1 ~c:1 in
    Format.printf "%d -> %a@." n
      (Core.Pretty.err_result
         ~ok:(fun ppf shapes ->
           Format.fprintf ppf "%d outputs" (List.length shapes))
         ~error:Shape_error.pp)
      (Split.Split_with_sizes.output_shapes ~x_shape
         {
           Split.Split_with_sizes.axis = Axis.H;
           sizes = List.init n (fun _ -> 1);
         })
  in
  List.iter at [ limit - 1; limit; limit + 1 ];
  [%expect
    {|
    4095 -> 4095 outputs
    4096 -> 4096 outputs, above the maximum of 4095
    4097 -> 4097 outputs, above the maximum of 4095 |}]

(* --- Concat -----------------------------------------------------------

   Hand-computed, same attribution discipline as the Unbind block above: each
   operand's values carry a distinguishing base offset (a: none, b: 1000, c:
   2000) so the printed output can be read straight off which operand and
   local coordinate produced each entry. *)
let%expect_test "Direct: concat along C, three operands of different width" =
  let module Cc = Concat.Concat.Compute (Direct) in
  let a_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let b_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  let c_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let a =
    Tensor.materialize a_shape (fun c ->
        float_of_int ((row c * 100) + (col c * 10)))
  in
  let b =
    Tensor.materialize b_shape (fun c ->
        float_of_int (1000 + (row c * 100) + (col c * 10) + chan c))
  in
  let c_ =
    Tensor.materialize c_shape (fun c ->
        float_of_int (2000 + (row c * 100) + (col c * 10)))
  in
  let params = { Concat.Concat.axis = Axis.C } in
  let result =
    eval_tensor
      (Concat.Concat.output_shape
         ~xs_shapes:[ a_shape; b_shape; c_shape ]
         params)
      (Cc.pixel params ~xs:[ (a_shape, a); (b_shape, b); (c_shape, c_) ])
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  [%expect
    {| tensor f32 [H=2 W=2 C=4] {0, 1000, 1001, 2000, 10, 1010, 1011, 2010, ...} |}]

(* Axis generality: the same op along an OUTER axis (H), two operands of
   different height. [a]'s single row lands first (output h=0); [b]'s two
   rows follow (output h=1,2). *)
let%expect_test "Direct: concat along H, two operands of different height" =
  let module Cc = Concat.Concat.Compute (Direct) in
  let a_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let b_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let a = Tensor.materialize a_shape (fun c -> float_of_int (col c * 10)) in
  let b =
    Tensor.materialize b_shape (fun c ->
        float_of_int (1000 + (row c * 100) + (col c * 10)))
  in
  let params = { Concat.Concat.axis = Axis.H } in
  let result =
    eval_tensor
      (Concat.Concat.output_shape ~xs_shapes:[ a_shape; b_shape ] params)
      (Cc.pixel params ~xs:[ (a_shape, a); (b_shape, b) ])
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  [%expect {| tensor f32 [H=3 W=2 C=1] {0, 10, 1000, 1010, 1100, 1110} |}]

(* The overflow/mismatch faults, checked at [output_shape] rather than
   [Compute.pixel] (which trusts the shapes it is given, like every other
   op). *)
let%expect_test "concat output_shape: empty list and a non-concat-axis mismatch"
    =
  let a_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let b_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:2 ~c:1 in
  let params = { Concat.Concat.axis = Axis.C } in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Concat.Concat.output_shape ~xs_shapes:[] params);
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Concat.Concat.output_shape ~xs_shapes:[ a_shape; b_shape ] params);
  [%expect
    {|
    concat: at least one tensor is required
    concat: axis H extent must agree across every tensor (it is not the concatenated axis): 2 vs 3 |}]

(* The ceiling is EXCLUSIVE, matching [Kernel.Limits.create]'s own [v >= hard]
   test, so 4095 is the largest accepted count. Checked here rather than in the
   builder because this is the boundary that runs BEFORE the list exists: a
   builder-side check would already have paid for the allocation it prevents. *)
let%expect_test "unbind output_shapes: the output-count ceiling is exclusive" =
  let limit = Kernel.Limits.Hard.outputs in
  let at n =
    let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:n ~w:1 ~c:1 in
    Format.printf "%d -> %a@." n
      (Core.Pretty.err_result
         ~ok:(fun ppf shapes ->
           Format.fprintf ppf "%d outputs" (List.length shapes))
         ~error:Shape_error.pp)
      (Split.Unbind.output_shapes ~x_shape { Split.Unbind.axis = Axis.H })
  in
  List.iter at [ limit - 1; limit; limit + 1 ];
  [%expect
    {|
    4095 -> 4095 outputs
    4096 -> 4096 outputs, above the maximum of 4095
    4097 -> 4097 outputs, above the maximum of 4095 |}]

(* ---- Stack -----------------------------------------------------------

   The case that caught the shape/coordinate direction bug during
   development: [dim] is NOT the outermost insertion position,
   so the real per-operand extent moves onto a DIFFERENT native axis than the
   one the operand's own storage uses (here: native W -> native H), and a
   naive pass-through of the output coordinate into the operand would read
   the wrong, always-extent-1 slot. [a]'s values are row*10+col, [b]'s are
   offset by +100, so a swapped operand or a misrouted coordinate is legible
   rather than merely unequal. *)
let%expect_test "Direct: stack at a non-outermost dim, two rank-2 operands" =
  let module St = Concat.Stack.Compute (Direct) in
  (* Native shape for a rank-2 ATen tensor [3,4]: the innermost two axes,
     W (rows) and C (cols). *)
  let a_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:4 in
  let a =
    Tensor.materialize a_shape (fun c -> float_of_int ((col c * 10) + chan c))
  in
  let b =
    Tensor.materialize a_shape (fun c ->
        float_of_int (100 + (col c * 10) + chan c))
  in
  (* ATen [torch.stack([a, b], dim=1)] on two [3,4] operands: axis_of_dim
     ~rank:3 1 = W, the SAME axis each operand's real row extent already
     occupies -- the case that exercises the direction bug. *)
  let params = { Concat.Stack.axis = Axis.W } in
  let result =
    eval_tensor
      (Concat.Stack.output_shape ~xs_shapes:[ a_shape; a_shape ] params)
      (St.pixel params ~xs:[ a; b ])
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  [%expect
    {|
    tensor f32 [H=3 W=2 C=4] {0, 1, 2, 3, 100, 101, 102, 103, ...} |}]

(* The boundary case, dim=0: here [Concat]'s and [Stack]'s coordinate spaces
   coincide, since inserting the OUTERMOST axis relabels nothing -- this is
   the case both fold directions agree on, so it alone would NOT have caught
   the bug above; it is here to pin the boundary now that the general case is
   covered. *)
let%expect_test "Direct: stack at the outermost dim" =
  let module St = Concat.Stack.Compute (Direct) in
  let a_shape = s1c 3 in
  let a = Tensor.materialize a_shape (fun c -> float_of_int (chan c)) in
  let b = Tensor.materialize a_shape (fun c -> float_of_int (100 + chan c)) in
  let params = { Concat.Stack.axis = Axis.W } in
  let result =
    eval_tensor
      (Concat.Stack.output_shape ~xs_shapes:[ a_shape; a_shape ] params)
      (St.pixel params ~xs:[ a; b ])
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  [%expect {| tensor f32 [W=2 C=3] {0, 1, 2, 100, 101, 102} |}]
