(* A [Stack] on an in-frame axis read only by a [Reshape] back into the frame
   (the sin/cos interleave of edgenext's positional embedding) lowers to a
   reshape per operand, a [Concat4] on the stacked axis and a final reshape.
   Each case compares every output element-for-element with Native's own
   evaluation, on values that are not exact in f32. *)

open Axis

let s = Fixtures.s

(* Each operand gets its own values: operands that are equal would make any
   order of them agree. *)
let distinct () =
  let call = ref 0 in
  fun shape ->
    incr call;
    let offset = float_of_int (17 * !call) in
    let i = ref (-1) in
    Tensor.materialize shape (fun _ ->
        incr i;
        Float.sin (float_of_int !i +. 0.5 +. offset) *. 3.7)

let graph name ~k ~shape ~axis ~target ~extra =
  Graph_builder.build ~name ~outputs:Fun.id
    (let open Graph_builder in
     let* xs = Lower_region_test.inputs_of k shape in
     let* stacked = stack { Concat.Stack.axis } xs in
     let* y = reshape { Reshape.Reshape.shape = target } stacked in
     if extra then
       let* other = relu stacked in
       Graph_builder.return [ y; other ]
     else Graph_builder.return [ y ])
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let check name ~k ~shape ~axis ~target ?(extra = false) () =
  Lower_region_test.compare_graphs ~fill:(distinct ()) name
    ~x_shapes:(List.init k (fun _ -> shape))
    (graph name ~k ~shape ~axis ~target ~extra)

(* [H=4 W=5 C=6] stacked on C is [D=4 H=5 W=6 C=k]; reading it back as
   [H=4 W=5 C=6k] interleaves the operands element by element. *)
let%expect_test "interleave on the last axis" =
  check "two" ~k:2 ~shape:(s 1 1 1 4 5 6) ~axis:C ~target:(s 1 1 1 4 5 12) ();
  check "three" ~k:3 ~shape:(s 1 1 1 4 5 6) ~axis:C ~target:(s 1 1 1 4 5 18) ();
  [%expect
    {|
    two: 1 outputs, 4 nodes, identical=true
    three: 1 outputs, 5 nodes, identical=true |}]

(* Stacked on H instead, the outer axis is pushed out and the stacked one sits
   in the middle of the frame. *)
let%expect_test "interleave on an inner axis" =
  check "on H" ~k:2 ~shape:(s 1 1 1 4 5 6) ~axis:H ~target:(s 1 1 1 8 5 6) ();
  check "on W" ~k:3 ~shape:(s 1 1 1 4 5 6) ~axis:W ~target:(s 1 1 1 4 15 6) ();
  [%expect
    {|
    on H: 1 outputs, 4 nodes, identical=true
    on W: 1 outputs, 5 nodes, identical=true |}]

let%expect_test "refused: the stacked value has a second reader" =
  check "shared" ~k:2 ~shape:(s 1 1 1 4 5 6) ~axis:C ~target:(s 1 1 1 4 5 12)
    ~extra:true ();
  [%expect {| shared: tensor t2 has extent on T or D: [D=4 H=5 W=6 C=2] |}]

let%expect_test "the region's map verifies" =
  Lower_region_test.verified_graph "interleave"
    (graph "interleave" ~k:2 ~shape:(s 1 1 1 4 5 6) ~axis:C
       ~target:(s 1 1 1 4 5 12) ~extra:false);
  [%expect
    {| interleave: 7 clusters: 3 proved (structural) [sampled 8], 4 vacuous |}]
