(* Native to Native4D conversion, continued from lower_test.ml (split under
   the tracked file-size ceiling, scripts/check-file-size.sh): the
   reshape/expand/repeat/transpose family, op3-impl.md commit 8's own scope.
   lower_fixtures.ml has the shared rendering helpers ([build]/[show]/
   [outcome]). *)

open Lower_fixtures

(* Sub is a direct binary legalization, exactly like Add: no relayout, one
   shape for equal-shape and a [C]-only operand for broadcast. *)
let%expect_test "lower: sub, equal shapes and broadcast" =
  show "sub equal"
    (build "sub"
       (let open Graph_builder in
        let* a = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        let* b = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        sub a b));
  show "sub broadcast"
    (build "sub"
       (let open Graph_builder in
        let* a = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        let* b = input ~shape:(Fixtures.chan 3) () in
        sub a b));
  [%expect
    {|
    sub equal:
      graph4
    inputs: [t0 [H=2 W=2 C=3],
    t1 [H=2 W=2 C=3]]
    nodes:
      n0: [t2] = sub a=t0 b=t1
    outputs: [t2 [H=2 W=2 C=3]]
    sub broadcast:
      graph4
    inputs: [t0 [H=2 W=2 C=3],
    t1 [C=3]]
    nodes:
      n0: [t2] = sub a=t0 b=t1
    outputs: [t2 [H=2 W=2 C=3]] |}]

(* Reshape4's target shape, read as ATen would serialize it: an ATen rank-r
   size list right-aligns onto the innermost r of [N T D H W C]
   (op3-impl.md F4), so a rank-3 target uses only H/W/C and a rank-4 target's
   LEADING entry lands on D -- D=1 is what [Shape4.of_vec6] requires, not N=1.
   Both valid targets below therefore have D=1 whether or not N itself is
   trivial; only a target whose ATen rank-4 leading entry is not 1 puts a real
   extent on D and leaves the dialect. *)
let reshape4 ~source ~target =
  build "reshape"
    (let open Graph_builder in
     let* x = input ~shape:source () in
     reshape { Reshape.Reshape.shape = target } x)

let%expect_test
    "lower: reshape4, a rank-3 and a rank-4 [1,h,w,c] target both lower" =
  let source = Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3 in
  (* rank-3 ATen target [4,3]: right-aligns onto W,C only (H trivial). *)
  show "rank-3 target"
    (reshape4 ~source ~target:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:4 ~c:3));
  (* rank-4 ATen target [1,2,2,3]: leading entry 1 lands on D, so D=1 and
     every one of N/H/W/C is populated. *)
  show "rank-4 [1,h,w,c] target"
    (reshape4 ~source ~target:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3));
  [%expect
    {|
    rank-3 target:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = reshape4 x=t0 params={shape=[N=1 H=1 W=4 C=3]}
    outputs: [t1 [W=4 C=3]]
    rank-4 [1,h,w,c] target:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = reshape4 x=t0 params={shape=[N=1 H=2 W=2 C=3]}
    outputs: [t1 [H=2 W=2 C=3]] |}]

let%expect_test
    "lower: reshape4, a rank-4 target with a non-1 leading entry is refused" =
  let source = Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3 in
  (* rank-4 ATen target [2,2,1,3]: leading entry 2 lands on D. *)
  outcome "rank-4 [n,h,w,c], n<>1"
    (reshape4 ~source ~target:(Vec6.shape ~n:1 ~t:1 ~d:2 ~h:2 ~w:1 ~c:3));
  [%expect
    {| rank-4 [n,h,w,c], n<>1     tensor t1 has extent on T or D: [D=2 H=2 W=1 C=3] |}]

(* [Expand4]'s own version of the same typed-target discipline: the target is
   a plain [Vec6.shape] at the Native op, so whether it stays four-axis is a
   fact about ITS OWN extents, not about [x_shape] -- unlike [Reshape4], no
   ATen right-alignment is involved here, since [Pointwise.Expand.params.size]
   already arrives as a full six-axis shape. *)
let expand4 ~source ~target =
  build "expand"
    (let open Graph_builder in
     let* x = input ~shape:source () in
     expand { Pointwise.Expand.size = target } x)

let%expect_test "lower: expand4, a target that stays four-axis lowers" =
  let source = Fixtures.nhwc ~n:1 ~h:1 ~w:4 ~c:1 in
  show "expand"
    (expand4 ~source ~target:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:4 ~c:5));
  [%expect
    {|
    expand:
      graph4
    inputs: [t0 [W=4 C=1]]
    nodes:
      n0: [t1] = expand4 x=t0 params={size=[N=1 H=3 W=4 C=5]}
    outputs: [t1 [H=3 W=4 C=5]] |}]

let%expect_test "lower: expand4, a target that broadcasts onto D is refused" =
  let source = Fixtures.nhwc ~n:1 ~h:1 ~w:1 ~c:3 in
  outcome "expand onto D"
    (expand4 ~source ~target:(Vec6.shape ~n:1 ~t:1 ~d:5 ~h:1 ~w:1 ~c:3));
  [%expect
    {| expand onto D              tensor t1 has extent on T or D: [D=5 H=1 W=1 C=3] |}]

(* [Repeat4]'s own version of the same typed-target discipline as [Expand4]:
   [Repeat.Repeat.params.repeats] is a plain [Vec6.shape] multiplier at the
   Native op, so whether it stays four-axis is a fact about ITS OWN extents.
   [Domain.check_node] admits every [Repeat] unconditionally (repeat.ml's own
   doc comment: every axis keeps its own identity, so a T/D-unit multiplier
   composed with a T/D-unit input is always T/D-unit on output) -- the real
   gate is [Graph_shape4]'s [four] wrap and the lowerer's own
   [Shape4.of_vec6], exercised here directly. *)
let repeat4 ~source ~repeats =
  build "repeat"
    (let open Graph_builder in
     let* x = input ~shape:source () in
     repeat { Repeat.Repeat.repeats } x)

let%expect_test "lower: repeat4, a T/D-unit multiplier lowers" =
  let source = Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3 in
  show "repeat"
    (repeat4 ~source ~repeats:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1));
  [%expect
    {|
    repeat:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = repeat4 x=t0 params={repeats=[N=1 H=1 W=2 C=1]}
    outputs: [t1 [H=2 W=4 C=3]] |}]

let%expect_test "lower: repeat4, a multiplier that tiles D is refused" =
  let source = Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3 in
  outcome "repeat onto D"
    (repeat4 ~source ~repeats:(Vec6.shape ~n:1 ~t:1 ~d:5 ~h:1 ~w:1 ~c:1));
  [%expect
    {| repeat onto D              tensor t1 has extent on T or D: [D=5 H=2 W=2 C=3] |}]

(* [RepeatInterleave4] carries ONE named axis, so it gets [Select4]'s own
   [check_dims] treatment: rejected at the DOMAIN check, before a
   destination graph is even attempted -- a nicer diagnostic than
   [Repeat4]'s post-hoc [Shape4.of_vec6] failure above, per
   [Ops4.RepeatInterleave4]'s own comment on why the two differ. *)
let repeat_interleave4 ~source ~axis ~repeats =
  build "repeat_interleave"
    (let open Graph_builder in
     let* x = input ~shape:source () in
     repeat_interleave
       { Repeat.RepeatInterleave.axis; repeats = Op_config.Pos.of_int repeats }
       x)

let%expect_test "lower: repeat_interleave4 on H lowers" =
  let source = Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3 in
  show "repeat_interleave" (repeat_interleave4 ~source ~axis:Axis.H ~repeats:3);
  [%expect
    {|
    repeat_interleave:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = repeat_interleave4 x=t0 params={axis=H repeats=3}
    outputs: [t1 [H=6 W=2 C=3]] |}]

let%expect_test "lower: repeat_interleave4 on T is refused at the domain check"
    =
  let source = Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3 in
  outcome "repeat_interleave on T"
    (repeat_interleave4 ~source ~axis:Axis.T ~repeats:3);
  [%expect
    {| repeat_interleave on T     node n0: axis T is outside the N/H/W/C dialect |}]

(* [Softmax4] carries ONE named axis, the same [check_dims] treatment
   [RepeatInterleave4] just above gets: rejected at the DOMAIN check, before a
   destination graph is even attempted. Unlike [RepeatInterleave4], there is
   no post-hoc [Shape4.of_vec6] alternative to contrast it with -- softmax's
   output shape is [x]'s own, unchanged, so the axis check is the only gate. *)
let softmax4 ~source ~axis =
  build "softmax"
    (let open Graph_builder in
     let* x = input ~shape:source () in
     softmax { Reduce.Softmax.axis } x)

let%expect_test "lower: softmax4 over W lowers" =
  let source = Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3 in
  show "softmax" (softmax4 ~source ~axis:Axis.W);
  [%expect
    {|
    softmax:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = softmax4 x=t0 params={axis=W}
    outputs: [t1 [H=2 W=2 C=3]] |}]

let%expect_test "lower: softmax4 over D is refused at the domain check" =
  let source = Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3 in
  outcome "softmax on D" (softmax4 ~source ~axis:Axis.D);
  [%expect
    {| softmax on D               node n0: axis D is outside the N/H/W/C dialect |}]

(* transpose.int on a rank-4 [1,h,w,c] ATen source: dim 0 is D (the leading
   entry, D=1 so the SHAPE is in-domain), dims 1/2/3 are H/W/C. Every rank-4
   fixture below uses DISTINCT, non-unit h/w/c so a wrong pair swapped is
   visible in the printed shape, not just in whether it converts -- a
   conventional shape like [2,3,4,5] would put a real batch on D and answer
   the shape-domain question instead of the axis-domain one (op3-impl.md
   round-1 review response). *)
let transpose4 ~source pairs =
  build "transpose"
    (let open Graph_builder in
     let* x = input ~shape:source () in
     permute (Fixtures.perm_of pairs) x)

let rank4_source = Fixtures.nhwc ~n:1 ~h:3 ~w:4 ~c:5

let%expect_test
    "lower: permute4 from transpose of dims 1/2, 1/3, 2/3 on [1,h,w,c]" =
  let open Axis in
  show "dims 1/2 (H,W)" (transpose4 ~source:rank4_source [ (H, W); (W, H) ]);
  show "dims 1/3 (H,C)" (transpose4 ~source:rank4_source [ (H, C); (C, H) ]);
  show "dims 2/3 (W,C)" (transpose4 ~source:rank4_source [ (W, C); (C, W) ]);
  [%expect
    {|
    dims 1/2 (H,W):
      graph4
    inputs: [t0 [H=3 W=4 C=5]]
    nodes:
      n0: [t1] = permute4 x=t0 perm=[H<-W, W<-H]
    outputs: [t1 [H=4 W=3 C=5]]
    dims 1/3 (H,C):
      graph4
    inputs: [t0 [H=3 W=4 C=5]]
    nodes:
      n0: [t1] = permute4 x=t0 perm=[H<-C, C<-H]
    outputs: [t1 [H=5 W=4 C=3]]
    dims 2/3 (W,C):
      graph4
    inputs: [t0 [H=3 W=4 C=5]]
    nodes:
      n0: [t1] = permute4 x=t0 perm=[W<-C, C<-W]
    outputs: [t1 [H=3 W=5 C=4]] |}]

(* transpose naming dim 0 (D) is a different question from the shape-domain
   one above -- [Domain.check] runs every per-node predicate BEFORE the shape
   sweep (domain.ml:265-269), so this fires even though [1,h,w,c]'s SHAPE is
   otherwise in-domain. *)
let%expect_test
    "lower: permute4 from transpose naming dim 0 is refused, naming D" =
  let open Axis in
  outcome "dims 0/1 (D,H)" (transpose4 ~source:rank4_source [ (D, H); (H, D) ]);
  [%expect
    {| dims 0/1 (D,H)             node n0: axis D is outside the N/H/W/C dialect |}]

(* The SHAPE question, kept separate from the axis question above: dims 1/2
   on a source whose ATen leading entry (n<>1) puts a real extent on D. The
   perm itself only names H/W, both in-domain -- it is the SOURCE that leaves
   the dialect. *)
let%expect_test
    "lower: transpose of dims 1/2 on [n,h,w,c], n<>1 is refused, not by the \
     axis check" =
  let open Axis in
  let source = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:3 ~w:4 ~c:5 in
  outcome "dims 1/2, d<>1" (transpose4 ~source [ (H, W); (W, H) ]);
  [%expect
    {| dims 1/2, d<>1             tensor t0 has extent on T or D: [D=2 H=3 W=4 C=5] |}]
