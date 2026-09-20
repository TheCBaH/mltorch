(* Wide permutes: a run of reshape, clone and permute steps between two
   four-axis tensors whose interior carries T and D, planned by
   [Wide_permute] and emitted by [Lower_region]. Each case builds the Native
   graph, lowers it and compares every output element-for-element with
   Native's own evaluation; the node count is the number of Native4D nodes
   emitted, so a single-step case that emitted two shows up there. *)

open Native4d
open Axis

let s = Fixtures.s
let perm_of = Fixtures.perm_of
let compare = Lower_region_test.compare_graph

type step = Reshape of Vec6.shape | Clone | Permute of (Axis.t * Axis.t) list

let run_graph name ~x_shape steps =
  Graph_builder.build ~name ~outputs:Fun.id
    (let open Graph_builder in
     let* x = input ~shape:x_shape () in
     let rec go cur = function
       | [] -> Graph_builder.return [ cur ]
       | Reshape shape :: rest ->
           let* t = reshape { Reshape.Reshape.shape } cur in
           go t rest
       | Clone :: rest ->
           let* t = clone cur in
           go t rest
       | Permute perm :: rest ->
           let* t = permute (perm_of perm) cur in
           go t rest
     in
     go x steps)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let check name ~x_shape steps =
  compare name ~x_shape (run_graph name ~x_shape steps)

(* mobilevitv2: [H=224 W=28 C=28] -> (T=224 D=14 H=2 W=14 C=2) -> clone ->
   patches to the front -> [H=224 W=4 C=196]. The permute spans five non-unit
   axes, which no single Permute4 can hold. *)
let mobilevit_perm = [ (D, H); (H, C); (W, D); (C, W) ]

let%expect_test "five non-unit axes, through a clone" =
  check "mobilevitv2" ~x_shape:(s 1 1 1 224 28 28)
    [
      Reshape (s 1 224 14 2 14 2);
      Clone;
      Permute mobilevit_perm;
      Reshape (s 1 1 1 224 4 196);
    ];
  [%expect {| mobilevitv2: 1 outputs, 5 nodes, identical=true |}]

(* A transpose of two axes hidden behind T and D fuses to ONE Permute4: the
   source shape already is the permute's input, and its output already is the
   result, so nothing else may be emitted. *)
let%expect_test "a run that is one Permute4 emits one node" =
  check "transpose" ~x_shape:(s 1 1 1 4 6 5)
    [
      Reshape (s 1 4 6 1 1 5);
      Permute [ (T, D); (D, T) ];
      Reshape (s 1 1 1 6 4 5);
    ];
  [%expect {| transpose: 1 outputs, 1 nodes, identical=true |}]

(* Four blocks in the order (b d a c): its own inverse it is not, so a
   permutation applied in the wrong direction is a mismatch, not a
   coincidence. (Three blocks cannot show this: any rotation of three fuses two
   of them.) *)
let%expect_test "four blocks, not an involution" =
  check "b d a c" ~x_shape:(s 2 1 1 3 4 5)
    [
      Reshape (s 1 2 3 4 5 1);
      Permute [ (T, D); (D, W); (H, T); (W, H) ];
      Reshape (s 3 1 1 5 2 4);
    ];
  [%expect {| b d a c: 1 outputs, 1 nodes, identical=true |}]

(* All six axes reversed: six blocks that no single permute can reorder. *)
let%expect_test "six non-unit axes" =
  check "reversal" ~x_shape:(s 1 1 1 4 6 10)
    [
      Reshape (s 2 2 2 3 2 5);
      Permute [ (N, C); (T, W); (D, H); (H, D); (W, T); (C, N) ];
      Reshape (s 5 1 1 6 4 2);
    ];
  [%expect {| reversal: 1 outputs, 5 nodes, identical=true |}]

(* Two reshapes cut different axes, so the source splits into eight atoms
   that stay apart: more blocks than the planner takes. The run as a whole is
   refused; the fused-axes relabelling then plans each permute on its own and
   still lowers it. *)
let reverse6 = [ (N, C); (T, W); (D, H); (H, D); (W, T); (C, N) ]

let eight_atoms =
  [
    Reshape (s 2 3 10 2 7 15);
    Permute reverse6;
    Reshape (s 3 5 14 2 5 6);
    Permute reverse6;
    Reshape (s 6 1 1 10 14 15);
  ]

let%expect_test "more blocks than the planner takes" =
  let plan_of = function
    | Reshape shape -> Wide_permute.Reshape shape
    | Clone -> Wide_permute.Clone
    | Permute perm -> Wide_permute.Permute (perm_of perm)
  in
  Fmt.pr "planner: %s@."
    (match
       Wide_permute.plan
         ~x:(Shape4.of_ints ~n:6 ~h:10 ~w:14 ~c:15)
         ~y:(Shape4.of_ints ~n:6 ~h:10 ~w:14 ~c:15)
         (List.map plan_of eight_atoms)
     with
    | None -> "refused"
    | Some steps -> Fmt.str "%d steps" (List.length steps));
  check "eight atoms" ~x_shape:(s 6 1 1 10 14 15) eight_atoms;
  [%expect
    {|
    planner: refused
    eight atoms: 1 outputs, 13 nodes, identical=true |}]

(* A cut that falls inside an atom: [W=6 C=4] cannot be read as (T=4, W=6)
   without splitting the 6, which no reshape in the run did. *)
let%expect_test "a cut inside an atom" =
  let ops =
    [
      Reshape (s 1 4 1 1 6 1);
      Permute [ (T, W); (W, T) ];
      Reshape (s 1 1 1 1 6 4);
    ]
  in
  Fmt.pr "planner: %s@."
    (match
       Wide_permute.plan
         ~x:(Shape4.of_ints ~n:1 ~h:1 ~w:6 ~c:4)
         ~y:(Shape4.of_ints ~n:1 ~h:1 ~w:6 ~c:4)
         (List.map
            (function
              | Reshape shape -> Wide_permute.Reshape shape
              | Clone -> Wide_permute.Clone
              | Permute perm -> Wide_permute.Permute (perm_of perm))
            ops)
     with
    | None -> "refused"
    | Some steps -> Fmt.str "%d steps" (List.length steps));
  check "misaligned" ~x_shape:(s 1 1 1 1 6 4) ops;
  [%expect
    {|
    planner: refused
    misaligned: 1 outputs, 5 nodes, identical=true |}]

(* Nothing outside the domain: the ordinary path lowers each node itself. *)
let%expect_test "an all-in-domain run is left alone" =
  check "in domain" ~x_shape:(s 1 1 1 2 3 4)
    [
      Reshape (s 1 1 1 6 2 2);
      Permute [ (H, W); (W, H) ];
      Reshape (s 1 1 1 2 3 4);
    ];
  [%expect {| in domain: 1 outputs, 3 nodes, identical=true |}]

(* An interior tensor with a second reader is not the run's alone. *)
let%expect_test "refused: an interior tensor is also an output" =
  let g =
    Graph_builder.build ~name:"shared" ~outputs:Fun.id
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 4 6 5) () in
       let* r = reshape { Reshape.Reshape.shape = s 1 4 6 1 1 5 } x in
       let* p = permute (perm_of [ (T, D); (D, T) ]) r in
       let* y = reshape { Reshape.Reshape.shape = s 1 1 1 6 4 5 } p in
       Graph_builder.return [ y; p ])
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  compare "shared" ~x_shape:(s 1 1 1 4 6 5) g;
  [%expect {| shared: node n1: axis T is outside the N/H/W/C dialect |}]

let%expect_test "the region's map verifies" =
  Lower_region_test.verified_graph "mobilevitv2 (small)"
    (run_graph "mobilevitv2" ~x_shape:(s 1 1 1 6 8 8)
       [
         Reshape (s 1 6 4 2 4 2);
         Clone;
         Permute mobilevit_perm;
         Reshape (s 1 1 1 6 4 16);
       ]);
  [%expect
    {| mobilevitv2 (small): 9 clusters: 1 proved (structural) [sampled 8], 1 unproved (over max_rounds) [sampled 8], 7 vacuous |}]

let show_plan name ~x ~y ops =
  let pp_step fmt = function
    | Wide_permute.Reshape4 shape -> Fmt.pf fmt "reshape %a" Shape4.pp shape
    | Wide_permute.Permute4 (perm, shape) ->
        Fmt.pf fmt "permute %a -> %a"
          Fmt.(
            list ~sep:(any ",") (fun fmt (o, i) ->
                pf fmt "%a<-%a" Axis4.pp o Axis4.pp i))
          (List.filter (fun (o, i) -> o <> i) perm)
          Shape4.pp shape
  in
  Fmt.pr "%s: %a@." name
    Fmt.(option ~none:(any "none") (list ~sep:(any "; ") pp_step))
    (Wide_permute.plan ~x ~y ops)

let%expect_test "the planned steps" =
  (* Two of the three axes move together, so they fuse into one. *)
  show_plan "rotate"
    ~x:(Shape4.of_ints ~n:1 ~h:4 ~w:6 ~c:5)
    ~y:(Shape4.of_ints ~n:1 ~h:6 ~w:5 ~c:4)
    [
      Wide_permute.Reshape (s 1 4 6 5 1 1);
      Wide_permute.Permute (perm_of [ (T, D); (D, H); (H, T) ]);
      Wide_permute.Reshape (s 1 1 1 6 5 4);
    ];
  [%expect
    {|
    rotate: reshape [N=1 H=1 W=4 C=30]; permute W<-C,C<-W -> [N=1 H=1 W=30 C=4]; reshape
    [N=1 H=6 W=5 C=4] |}]

(* A clone between the reshape and the permute is absorbed into the existing
   regions: the qkv split and the convolution weight relayout. *)
let%expect_test "a clone inside a reshape-permute region" =
  let qkv_clone =
    Graph_builder.build ~name:"qkv_clone" ~outputs:Fun.id
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 1 5 24) () in
       let* r = reshape { Reshape.Reshape.shape = s 1 1 5 3 2 4 } x in
       let* c = clone r in
       let* p = permute (perm_of Lower_region_test.qkv) c in
       unbind { Split.Unbind.axis = T } p)
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  compare "qkv clone" ~x_shape:(s 1 1 1 1 5 24) qkv_clone;
  let relayout =
    run_graph "relayout_clone" ~x_shape:(s 1 1 1 1 8 12)
      [
        Reshape (s 1 1 8 3 2 2);
        Clone;
        Permute [ (N, D); (D, N); (H, W); (W, C); (C, H) ];
      ]
  in
  compare "relayout clone" ~x_shape:(s 1 1 1 1 8 12) relayout;
  [%expect
    {|
    qkv clone: 3 outputs, 9 nodes, identical=true
    relayout clone: 1 outputs, 2 nodes, identical=true |}]
