(* [Lower_region]: a [Reshape -> Permute (-> Select/Unbind)] run whose internal
   tensors carry T or D is replaced by four-axis data movement. Each case builds
   the Native graph, lowers it, and compares EVERY output element-for-element
   against Native's own evaluation of the same input, so a wrong slice or a
   wrong relabelling is a mismatch rather than a plausible-looking graph. The
   refusals pin what stays out of domain. *)

open Native4d

let s = Fixtures.s
let perm_of = Fixtures.perm_of

let seq shape =
  let i = ref (-1.) in
  Tensor.materialize shape (fun _ ->
      i := !i +. 1.;
      !i)

let values t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Tensor.read_at t (Vec6.get c) :: !acc);
  List.rev !acc

type tail = Unbind of Axis.t | Selects of Axis.t * int list | No_tail

let region_graph name ~x_shape ~tgt ~perm ~tail =
  Graph_builder.build ~name ~outputs:Fun.id
    (let open Graph_builder in
     let* x = input ~shape:x_shape () in
     let* r = reshape { Reshape.Reshape.shape = tgt } x in
     let* p = permute (perm_of perm) r in
     match tail with
     | No_tail -> Graph_builder.return [ p ]
     | Unbind axis -> unbind { Split.Unbind.axis } p
     | Selects (axis, ks) ->
         let rec go = function
           | [] -> Graph_builder.return []
           | k :: rest ->
               let* y = select { Split.Select.axis; index = Dim.index k } p in
               let* ys = go rest in
               Graph_builder.return (y :: ys)
         in
         go ks)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let compare_graphs ?(fill = seq) name ~x_shapes g =
  match Snapshot.create g with
  | Error _ -> Format.printf "%s: snapshot failed@." name
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert src with
      | Error e -> Format.printf "%s: %a@." name Error.pp (Err.Error.kind e)
      | Ok (Lower.Pack r) ->
          let dst = Lower.graph r in
          let xs = List.map fill x_shapes in
          let native =
            Eval_direct.run g
              ~inputs:(List.combine g.Graph_common.Graph.inputs xs)
            |> Err.or_raise ~pp_error:Eval_direct.pp_error
          in
          let four =
            Eval_direct4.run dst
              ~constants:(Tensor_id.Map.bindings r.Lower.constants)
              ~inputs:(List.combine dst.Graph_common.Graph.inputs xs)
            |> Err.or_raise ~pp_error:Eval_direct4.pp_error
          in
          let same =
            List.for_all2
              (fun a b ->
                values (Tensor_id.Map.find a native)
                = values (Tensor_id.Map.find b four))
              g.Graph_common.Graph.outputs dst.Graph_common.Graph.outputs
          in
          Format.printf "%s: %d outputs, %d nodes, identical=%b@." name
            (List.length g.Graph_common.Graph.outputs)
            (List.length dst.Graph_common.Graph.nodes)
            same)

let compare_graph name ~x_shape g = compare_graphs name ~x_shapes:[ x_shape ] g

let compare name ~x_shape ~tgt ~perm ~tail =
  compare_graph name ~x_shape (region_graph name ~x_shape ~tgt ~perm ~tail)

open Axis

(* [N=n] tokens, [k] q/k/v-style splits, [h] heads, [d] head width; the qkv
   permutation of every attention block in the corpus. *)
let qkv = [ (T, H); (D, T); (H, W); (W, D) ]

let%expect_test "qkv split: three-way unbind" =
  compare "unbind k=3" ~x_shape:(s 1 1 1 1 5 24) ~tgt:(s 1 1 5 3 2 4) ~perm:qkv
    ~tail:(Unbind T);
  [%expect {| unbind k=3: 3 outputs, 9 nodes, identical=true |}]

(* edgenext moves the token axis last: [T=3 D=1 H=heads W=hd C=tokens]. *)
let%expect_test "qkv split: transposed head" =
  compare "transposed" ~x_shape:(s 1 1 1 1 5 24) ~tgt:(s 1 1 5 3 2 4)
    ~perm:[ (T, H); (D, T); (H, W); (W, C); (C, D) ]
    ~tail:(Unbind T);
  [%expect {| transposed: 3 outputs, 9 nodes, identical=true |}]

(* convit: two-way, consumed as selects, out of order and with one repeated. *)
let%expect_test "qkv split: two-way selects" =
  compare "selects k=2" ~x_shape:(s 1 1 1 1 5 16) ~tgt:(s 1 1 5 2 2 4) ~perm:qkv
    ~tail:(Selects (T, [ 1; 0; 1 ]));
  [%expect {| selects k=2: 3 outputs, 9 nodes, identical=true |}]

(* mvitv2: one head, so the head axis is unit in the target. *)
let%expect_test "qkv split: one head" =
  compare "heads=1" ~x_shape:(s 1 1 1 1 6 12) ~tgt:(s 1 1 6 3 1 4) ~perm:qkv
    ~tail:(Unbind T);
  [%expect {| heads=1: 3 outputs, 3 nodes, identical=true |}]

(* The batch is on N when x carries one, so the split axis is still [C]. *)
let%expect_test "qkv split: leading batch on x" =
  compare "batched x" ~x_shape:(s 1 1 1 3 5 24) ~tgt:(s 1 1 15 3 2 4) ~perm:qkv
    ~tail:(Unbind T);
  [%expect {| batched x: 3 outputs, 9 nodes, identical=true |}]

(* No select: only the reshape target is out of domain (a conv weight
   relayout: [W=8 C=12] -> [D=8 H=3 W=2 C=2] -> [N=8 H=2 W=2 C=3]). *)
let%expect_test "reshape then permute, in-domain result" =
  compare "relayout" ~x_shape:(s 1 1 1 1 8 12) ~tgt:(s 1 1 8 3 2 2)
    ~perm:[ (N, D); (D, N); (H, W); (W, C); (C, H) ]
    ~tail:No_tail;
  [%expect {| relayout: 1 outputs, 2 nodes, identical=true |}]

(* No region: the split does not align with any source axis, so the fused-axes
   relabel lowers it instead. *)
let%expect_test "misaligned split, lowered by the relabel" =
  compare "misaligned" ~x_shape:(s 1 1 1 1 15 8) ~tgt:(s 1 1 5 3 2 4) ~perm:qkv
    ~tail:(Unbind T);
  [%expect {| misaligned: 3 outputs, 5 nodes, identical=true |}]

(* Refusals: nothing is silently rewritten into a wrong graph. *)
let%expect_test "refused: a real batch on T as well as tokens on D" =
  compare "batch and tokens" ~x_shape:(s 1 1 1 2 5 24) ~tgt:(s 1 2 5 3 2 4)
    ~perm:[ (T, H); (D, T); (H, W); (W, D) ]
    ~tail:(Unbind T);
  [%expect
    {| batch and tokens: node n1: axis T is outside the N/H/W/C dialect |}]

(* The map that ties the two graphs together is checked too: a region is one
   node cluster over several source nodes. *)
let verified_graph name g =
  match Snapshot.create g with
  | Error _ -> Format.printf "%s: snapshot failed@." name
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert src with
      | Error e -> Format.printf "%s: %a@." name Error.pp (Err.Error.kind e)
      | Ok (Lower.Pack r) -> (
          let effort = Map_verify.Effort.Standard in
          match
            Framework.Verify_from_native.run
              ~budget:(Map_verify.Effort.budget effort)
              ~probe:(Map_verify.Effort.probe effort)
              ~src_constants:Tensor_id.Map.empty
              ~dst_constants:r.Lower.constants r.Lower.map ~src ~dst:r.Lower.dst
          with
          | Error e ->
              Format.printf "%s: %a@." name Map_verify.pp_error
                (Err.Error.kind e)
          | Ok report ->
              Format.printf "%s: %s@." name (Map_verify.Report.summary report)))

let verified name ~x_shape ~tgt ~perm ~tail =
  verified_graph name (region_graph name ~x_shape ~tgt ~perm ~tail)

let%expect_test "the region's map verifies" =
  verified "qkv" ~x_shape:(s 1 1 1 1 5 24) ~tgt:(s 1 1 5 3 2 4) ~perm:qkv
    ~tail:(Unbind T);
  verified "relayout" ~x_shape:(s 1 1 1 1 8 12) ~tgt:(s 1 1 8 3 2 2)
    ~perm:[ (N, D); (D, N); (H, W); (W, C); (C, H) ]
    ~tail:No_tail;
  [%expect
    {|
    qkv: 12 clusters: 4 proved (structural) [sampled 8], 8 vacuous
    relayout: 4 clusters: 2 proved (structural) [sampled 8], 2 vacuous |}]

(* Option B: a permutation that only routes unit axes through T and D, over an
   operand and result that are both in domain. *)
let unit_perm ~x_shape ~perm =
  Fixtures.build "unit_perm"
    (let open Graph_builder in
     let* x = input ~shape:x_shape () in
     permute (perm_of perm) x)

let%expect_test "a permutation through unit T/D axes lowers" =
  compare_graph "square swap through D" ~x_shape:(s 1 1 1 1 4 4)
    (unit_perm ~x_shape:(s 1 1 1 1 4 4)
       ~perm:[ (N, C); (D, N); (H, D); (W, H); (C, W) ]);
  compare_graph "transpose through T and D" ~x_shape:(s 1 1 1 1 3 5)
    (unit_perm ~x_shape:(s 1 1 1 1 3 5)
       ~perm:[ (T, D); (D, T); (W, C); (C, W) ]);
  (* A non-unit extent landing on D is still out of domain. *)
  compare_graph "C onto D" ~x_shape:(s 1 1 1 1 3 5)
    (unit_perm ~x_shape:(s 1 1 1 1 3 5) ~perm:[ (D, C); (C, D) ]);
  [%expect
    {|
    square swap through D: 1 outputs, 1 nodes, identical=true
    transpose through T and D: 1 outputs, 1 nodes, identical=true
    C onto D: node n0: axis D is outside the N/H/W/C dialect |}]

(* ---- D read as N: split attention ---------------------------------------- *)

(* Values that are not exact in f32, so a different summation order or a stray
   rounding step changes the bits instead of hiding in small integers. *)
let inexact shape =
  let i = ref (-1) in
  Tensor.materialize shape (fun _ ->
      incr i;
      Float.sin (float_of_int !i +. 0.5) *. 3.7)

let rec inputs_of k shape =
  let open Graph_builder in
  if k = 0 then Graph_builder.return []
  else
    let* x = input ~shape () in
    let* rest = inputs_of (k - 1) shape in
    Graph_builder.return (x :: rest)

let sk_graph ~radix ~h =
  Graph_builder.build ~name:"split_attention" ~outputs:Fun.id
    (let open Graph_builder in
     let* xs = inputs_of radix (s 1 1 1 h 3 2) in
     let* logits = input ~shape:(s 1 1 1 (radix * h) 1 1) () in
     let* stacked = stack { Concat.Stack.axis = D } xs in
     let* pooled = sum { Reduce.Sum.dims = [ D ]; keepdim = false } stacked in
     let* w = reshape { Reshape.Reshape.shape = s 1 1 radix h 1 1 } logits in
     let* w = softmax { Reduce.Softmax.axis = D } w in
     let* weighted = mul stacked w in
     let* out = sum { Reduce.Sum.dims = [ D ]; keepdim = false } weighted in
     Graph_builder.return [ pooled; out ])
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let sk_shapes ~radix ~h =
  List.init radix (fun _ -> s 1 1 1 h 3 2) @ [ s 1 1 1 (radix * h) 1 1 ]

let%expect_test "split attention: stack, softmax and weighted sum over D" =
  List.iter
    (fun radix ->
      compare_graphs ~fill:inexact
        (Printf.sprintf "radix=%d" radix)
        ~x_shapes:(sk_shapes ~radix ~h:4) (sk_graph ~radix ~h:4))
    [ 2; 3 ];
  [%expect
    {|
    radix=2: 2 outputs, 6 nodes, identical=true
    radix=3: 2 outputs, 6 nodes, identical=true |}]

(* ResNeSt introduces the axis with a reshape of the leading axis instead. *)
let%expect_test "split attention: reshape introduces the axis" =
  let radix = 2 and h = 4 in
  let g =
    Graph_builder.build ~name:"resnest" ~outputs:Fun.id
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 (radix * h) 3 2) () in
       let* logits = input ~shape:(s 1 1 1 (radix * h) 1 1) () in
       let* split = reshape { Reshape.Reshape.shape = s 1 1 radix h 3 2 } x in
       let* pooled = sum { Reduce.Sum.dims = [ D ]; keepdim = false } split in
       let* w = reshape { Reshape.Reshape.shape = s 1 1 radix h 1 1 } logits in
       let* weighted = mul split w in
       let* out = sum { Reduce.Sum.dims = [ D ]; keepdim = false } weighted in
       Graph_builder.return [ pooled; out ])
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  compare_graphs ~fill:inexact "reshape"
    ~x_shapes:[ s 1 1 1 (radix * h) 3 2; s 1 1 1 (radix * h) 1 1 ]
    g;
  [%expect {| reshape: 2 outputs, 5 nodes, identical=true |}]

(* Refusals: a group that meets anything outside the supported set, or that
   leaves the graph, is not partially rewritten. *)
let%expect_test "a D-carrying value that is a graph output stays refused" =
  let g =
    Graph_builder.build ~name:"out_d" ~outputs:Fun.id
      (let open Graph_builder in
       let* a = input ~shape:(s 1 1 1 4 3 2) () in
       let* b = input ~shape:(s 1 1 1 4 3 2) () in
       let* stacked = stack { Concat.Stack.axis = D } [ a; b ] in
       Graph_builder.return [ stacked ])
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  compare_graphs "stack output" ~x_shapes:[ s 1 1 1 4 3 2; s 1 1 1 4 3 2 ] g;
  let g =
    Graph_builder.build ~name:"permute_d" ~outputs:Fun.id
      (let open Graph_builder in
       let* a = input ~shape:(s 1 1 1 4 3 2) () in
       let* b = input ~shape:(s 1 1 1 4 3 2) () in
       let* stacked = stack { Concat.Stack.axis = D } [ a; b ] in
       let* p = permute (perm_of [ (D, H); (H, D) ]) stacked in
       Graph_builder.return [ p ])
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  compare_graphs "permute of the stack"
    ~x_shapes:[ s 1 1 1 4 3 2; s 1 1 1 4 3 2 ]
    g;
  [%expect
    {|
    stack output: node n0: axis D is outside the N/H/W/C dialect
    permute of the stack: node n0: axis D is outside the N/H/W/C dialect |}]

let%expect_test "the relabelled group's map verifies" =
  verified_graph "split attention" (sk_graph ~radix:2 ~h:4);
  [%expect
    {| split attention: 13 clusters: 1 proved (structural), 3 proved (structural) [sampled 8], 1 unproved (over max_rounds) [sampled 8], 8 vacuous |}]
