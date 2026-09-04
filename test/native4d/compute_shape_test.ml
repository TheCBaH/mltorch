(* The shape/gather family of Native4D Direct-evaluation tests, split out of
   compute_test.ml under the tracked file-size ceiling
   (scripts/check-file-size.sh). Same two properties that file's own header
   describes -- hand values against numbers worked out by hand, not against
   Native, since both sides would instantiate the same [Compute (S)] functor. *)

open Native4d
open Compute_fixtures

let%expect_test "direct4: permute4 swaps H and W" =
  let shape = s4 ~n:1 ~h:2 ~w:3 ~c:1 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       permute4 (Ops4.Permute4.of_fn (function H -> W | W -> H | a -> a)) x)
  in
  let out = single g ~inputs:[ (List.hd g.Graph.Graph.inputs, seq shape) ] () in
  let (Tensor.Tensor tt) = out in
  Format.printf "in  [H=2 W=3]: %a@." pp_values (values (seq shape));
  Format.printf "out %a: %a@." Vec6.pp_shape tt.Tensor.shape pp_values
    (values out);
  [%expect
    {|
    in  [H=2 W=3]: [0 1 2 3 4 5]
    out [H=3 W=2 C=1]: [0 3 1 4 2 5] |}]

(* Hand values, both modes. What this checks that the per-op table below cannot:
   the dialect's [Axis4.t] keys reach [Pad.Pad.Compute] as the axes they name.
   Swap H and W in [Graph_shape4.pad_params] and both modes here go red, while
   the direct-versus-symbolic table stays entirely green — both of its sides
   read the same adapter. *)
let%expect_test "direct4: pad4, constant and reflect" =
  let shape = s4 ~n:1 ~h:2 ~w:3 ~c:1 in
  let pad mode pads =
    let g =
      build
        ~outputs:(fun o -> [ o ])
        (let open Builder in
         let* x = input ~shape () in
         pad4 { Ops4.Pad4.pads; mode } x)
    in
    let out =
      single g ~inputs:[ (List.hd g.Graph.Graph.inputs, seq shape) ] ()
    in
    let (Tensor.Tensor tt) = out in
    Format.printf "out %a: %a@." Vec6.pp_shape tt.Tensor.shape pp_values
      (values out)
  in
  Format.printf "in  [H=2 W=3]: %a@." pp_values (values (seq shape));
  (* One column of fill on each side of W, none on H: rows [9 0 1 2 9] and
     [9 3 4 5 9]. *)
  pad (Pad.Pad.Constant 9.) [ (Axis4.W, { Pad.Pad.before = 1; after = 1 }) ];
  (* Mirror about the boundary element, which is NOT repeated: row [0 1 2]
     extended left by one is [1 0 1 2], and by two on the right is [.. 1 0]. *)
  pad Pad.Pad.Reflect [ (Axis4.W, { Pad.Pad.before = 1; after = 2 }) ];
  (* H is the OUTER axis here, so a mirror on it moves whole rows: [0 1 2] and
     [3 4 5] become [3 4 5], [0 1 2], [3 4 5]. A pad that silently used W would
     print the same extents as a W-mirror only if H and W matched, which is why
     this fixture is 2x3. *)
  pad Pad.Pad.Reflect [ (Axis4.H, { Pad.Pad.before = 1; after = 0 }) ];
  (* Cropping: a negative entry removes the last row, and nothing is
     synthesized, so no fill value can appear. *)
  pad (Pad.Pad.Constant 9.) [ (Axis4.H, { Pad.Pad.before = 0; after = -1 }) ];
  [%expect
    {|
    in  [H=2 W=3]: [0 1 2 3 4 5]
    out [H=2 W=5 C=1]: [9 0 1 2 9 9 3 4 5 9]
    out [H=2 W=6 C=1]: [1 0 1 2 1 0 4 3 4 5 4 3]
    out [H=3 W=3 C=1]: [3 4 5 0 1 2 3 4 5]
    out [W=3 C=1]: [0 1 2] |}]

(* Hand values, not Native-as-oracle: both sides would instantiate the same
   [Split.Unbind.Compute] functor, so agreement would prove the adapter and the
   staging rather than the arithmetic.

   Unbinding C on [N=1 H=1 W=2 C=3] drops the innermost axis, so H shifts onto W
   and W onto C: each slice is [W=1 C=2], holding one column. Every ordinal is
   printed, since that is what a single-output test cannot see. *)
let%expect_test "direct4: unbind takes one slice per coordinate" =
  let shape = s4 ~n:1 ~h:1 ~w:2 ~c:3 in
  let g =
    build ~outputs:Fun.id
      (let open Builder in
       let* x = input ~shape () in
       unbind Axis4.C x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.W) * 10)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  let env = run_direct g ~inputs:[ (List.hd g.Graph.Graph.inputs, x) ] in
  List.iteri
    (fun i out ->
      Format.printf "out%d = %a@." i Tensor.pp (Tensor_id.Map.find out env))
    g.Graph.Graph.outputs;
  [%expect
    {|
    out0 = tensor f32 [C=2] {0, 10}
    out1 = tensor f32 [C=2] {1, 11}
    out2 = tensor f32 [C=2] {2, 12} |}]

(* Hand values, not Native-as-oracle: both sides would instantiate the same
   [Split.Select.Compute] functor, so agreement would prove the adapter and
   the staging rather than the arithmetic.

   The same shape and value formula [unbind]'s own oracle test above uses, so
   selecting index 1 along C reads the same one-wide window through the same
   repacking as that test's out1 -- checkable against it by eye. *)
let%expect_test "direct4: select4 reads one slice at the chosen index" =
  let shape = s4 ~n:1 ~h:1 ~w:2 ~c:3 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       select4 { Ops4.Select4.axis = Axis4.C; index = 1 } x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.W) * 10)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  let env = run_direct g ~inputs:[ (List.hd g.Graph.Graph.inputs, x) ] in
  Format.printf "%a@." Tensor.pp
    (Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env);
  [%expect {| tensor f32 [C=2] {1, 11} |}]

(* Unequal sizes [1;3] on a KEPT axis, unlike [unbind] above: a wrong offset
   for the second piece would read starting from W=0 instead of W=1, so this
   would print {0, 1, 2} for out1 instead of the correct {1, 2, 3}. *)
let%expect_test "direct4: split_with_sizes keeps the axis, sliced by window" =
  let shape = s4 ~n:1 ~h:1 ~w:4 ~c:1 in
  let g =
    build ~outputs:Fun.id
      (let open Builder in
       let* x = input ~shape () in
       split_with_sizes4 Axis4.W [ 1; 3 ] x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.W)))
  in
  let env = run_direct g ~inputs:[ (List.hd g.Graph.Graph.inputs, x) ] in
  List.iteri
    (fun i out ->
      Format.printf "out%d = %a@." i Tensor.pp (Tensor_id.Map.find out env))
    g.Graph.Graph.outputs;
  [%expect
    {|
    out0 = tensor f32 [C=1] {0}
    out1 = tensor f32 [W=3 C=1] {1, 2, 3} |}]

(* Hand values, not Native-as-oracle: both sides would instantiate the same
   [Index_tensor.Index_tensor.Compute] functor, so agreement would prove the
   adapter and the staging rather than the arithmetic.

   The gathered axis is W, deliberately NOT C -- the position [Index_tensor]'s
   own [Compute.pixel] doc warns is the one a coordinate inherited from [out]
   directly would misread. [self]'s other axes (H, C) carry nonzero values that
   pass straight through, so a wrong axis map would show up as a wrong VALUE at
   an otherwise-correct-looking position, not merely a wrong shape. *)
let%expect_test "direct4: index_tensor4 gathers along a non-C axis" =
  let self_shape = s4 ~n:1 ~h:2 ~w:3 ~c:2 in
  let index_shape = s4 ~n:1 ~h:1 ~w:1 ~c:2 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* self = input ~shape:self_shape () in
       let* index =
         input ~shape:index_shape ~fmt:(Payload.Fmt Payload.I64) ()
       in
       index_tensor4 { Ops4.IndexTensor4.axis = Axis4.W } ~self ~index)
  in
  let self_t =
    Tensor.materialize (Shape4.to_vec6 self_shape) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.H) * 100)
          + (Dim.to_int (Vec6.get c Axis.W) * 10)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  let index_t =
    Tensor.materialize_i64 (Shape4.to_vec6 index_shape) (fun c ->
        Int64.of_int (2 - Dim.to_int (Vec6.get c Axis.C)))
  in
  let self_id, index_id =
    match g.Graph.Graph.inputs with
    | [ a; b ] -> (a, b)
    | _ -> invalid_arg "expected two inputs"
  in
  let env = run_direct g ~inputs:[ (self_id, self_t); (index_id, index_t) ] in
  Format.printf "%a@." Tensor.pp
    (Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env);
  [%expect {| tensor f32 [H=2 W=2 C=2] {20, 21, 10, 11, 120, 121, 110, 111} |}]
