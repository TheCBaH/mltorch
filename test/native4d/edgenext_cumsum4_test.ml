(* Gate 7 item 4 (Native4D half): the Native4D twin of [edgenext_cumsum_test.
   ml] -- the same real EdgeNeXt subgraph (`_to_copy.default(dtype=BOOL)` ->
   `bitwise_not.default` -> two `cumsum.default` nodes, one per spatial axis,
   both reading the SAME [Bitwise_not] output, both with an explicit
   Float32 [dtype] argument) built on [Native4d.Builder]/[Eval_direct4]
   instead of Native's own [Graph_builder]/[Eval_direct]. [Eval_op4]'s own
   [Cumsum4] dispatch (`eval_op4.ml:151`) reuses the SAME [Reduce.Cumsum.
   Compute] functor Native's [Cumsum] dispatch does, so this fixture expects
   the identical "no new dispatch code needed" outcome [edgenext_cumsum_test.
   ml]'s own P7.2 note already found for Native -- proven here, not assumed,
   since Native4D is a distinct dialect front end with its own builder/eval
   modules ([to_copy_bool_test.ml]/[bitwise_not_bool_test.ml] show
   Native4D's own Bool dispatch was a real per-op landing, not automatic). *)

open Native4d

let shape4 = Shape4.of_ints ~n:1 ~h:2 ~w:4 ~c:1

let%expect_test
    "direct4: To_copy(Bool) -> Bitwise_not -> two Float32 cumsum4 (H and W) \
     sharing one Bool source" =
  let g =
    Builder.build
      ~outputs:(fun (h, w) -> [ h; w ])
      (let open Builder in
       let* x = input ~shape:shape4 () in
       let* mask = to_copy Pointwise.To_copy.Bool x in
       let* inv = bitwise_not mask in
       let* cum_h = cumsum4 { Ops4_cumsum.Cumsum4.axis = Axis4.H } inv in
       let* cum_w = cumsum4 { Ops4_cumsum.Cumsum4.axis = Axis4.W } inv in
       return (cum_h, cum_w))
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let row = [| [| 0.; 1.; 0.; 2. |]; [| 3.; 0.; 0.; 5. |] |] in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape4) (fun c ->
        row.(Dim.to_int (Vec6.get c Axis.H)).(Dim.to_int (Vec6.get c Axis.W)))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  (* [to_copy] is node 0, [bitwise_not] is node 1, mirroring
     [edgenext_cumsum_test.ml]'s own by-index read -- there is no
     [id_of_name] helper for these names. *)
  let node_output i =
    match List.nth g.Graph.Graph.nodes i with
    | { Graph_common.Node.outputs = id :: _; _ } -> id
    | _ -> assert false
  in
  let mask = Tensor_id.Map.find (node_output 0) env in
  let inv = Tensor_id.Map.find (node_output 1) env in
  let outs =
    List.map (fun oid -> Tensor_id.Map.find oid env) g.Graph.Graph.outputs
  in
  Fmt.pr "mask = %a@.inv  = %a@." Tensor.pp mask Tensor.pp inv;
  List.iter (fun t -> Fmt.pr "%a@." Tensor.pp t) outs;
  (* Same values [edgenext_cumsum_test.ml] already hand-verified: x =
     {{0,1,0,2},{3,0,0,5}} -> mask (genuine Bool) = {0,1,0,1,1,0,0,1} -> inv
     = {1,0,1,0,0,1,1,0} -> cum_h (down H) = {1,0,1,0,1,1,2,0} -> cum_w
     (across W) = {1,1,2,2,0,1,2,2}, both F32 outputs. *)
  [%expect
    {|
    mask = tensor bool [H=2 W=4 C=1] {0, 1, 0, 1, 1, 0, 0, 1}
    inv  = tensor bool [H=2 W=4 C=1] {1, 0, 1, 0, 0, 1, 1, 0}
    tensor f32 [H=2 W=4 C=1] {1, 0, 1, 0, 1, 1, 2, 0}
    tensor f32 [H=2 W=4 C=1] {1, 1, 2, 2, 0, 1, 2, 2} |}]
