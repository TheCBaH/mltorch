(* The Native4D Symbolic half: the Native4D twin of
   [eval_symbolic_edgenext_cumsum_test.ml] -- [Eval_symbolic4.run] produces
   the SAME [Stage_program.t] type Native's own [Eval_symbolic.run] does (see
   that module's own top comment: "into the SAME [Stage_program.t] a Native
   graph produces"), so Native's [Kernel_adapt.of_stage_program] applies to a
   Native4D-built program unchanged -- no Native4D-specific Kernel_adapt
   exists, and none is needed. [inv] (the [Bitwise_not] output) is shared by
   both [cumsum4] consumers, so it is a genuine stored [Kernel.Value.t],
   declared [Bool]; Bool storage is admitted at the Kernel boundary, so both
   Float32 cumsums must equal the Direct route's hand-verified prefix sums. *)

open Native4d

let shape4 = Shape4.of_ints ~n:1 ~h:2 ~w:4 ~c:1

let build =
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

let row = [| [| 0.; 1.; 0.; 2. |]; [| 3.; 0.; 0.; 5. |] |]

let x =
  Tensor.materialize (Shape4.to_vec6 shape4) (fun c ->
      row.(Dim.to_int (Vec6.get c Axis.H)).(Dim.to_int (Vec6.get c Axis.W)))

let node_output i =
  match List.nth build.Graph.Graph.nodes i with
  | { Graph_common.Node.outputs = id :: _; _ } -> id
  | _ -> assert false

let%expect_test
    "Native4D Symbolic -> Native's Kernel: EdgeNeXt's shared Bool [inv] \
     intermediate is stored as Bool and both Float32 cumsum4s match the Direct \
     route" =
  let kernel =
    Kernel_adapt.of_stage_program (Eval_symbolic4.run build)
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun id ->
        if Tensor_id.equal id (List.hd build.Graph.Graph.inputs) then Some x
        else None)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  let show name id =
    match Tensor_id.Map.find_opt id result with
    | Some t -> Fmt.pr "%s = %a@." name Tensor.pp t
    | None -> Fmt.pr "%s MISSING@." name
  in
  show "mask" (node_output 0);
  show "inv " (node_output 1);
  List.iter (show "out ") build.Graph.Graph.outputs;
  [%expect
    {|
    mask = tensor bool [H=2 W=4 C=1] {0, 1, 0, 1, 1, 0, 0, 1}
    inv  = tensor bool [H=2 W=4 C=1] {1, 0, 1, 0, 0, 1, 1, 0}
    out  = tensor f32 [H=2 W=4 C=1] {1, 0, 1, 0, 1, 1, 2, 0}
    out  = tensor f32 [H=2 W=4 C=1] {1, 1, 2, 2, 0, 1, 2, 2} |}]
