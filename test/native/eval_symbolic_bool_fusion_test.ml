(* P7.3: a Bool-declared intermediate with one pointwise consumer, run stored
   ([Kernel_eval.run]) and fused ([Fusion_plan.plan] + [run_plan]). The
   producer's [Nonzero_bool] conversion is part of the value as its consumer
   sees it, so eliminating the Bool buffer must not change the consumer's
   result: the subnormal 1e-40 (nonzero, so true) and NaN (true) are exactly
   the inputs a [Round_f32]-style boundary or a float comparison would get
   wrong, and both zeros stay false. *)

open Graph_ir

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:6

let x =
  Tensor.materialize shape (fun c ->
      [| 0.; -0.; 1e-40; Float.nan; Float.infinity; -3. |].(Dim.to_int
                                                              (Vec6.get c Axis.C)))

let build =
  Graph_builder.(
    build ~name:"bool_fusion" ~outputs:(fun r -> [ r ])
    @@
    let* x = input ~shape ~name:"x" () in
    let* mask = to_copy ~name:"mask" Pointwise.To_copy.Bool x in
    to_copy ~name:"out" Pointwise.To_copy.Float mask)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let kernel =
  Kernel_adapt.of_stage_program (Eval_symbolic.run build)
  |> Err.or_raise ~pp_error:Kernel_adapt.pp_error

let bind id =
  if Tensor_id.equal id (List.hd build.Graph.inputs) then Some x else None

let mask_id =
  match List.nth build.Graph.nodes 0 with
  | { Node.outputs = id :: _; _ } -> id
  | _ -> assert false

let out_id = List.hd build.Graph.outputs

let show name result id =
  match Tensor_id.Map.find_opt id result with
  | Some t -> Format.printf "%s = %a@." name Tensor.pp t
  | None -> Format.printf "%s not stored@." name

let%expect_test "Bool intermediate: stored run and fused run agree" =
  let stored =
    Kernel_eval.run kernel ~bind |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  show "stored mask" stored mask_id;
  show "stored out " stored out_id;
  let plan, decisions = Fusion_plan.plan kernel in
  List.iter (Format.printf "%a@." Fusion_plan.Decision.pp) decisions;
  let fused =
    Kernel_eval.run_plan plan ~bind
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  show "fused mask " fused mask_id;
  show "fused out  " fused out_id;
  [%expect
    {|
    stored mask = tensor bool [C=6] {0, 0, 1, 1, 1, 1}
    stored out  = tensor f32 [C=6] {0, 0, 1, 1, 1, 1}
    virtualize t1->t2
    fused mask  not stored
    fused out   = tensor f32 [C=6] {0, 0, 1, 1, 1, 1} |}]

let%expect_test "Stage_program.ground stores a Bool-declared stage as Bool" =
  let grounded =
    Stage_program.ground (Eval_symbolic.run build) ~bind:(fun id ->
        Option.get (bind id))
    |> Err.or_raise ~pp_error:Stage_program.pp_error
  in
  show "grounded mask" grounded mask_id;
  show "grounded out " grounded out_id;
  [%expect
    {|
    grounded mask = tensor bool [C=6] {0, 0, 1, 1, 1, 1}
    grounded out  = tensor f32 [C=6] {0, 0, 1, 1, 1, 1} |}]
