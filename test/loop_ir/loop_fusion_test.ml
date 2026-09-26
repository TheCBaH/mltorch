open Loop_ir
open Loop_fixtures
open Loop_programs

(* Virtual edges: the consumer's body is [Kernel_elab.elaborate]'s tree, so the
   producer's buffer is absent from the program and its result conversion
   survives inside the consumer. *)

let bind id =
  if Tensor_id.equal id (tid 0) then
    Some (f32_tensor (shape_w 2) (fun _ -> two24))
  else None

let report name plan =
  let program =
    match Err.payload (Loop_lower.lower plan) with
    | Ok p -> p
    | Error (`Unsupported u) -> Fmt.failwith "refused: %a" Loop_unsupported.pp u
  in
  let ids =
    List.map
      (fun (b : Loop_buffer.t) -> Fmt.str "%a" Tensor_id.pp b.Loop_buffer.id)
      program.Loop_program.buffers
  in
  let result =
    match Err.payload (Loop_interp.run program ~bind) with
    | Ok m ->
        let t = Tensor_id.Map.find (tid 2) m in
        Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0)
    | Error _ -> nan
  in
  Fmt.pr "%s: %a; buffers %s; t2 = %.1f@." name Loop_check.pp_verdict
    (Loop_check.run plan ~bind)
    (String.concat "," ids) result

let%expect_test "an eliminated producer has no buffer and keeps its round" =
  let k = chain ~outputs:[ tid 2 ] in
  report "default" (Fusion_plan.default k);
  let plan, _ = Fusion_plan.plan k in
  report "fused" plan;
  Fmt.pr "without the inner round it would be %.1f@." (two24 +. 2.);
  [%expect
    {|
    default: agree; buffers t0,t1,t2; t2 = 16777216.0
    fused: agree; buffers t0,t2; t2 = 16777216.0
    without the inner round it would be 16777218.0 |}]

let%expect_test
    "an externally live producer is virtual for its consumer and stored" =
  let k = chain ~outputs:[ tid 1; tid 2 ] in
  let plan, _ = Fusion_plan.plan k in
  report "fused, also stored" plan;
  [%expect {| fused, also stored: agree; buffers t0,t1,t2; t2 = 16777216.0 |}]
