open Loop_ir
open Loop_fixtures
open Loop_programs

(* Region groups: one shared recurrence per canonical key and one store per
   member, each with its own conversion and format. *)

let show name kernel =
  Fmt.pr "%s: %a@." name Loop_check.pp_verdict
    (Loop_check.run (Fusion_plan.default kernel) ~bind:bind_none)

let%expect_test "each member is converted and stored by its own value" =
  (* [read - 2] is -1, 0, 1, 2 across W: true, false, true, true. *)
  show "bool member"
    (bool_group_kernel ~second:(fun read ->
         Expr.Value.sub read (Expr.Value.const 2.)));
  [%expect {| bool member: agree |}]

(* Below binary32's smallest subnormal: [Nonzero_bool] on the working value says
   true, and storing to f32 first says false. Only a body in that range tells a
   member's own conversion from another's or from the store's. *)
let%expect_test "a working value below binary32's range is still true" =
  let kernel =
    bool_group_kernel ~second:(fun read ->
        Expr.Value.mul read (Expr.Value.const 1e-50))
  in
  show "1e-50" kernel;
  (match Err.payload (Kernel_eval.run kernel ~bind:bind_none) with
  | Ok m ->
      let t = Tensor_id.Map.find (tid 2) m in
      Fmt.pr "reference reads: %a@."
        Fmt.(list ~sep:(any " ") float)
        (List.init 4 (fun w ->
             Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w ~c:0)))
  | Error _ -> ());
  [%expect {|
    1e-50: agree
    reference reads: 1 1 1 1 |}]
