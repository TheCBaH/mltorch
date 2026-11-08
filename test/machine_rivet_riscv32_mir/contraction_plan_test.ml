module M = Machine_model.Mir_model
module T = Machine_model_test.Model_test
module G = Machine_model_test.Contraction_plan_test
module Q = Machine_rivet_riscv32_mir.Rv32_mir_route

let artifact matrix =
  let _, model = G.prepare matrix in
  let g = match M.generic model with [ g ] -> g | _ -> assert false in
  match Q.publish ~allocation:Q.Allocation.Scanned ~sites:[||] g with
  | Ok p -> p.Q.artifact
  | Error e -> failwith e

let%test_unit "a selected vector plan does not ask for a matrix realization" =
  let a = artifact false in
  let helpers =
    Q.helper_modules ~other:(fun _ -> failwith "matrix reselection") a
  in
  assert (Result.is_ok helpers)

let%test_unit "a selected matrix plan is refused on a vector-only helper route"
    =
  match Q.helper_modules (artifact true) with
  | Error _ -> ()
  | Ok _ -> assert false

let%expect_test "an explicit vector plan runs under qemu" =
  let r =
    match Q.route () with
    | M.Route.Custom r -> M.Route.Custom { r with plan = G.finalize false }
    | _ -> assert false
  in
  T.check ~route:r ~pipeline:G.pipeline ~contractions:true ~oracle:T.Fp32
    ~calls:1 "selected vector" (fun () ->
      T.linear_fixture ~m:2 ~n:3 ~k:4 ~bias:false ());
  [%expect {| selected vector (1 invocations): bitwise |}]
