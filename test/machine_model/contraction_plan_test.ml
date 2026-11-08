module T = Model_test
module M = Machine_model.Mir_model
module R = Machine_model.Mir_model_route
module P = Ssa_ir.Ssa_contract_plan
module MP = Machine_ir.Mir_planning

let capabilities matrix =
  { Contraction.Select.Capabilities.matrix; vector = true }

let finalize matrix p =
  Err.payload (P.program ~capabilities:(capabilities matrix) p)
  |> Result.map_error (Fmt.str "%a" P.pp_error)

let route matrix =
  R.Route.Custom
    {
      R.Custom.name = "final SSA contract on generic oracle";
      plan = finalize matrix;
      exec = (fun ~sites:_ g -> Ok (R.generic g));
    }

let pipeline =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.zve32f128;
    }

let prepare matrix =
  let p = ref None in
  let graph = T.linear_fixture ~m:2 ~n:3 ~k:4 ~bias:false () in
  let model =
    match
      M.prepare ~route:(route matrix) ~pipeline ~contractions:true
        ~on_program:(fun _ ssa _ -> p := Some ssa)
        (T.bundle graph)
    with
    | Ok model -> model
    | Error rs -> Fmt.failwith "%a" Fmt.(list M.Refusal.pp) rs
  in
  (Option.get !p, model)

let summary model =
  match M.generic model with
  | [ g ] -> Option.get (Machine_ir.Mir_verify.Generic.program g).planning
  | _ -> assert false

let%test_unit "the observed final SSA and serialized compilation summary agree"
    =
  List.iter
    (fun matrix ->
      let p, m = prepare matrix in
      let summary = summary m in
      let want = P.choices p in
      assert (List.length want = 1);
      assert (
        List.map
          (fun (c : MP.Contraction_engine.t) -> (c.desc, c.engine))
          (MP.contraction_engines summary)
        = want);
      assert (summary.subject = Machine_lower.Mir_lower.subject p);
      assert (MP.of_string (MP.to_string summary) = Ok summary))
    [ false; true ]

let%test_unit "lowering rejects a summary that lost its SSA engine choice" =
  let p, m = prepare true in
  let s = summary m in
  let capabilities =
    List.filter
      (function MP.Capability.Contraction_engine _ -> false | _ -> true)
      s.capabilities
  in
  let tampered =
    MP.make ~subject:s.subject ~policy:s.policy ~schedule:s.schedule
      ~precision:s.precision ~lanes:s.lanes ~fma:s.fma ~capabilities
  in
  match
    Err.payload (Machine_lower.Mir_lower.program ~planning:(Some tampered) p)
  with
  | Error
      (Machine_lower.Mir_lower.Refusal.Planning `Planning_contraction_mismatch)
    ->
      ()
  | _ -> assert false

let%expect_test "selected vector and matrix plans keep generic results bitwise"
    =
  List.iter
    (fun matrix ->
      T.check ~route:(route matrix) ~pipeline ~contractions:true ~oracle:T.Fp32
        ~calls:1
        (if matrix then "matrix plan" else "vector plan")
        (fun () -> T.linear_fixture ~m:2 ~n:3 ~k:4 ~bias:false ()))
    [ false; true ];
  [%expect
    {|
    vector plan (1 invocations): bitwise
    matrix plan (1 invocations): bitwise |}]
