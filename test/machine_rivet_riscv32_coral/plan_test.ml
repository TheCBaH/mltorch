module T = Machine_model_test.Model_test
module M = Machine_model.Mir_model
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral
module P = Ssa_ir.Ssa_contract_plan

let graph () = T.linear_fixture ~m:2 ~n:3 ~k:4 ~bias:false ()
let pipeline = Machine_model_test.Contraction_plan_test.pipeline

let config configuration =
  {
    C.Config.dir = Sys.getenv "CORALNPU_DATA";
    configuration;
    deadline = Deadline.seconds;
    flags = [];
  }

let%expect_test
    "Coral records its engine before lowering and preserves the oracle" =
  List.iter
    (fun (name, configuration, engine, want) ->
      let observed = ref [] in
      T.check
        ~route:(C.route ?engine (config configuration))
        ~pipeline ~contractions:true ~oracle:T.Fp32 ~calls:1 name graph
        ~on_program:(fun _ p _ -> observed := P.choices p @ !observed);
      assert (List.length !observed = 1);
      assert (List.for_all (fun (_, e) -> e = want) !observed))
    [
      ( "VME matrix plan",
        "vme-core-mini-axi",
        None,
        Contraction.Select.Engine.Zvt );
      ( "VME vector plan",
        "vme-core-mini-axi",
        Some false,
        Contraction.Select.Engine.Rvv );
      ( "MPACT vector plan",
        "mpact-m3",
        Some false,
        Contraction.Select.Engine.Rvv );
    ];
  [%expect
    {|
    VME matrix plan (1 invocations): bitwise
    VME vector plan (1 invocations): bitwise
    MPACT vector plan (1 invocations): bitwise |}]

(* Counterfactual for the engine-planning change: the legacy route makes the
   same engine decision after lowering. Recording it in SSA must preserve
   every emitted byte and the entry-call cycles, including tails and bias. *)
let%test_unit "recording engines preserves legacy ELFs and RTL cycles" =
  List.iter
    (fun (name, fixture) ->
      let g = fixture () in
      List.iter
        (fun engine ->
          let mode = if engine then "matrix" else "vector" in
          let measure legacy =
            let cycles = ref [] and elfs = ref [] in
            let on_run (r : C.Run.t) = cycles := r.mcycle :: !cycles in
            let on_elf bytes = elfs := bytes :: !elfs in
            let route =
              C.route ~engine ~on_run ~on_elf (config "vme-core-mini-axi")
            in
            let route =
              match (legacy, route) with
              | true, M.Route.Custom r ->
                  M.Route.Custom { r with plan = Result.ok }
              | _ -> route
            in
            T.check ~route ~pipeline ~contractions:true ~oracle:T.Fp32
              ~same:T.same_portable ~input:T.adversarial_input
              ~constant:(fun sg -> T.adversarial ~salt:3 sg)
              ~calls:1
              (name ^ ", " ^ mode ^ if legacy then ", legacy" else ", planned")
              (fun () -> g);
            assert (List.length !cycles = 1 && List.length !elfs = 1);
            (List.hd !cycles, List.hd !elfs)
          in
          let before, elf_before = measure true in
          let after, elf_after = measure false in
          assert (elf_before = elf_after);
          assert (before = after);
          Fmt.pr "  identical ELF, %Ld cycles@." after)
        [ false; true ])
    T.contraction_fixtures
