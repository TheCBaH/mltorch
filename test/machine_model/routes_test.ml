open Graph_ir
module F = Native_test.Graph_fixtures
module M = Machine_model.Mir_model
module T = Model_test

(* M12.2: every bundle on every Machine IR route — generic, then each target's
   selected program and its reference allocation — against the per-node
   reference and the direct C bundle made from the same SSA kernels. *)

let routes =
  M.Route.
    [
      Generic;
      Aarch64 M.Stage.Selected;
      Aarch64 M.Stage.Allocated;
      Aarch64 M.Stage.Realized;
      X86_64 M.Stage.Selected;
      X86_64 M.Stage.Allocated;
      X86_64 M.Stage.Realized;
    ]

let with_dir f =
  let dir = Loop_c_exec.Proc.temp_dir "machine_model_routes" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () -> f dir)

(* The direct C bundle of the same exact-pipeline SSA kernels. *)
let c_outputs b ~constants ~inputs =
  with_dir (fun dir ->
      let p =
        Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
          (Loop_c_exec.Host.prepare
             ~kernel:(Ssa_backends.c ~pipeline:Ssa_backends.Pipeline.Exact)
             ~dir b ~constants:(fun id -> List.assoc_opt id constants))
      in
      match
        Err.payload
          (Loop_c_exec.Host.run p ~bind:(fun id -> List.assoc_opt id inputs))
      with
      | Ok outs -> Ok outs
      | Error e -> Error (Fmt.str "%a" Loop_c_exec.Host.pp_error e))

let compare_all name ?(constant = T.values ~salt:3)
    ?(input = fun i -> T.values ~salt:(i + 1)) g =
  let b =
    Loop_ir.Loop_bundle.build ~config:Loop_ir.Loop_bundle_c.default_config g
    |> Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
  in
  let constants =
    List.map
      (fun id -> (id, constant (T.sig_of g id)))
      b.Loop_ir.Loop_bundle.constants
  in
  let inputs =
    List.mapi
      (fun i id -> (id, input i (T.sig_of g id)))
      b.Loop_ir.Loop_bundle.inputs
  in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~constants ~inputs)
  in
  let same outs =
    List.for_all2
      (fun id t -> T.bits t = T.bits (Tensor_id.Map.find id reference))
      g.Graph.outputs outs
  in
  let verdict = function
    | Ok outs -> if same outs then "bitwise" else "DIFFERS"
    | Error s -> s
  in
  let on route =
    match M.prepare ~route ~pipeline:Ssa_backends.Pipeline.Exact b with
    | Error rs ->
        Fmt.str "refused: %a" Fmt.(list ~sep:(any "; ") M.Refusal.pp) rs
    | Ok m -> (
        match
          M.Context.create m ~constants:(fun id -> List.assoc_opt id constants)
        with
        | Error s -> Fmt.str "%a" M.Stop.pp s
        | Ok cx ->
            verdict
              (Result.map_error (Fmt.str "%a" M.Stop.pp)
                 (M.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs)))
        )
  in
  Fmt.pr "%s:@." name;
  List.iter (fun r -> Fmt.pr "  %s: %s@." (M.Route.name r) (on r)) routes;
  Fmt.pr "  direct C: %s@." (verdict (c_outputs b ~constants ~inputs))

let%expect_test "conv, batch norm, relu; batched matmul; linear" =
  compare_all "chain" ~constant:T.positive (F.chain ());
  compare_all "bmm 2x(5x7 . 7x3)"
    (F.build "bmm"
       Graph_builder.(
         let* a = input ~shape:(F.s 1 1 1 2 5 7) () in
         let* b = input ~shape:(F.s 1 1 1 2 7 3) () in
         bmm a b));
  compare_all "linear 8 -> 4"
    (F.build "linear"
       Graph_builder.(
         let* x = input ~shape:(F.s 1 1 1 1 1 8) () in
         let* w = constant ~shape:(F.s 4 1 1 1 1 8) () in
         linear { Linear.Linear.in_features = Dim.extent 8 } ~x ~weight:w ()));
  [%expect
    {|
    chain:
      generic: bitwise
      aarch64 selected: bitwise
      aarch64 allocated: bitwise
      aarch64 realized: bitwise
      x86_64 selected: bitwise
      x86_64 allocated: bitwise
      x86_64 realized: bitwise
      direct C: bitwise
    bmm 2x(5x7 . 7x3):
      generic: bitwise
      aarch64 selected: bitwise
      aarch64 allocated: bitwise
      aarch64 realized: bitwise
      x86_64 selected: bitwise
      x86_64 allocated: bitwise
      x86_64 realized: bitwise
      direct C: bitwise
    linear 8 -> 4:
      generic: bitwise
      aarch64 selected: bitwise
      aarch64 allocated: bitwise
      aarch64 realized: bitwise
      x86_64 selected: bitwise
      x86_64 allocated: bitwise
      x86_64 realized: bitwise
      direct C: bitwise |}]

let sdpa () =
  F.build "sdpa_mask"
    Graph_builder.(
      let* q = input ~shape:(F.s 1 1 2 3 4 5) () in
      let* k = input ~shape:(F.s 1 1 2 3 6 5) () in
      let* v = input ~shape:(F.s 1 1 2 3 6 5) () in
      let* m = input ~shape:(F.s 1 1 2 3 4 6) () in
      sdpa
        { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
        ~query:q ~key:k ~value:v ~mask:m ())

let%expect_test "attention: some cells, a whole row, every cell masked" =
  List.iter
    (fun (name, masked) ->
      compare_all ("sdpa, " ^ name) (sdpa ()) ~input:(fun i sg ->
          if i = 3 then
            Tensor.materialize sg.Tensor_sig.shape (fun c ->
                if masked c then neg_infinity else 0.)
          else T.values ~salt:(i + 1) sg))
    [
      ("some cells", fun c -> (Vec6.get c Axis.C :> int) mod 4 = 1);
      ("a whole row", fun c -> (Vec6.get c Axis.W :> int) = 1);
      ("every cell", fun _ -> true);
    ];
  compare_all "softmax over C"
    (F.build "softmax"
       Graph_builder.(
         let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
         softmax { Reduce.Softmax.axis = Axis.C } x));
  [%expect
    {|
    sdpa, some cells:
      generic: bitwise
      aarch64 selected: bitwise
      aarch64 allocated: bitwise
      aarch64 realized: bitwise
      x86_64 selected: bitwise
      x86_64 allocated: bitwise
      x86_64 realized: bitwise
      direct C: bitwise
    sdpa, a whole row:
      generic: bitwise
      aarch64 selected: bitwise
      aarch64 allocated: bitwise
      aarch64 realized: bitwise
      x86_64 selected: bitwise
      x86_64 allocated: bitwise
      x86_64 realized: bitwise
      direct C: bitwise
    sdpa, every cell:
      generic: bitwise
      aarch64 selected: bitwise
      aarch64 allocated: bitwise
      aarch64 realized: bitwise
      x86_64 selected: bitwise
      x86_64 allocated: bitwise
      x86_64 realized: bitwise
      direct C: bitwise
    softmax over C:
      generic: bitwise
      aarch64 selected: bitwise
      aarch64 allocated: bitwise
      aarch64 realized: bitwise
      x86_64 selected: bitwise
      x86_64 allocated: bitwise
      x86_64 realized: bitwise
      direct C: bitwise |}]

let%expect_test "the first failing invocation on every route" =
  let g = T.gather () in
  let b = T.bundle g in
  let self_id, index_id =
    match b.Loop_ir.Loop_bundle.inputs with
    | [ s; i ] -> (s, i)
    | _ -> assert false
  in
  let inputs =
    [
      (self_id, T.values ~salt:1 (T.sig_of g self_id));
      ( index_id,
        Tensor.materialize_i64 (F.s 1 1 1 1 1 2) (fun c ->
            if (Vec6.get c Axis.C :> int) = 0 then 1L else 3L) );
    ]
  in
  List.iter
    (fun route ->
      let m =
        Result.get_ok (M.prepare ~route ~pipeline:Ssa_backends.Pipeline.Exact b)
      in
      let cx = Result.get_ok (M.Context.create m ~constants:(fun _ -> None)) in
      Fmt.pr "%s: %s@." (M.Route.name route)
        (match
           M.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs)
         with
        | Ok _ -> "ran"
        | Error s -> Fmt.str "%a" M.Stop.pp s))
    routes;
  [%expect
    {|
    generic: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64)
    aarch64 selected: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64)
    aarch64 allocated: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64)
    aarch64 realized: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64)
    x86_64 selected: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64)
    x86_64 allocated: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64)
    x86_64 realized: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64) |}]

(* A table the bundle declares complete: a site-bearing failure it does not
   list takes the sentinel, and a record that reaches it is a defect, never a
   failure. Here the failure is reachable — exactly what the declaration rules
   out — so the decoder's defect is what must show. *)
let%expect_test "an unlisted site that is reached decodes as a defect" =
  let p =
    Machine_source_test.Mir_local_test.build (fun bld ->
        let module B = Ssa_ir.Ssa_builder in
        let module L = Machine_source_test.Mir_local_test in
        let h = B.local_alloc ~var:L.local_var bld ~slots:2L in
        B.local_write bld h (L.idx bld 0) (B.f64 bld 1.);
        L.store_at bld (L.idx bld 0) (B.local_read bld h (L.idx bld 2)))
  in
  let planning =
    Machine_ir.Mir_planning.make
      ~subject:(Machine_lower.Mir_lower.subject p)
      ~policy:"reference_f64" ~schedule:"scalar"
      ~precision:Machine_ir.Mir_planning.Precision.F64
      ~lanes:(Machine_ir.Mir_type.Lanes.of_int 1)
      ~fma:Machine_ir.Mir_planning.Fma.Forbidden ~capabilities:[]
  in
  let g =
    (Err.or_raise ~pp_error:Machine_lower.Mir_lower.Refusal.pp
       (Machine_lower.Mir_lower.program ~planning:(Some planning) p))
      .Machine_lower.Mir_lower.program
  in
  List.iter
    (fun route ->
      match Machine_model.Mir_model_route.exec route ~sites:[||] g with
      | Error e -> Fmt.pr "%s: %s@." (M.Route.name route) e
      | Ok exec ->
          let memory = Machine_interp.Mir_memory.create () in
          let binding =
            Result.get_ok
              (exec.Machine_model.Mir_model_route.Exec.instantiate memory
                 ~shared:(fun _ -> None))
          in
          Fmt.pr "%s: %a@." (M.Route.name route)
            Machine_ir.Mir_compare.Difference.pp_status
            (exec.Machine_model.Mir_model_route.Exec.run memory binding
               ~invocation:0l))
    routes;
  [%expect
    {|
    generic: failure(unbound_local(#0))
    aarch64 selected: defect(sentinel_site)
    aarch64 allocated: defect(sentinel_site)
    aarch64 realized: defect(sentinel_site)
    x86_64 selected: defect(sentinel_site)
    x86_64 allocated: defect(sentinel_site)
    x86_64 realized: defect(sentinel_site) |}]
