open Lowering_key

let variants : variant list =
  let routes =
    [
      Route.default;
      Route.Loop (Loop_spec.make []);
      Route.Loop
        (Loop_spec.make ~numerics:Loop_ir.Loop_numerics.Reference_f64
           [ Loop_ir.Loop_opt.Pass.Cse; Fold ]);
      Route.Ssa Ssa_spec.default;
      Route.Ssa
        {
          frontend = Direct;
          pipeline = Exact;
          numerics = Some Loop_ir.Loop_numerics.Simd_fp32_relaxed;
        };
    ]
  in
  List.concat_map
    (fun f -> List.map (fun v -> v) (List.map f routes))
    [
      (fun r -> (C r : variant));
      (fun r -> (Js r : variant));
      (fun r -> (Wat r : variant));
    ]
  @ List.map (fun t -> (Asm t : variant)) Target.all
  @ List.map (fun v -> (Coral v : variant)) Coral_view.all
  @ List.concat_map
      (fun target ->
        List.map
          (fun stage -> (Machine_ir { Mir_spec.target; stage } : variant))
          Mir_stage.all)
      Target.all
  @ [ Loop Loop_spec.default; Loop (Loop_spec.make []) ]
  @ [ Ssa Ssa_spec.default ]

let%expect_test "every variant round trips through its spelling" =
  List.iter
    (fun v ->
      match of_name (name v) with
      | Ok v' when v' = v -> ()
      | Ok v' -> Fmt.pr "MISMATCH %s -> %s@." (name v) (name v')
      | Error e -> Fmt.pr "ERROR %s: %a@." (name v) Parse_error.pp e)
    variants;
  Fmt.pr "%d checked@." (List.length variants);
  [%expect {| 46 checked |}]

let%expect_test "the bare family name is its default" =
  List.iter
    (fun f ->
      let v = default f in
      Fmt.pr "%s -> %s (%b)@." (family_name f) (name v)
        (of_name (family_name f) = Ok v))
    families;
  [%expect
    {|
    asm -> asm (true)
    c -> c (true)
    coral -> coral (true)
    js -> js (true)
    loop -> loop (true)
    machine_ir -> machine_ir (true)
    ssa -> ssa (true)
    wat -> wat (true)
    |}]

let show s =
  match of_name s with
  | Ok v -> Fmt.pr "%-34s -> %s@." s (name v)
  | Error e -> Fmt.pr "%-34s !! %a@." s Parse_error.pp e

let%expect_test "spellings, normalisation and rejection" =
  List.iter show
    [
      "c:ssa/planned/simd_fp32_relaxed";
      "c:ssa/simd_fp32_relaxed/planned";
      "c:ssa";
      "c:ssa/structured/planned";
      "c:loop";
      "c:loop/raw";
      "asm:x86_64";
      "asm:riscv32";
      "loop:raw";
      "loop:fold,cse";
      "loop:cse,fold,cse";
      "loop:all";
      "machine_ir:riscv32/selected";
      "machine_ir:generic/aarch64";
      "ssa:direct/exact";
      "coral:zvt_kernel";
      "nope";
      "c:";
      "c:ssa//exact";
      "c:ssa/exact/representation";
      "c:vector";
      "loop:fold,nope";
      "loop:raw/all";
      "asm:x86_64/x86_64";
      "coral:estimate/estimate";
    ];
  [%expect
    {|
    c:ssa/planned/simd_fp32_relaxed    -> c:ssa/simd_fp32_relaxed
    c:ssa/simd_fp32_relaxed/planned    -> c:ssa/simd_fp32_relaxed
    c:ssa                              -> c:ssa
    c:ssa/structured/planned           -> c:ssa
    c:loop                             -> c
    c:loop/raw                         -> c:loop/raw
    asm:x86_64                         -> asm
    asm:riscv32                        -> asm:riscv32
    loop:raw                           -> loop:raw
    loop:fold,cse                      -> loop:fold,cse
    loop:cse,fold,cse                  -> loop:fold,cse
    loop:all                           -> loop
    machine_ir:riscv32/selected        -> machine_ir:riscv32/selected
    machine_ir:generic/aarch64         -> machine_ir:aarch64/generic
    ssa:direct/exact                   -> ssa:direct/exact
    coral:zvt_kernel                   -> coral:zvt_kernel
    nope                               !! unknown family "nope"
    c:                                 !! c: empty segment
    c:ssa//exact                       !! c: empty segment
    c:ssa/exact/representation         !! c: segment "representation" repeats an axis already given
    c:vector                           !! c: unknown segment "vector"
    loop:fold,nope                     !! loop: unknown segment "fold,nope"
    loop:raw/all                       !! loop: segment "all" repeats an axis already given
    asm:x86_64/x86_64                  !! asm: unknown segment "x86_64"
    coral:estimate/estimate            !! coral: unknown segment "estimate"
    |}]

let%expect_test "key printing" =
  let k =
    make (Node_id.of_int 3)
      (C (Route.Ssa { Ssa_spec.default with pipeline = Exact }) : variant)
  in
  Fmt.pr "%a %b@." pp_key k (equal k k);
  [%expect {| native/n3/c:ssa/exact true |}]
