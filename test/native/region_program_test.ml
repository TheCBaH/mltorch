open Expr

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3

let pp =
  Core.Pretty.err_result ~ok:Region_program.pp ~error:Region_program.pp_error

let%expect_test
    "Region_partition enumerates keys and owned outputs in canonical order" =
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Axis.C ])
  in
  let keys =
    Region_partition.fold_keys ~output_shape:shape ~init:[]
      ~f:(fun acc key -> Format.asprintf "%a" Vec6.pp_coord key :: acc)
      partition
    |> List.rev
  in
  let outputs =
    Region_partition.fold_outputs ~output_shape:shape ~key:Vec6.origin ~init:[]
      ~f:(fun acc output -> Format.asprintf "%a" Vec6.pp_coord output :: acc)
      partition
    |> List.rev
  in
  Fmt.pr "keys: %s@.outputs: %s@." (String.concat "," keys)
    (String.concat "," outputs);
  [%expect {|
    keys: (0),(1,0)
    outputs: (0),(1),(2)
    |}]

let%expect_test
    "Region_partition owned enumeration matches the former filtered domain scan"
    =
  let output_shape = Vec6.shape ~n:2 ~t:2 ~d:1 ~h:2 ~w:3 ~c:2 in
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Axis.T; Axis.W; Axis.C ])
  in
  let keys =
    Region_partition.fold_keys ~output_shape ~init:[]
      ~f:(fun keys key -> key :: keys)
      partition
    |> List.rev
  in
  let owned key =
    Region_partition.fold_outputs ~output_shape ~key ~init:[]
      ~f:(fun outputs output -> output :: outputs)
      partition
    |> List.rev
  in
  let filtered key =
    Vec6.fold_coords output_shape ~init:[] ~f:(fun outputs output ->
        let belongs axis =
          match Region_partition.mode partition axis with
          | Region_partition.Axis_mode.Whole -> true
          | Singleton -> Dim.equal (Vec6.get key axis) (Vec6.get output axis)
        in
        if List.for_all belongs Axis.all then output :: outputs else outputs)
    |> List.rev
  in
  let matches = List.for_all (fun key -> owned key = filtered key) keys in
  let first =
    List.hd keys |> owned |> List.map (Format.asprintf "%a" Vec6.pp_coord)
  in
  Fmt.pr "keys=%d outputs-per-key=%d matches=%b first=%s@." (List.length keys)
    (List.length (owned (List.hd keys)))
    matches (String.concat "," first);
  [%expect
    {|
    keys=4 outputs-per-key=12 matches=true first=(0),(1),(1,0),(1,1),(2,0),(2,1),(1,0,0,0,0),(1,0,0,0,1),(1,0,0,1,0),(1,0,0,1,1),(1,0,0,2,0),(1,0,0,2,1)
    |}]

let%expect_test
    "Region_program validates dependent scalar locals and whole-axis invariance"
    =
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Axis.C ])
  in
  let good =
    Region_program.Builder.run
      (Region_program.Builder.scalar (Value.const 2.) (fun first ->
           Region_program.Builder.scalar
             (Value.add first (Value.const 3.))
             (fun second ->
               Region_program.Builder.finish ~max_size:32 ~max_depth:16
                 ~partition
                 ~output:(Value.add second (Value.const 1.)))))
  in
  Fmt.pr "good:@.%a@." pp good;
  [%expect
    {|
    good:
    region [N=singleton T=singleton D=singleton H=singleton W=singleton C=whole]
      let l0 : scalar = 2
      let l1 : scalar = (l0 + 3)
      emit (l1 + 1)
    |}];
  let id = Builder.run Builder.fresh_local in
  let varying =
    Region_program.create ~max_size:32 ~max_depth:16 ~partition
      ~locals:
        [
          Region_local.scalar ~id
            ~value:
              (Value.value_of_index (Index.of_position (Index.output Axis.C)));
        ]
      ~output:(Value.local id)
  in
  Fmt.pr "varying: %a@." pp varying;
  [%expect {| varying: local #0 varies over whole axis C |}]

let%expect_test
    "Region_program distinguishes duplicate, forward, and unknown locals" =
  let partition = Region_partition.singleton in
  let (first, second), state =
    Builder.run_from Builder.initial
      (let open Builder.Syntax in
       let* first = Builder.fresh_local in
       let* second = Builder.fresh_local in
       Builder.return (first, second))
  in
  let local id value = Region_local.scalar ~id ~value in
  let render locals output =
    Region_program.create ~max_size:32 ~max_depth:16 ~partition ~locals ~output
  in
  Fmt.pr "duplicate: %a@." pp
    (render
       [ local first (Value.const 1.); local first (Value.const 2.) ]
       (Value.const 0.));
  Fmt.pr "forward: %a@." pp
    (render
       [ local first (Value.local second); local second (Value.const 2.) ]
       (Value.local second));
  let unknown, _ = Builder.run_from state Builder.fresh_local in
  Fmt.pr "unknown: %a@." pp
    (render [ local first (Value.local unknown) ] (Value.local first));
  [%expect
    {|
    duplicate: duplicate local #0
    forward: local #0 refers forward to #1
    unknown: local #0 refers to unknown local #2
    |}]

let%expect_test "Region_eval shares scalar locals across a whole axis" =
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Axis.C ])
  in
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.Builder.run
         (Region_program.Builder.scalar (Value.const 2.) (fun local ->
              Region_program.Builder.finish ~max_size:32 ~max_depth:16
                ~partition
                ~output:(Value.add local (Value.const 1.)))))
  in
  let env =
    {
      Eval.Env.load = (fun _ _ -> assert false);
      load_index = (fun _ _ -> assert false);
    }
  in
  let tensor =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_eval.materialize program
         ~output_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3)
         ~env)
  in
  Fmt.pr "%g,%g@."
    (Tensor.read tensor Vec6.origin)
    (Tensor.read tensor (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:2));
  [%expect {| 3,3 |}]

let%expect_test
    "Region trace proves whole-domain ownership and projection observes \
     materialization" =
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Axis.C ])
  in
  let program =
    let source = Expr_bridge.source_of_id (Tensor_id.of_int 41) in
    let output =
      Expr.Coord.make ~n:(Expr.Index.output Axis.N)
        ~t:(Expr.Index.output Axis.T) ~d:(Expr.Index.output Axis.D)
        ~h:(Expr.Index.output Axis.H) ~w:(Expr.Index.output Axis.W)
        ~c:(Expr.Index.output Axis.C)
    in
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.Builder.run
         (Region_program.Builder.scalar (Value.const 0.) (fun local ->
              Region_program.Builder.finish ~max_size:32 ~max_depth:16
                ~partition
                ~output:(Value.add local (Value.load source output)))))
  in
  let output_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3 in
  let input =
    Tensor.materialize output_shape (fun coord ->
        match (Dim.to_int coord.w, Dim.to_int coord.c) with
        | 0, 0 -> nan
        | 0, 1 -> infinity
        | 0, _ -> neg_infinity
        | 1, 0 -> -0.
        | 1, 1 -> 0.
        | _ -> 1.)
  in
  let env =
    Expr_bridge.env ~binding:(fun id ->
        if Tensor_id.equal id (Tensor_id.of_int 41) then Some input else None)
  in
  let trace =
    Err.or_raise ~pp_error:Region_trace.pp_error
      (Region_trace.collect program ~output_shape)
  in
  let lowered =
    match
      Err.or_raise ~pp_error:Region_program.pp_error
        (Region_execution.lower ~max_size:32 ~max_depth:16 program)
    with
    | Region_execution.Region_loop lowered -> lowered
    | Region_execution.Pixel_loop _ -> assert false
  in
  let tensor =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_execution.materialize lowered ~output_shape ~env)
  in
  let agrees =
    Vec6.fold_coords output_shape ~init:true ~f:(fun agrees output ->
        let projected =
          Err.or_raise ~pp_error:Region_eval.pp_error
            (Region_execution.value_at lowered ~output_shape ~env ~output)
        in
        agrees
        && Core.Float_bits.equal_exact (Tensor.read tensor output) projected)
  in
  let mutated =
    match trace.entries with
    | { key; outputs = first :: second :: _ } :: rest ->
        Region_trace.summarize ~output_shape
          ({ key; outputs = [ first; first; second ] } :: rest)
    | _ -> assert false
  in
  Fmt.pr
    "keys=%d total=%d visited=%d duplicates=%d missing=%d agrees=%b \
     mutation=(duplicates=%d missing=%d)@."
    trace.coverage.keys trace.coverage.total trace.coverage.visited
    trace.coverage.duplicates trace.coverage.missing agrees mutated.duplicates
    mutated.missing;
  [%expect
    {|
    keys=2 total=6 visited=6 duplicates=0 missing=0 agrees=true mutation=(duplicates=1 missing=1)
    |}]

let%expect_test
    "Region_program specializes dependent locals with a saturating preflight" =
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Axis.C ])
  in
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.Builder.run
         (Region_program.Builder.scalar (Value.const 2.) (fun first ->
              Region_program.Builder.scalar (Value.add first first)
                (fun second ->
                  Region_program.Builder.finish ~max_size:32 ~max_depth:16
                    ~partition ~output:(Value.add second second)))))
  in
  let specialized =
    Region_program.specialize_pixel ~max_size:32 ~max_depth:16 program
  in
  Fmt.pr "specialized: %a@.reconstructs: %a@.too-small: %a@." Pp.value
    (Err.or_raise ~pp_error:Region_program.pp_error specialized)
    Fmt.bool
    (Err.or_raise ~pp_error:Region_program.pp_error
       (Region_program.reconstructs ~max_size:32 ~max_depth:16
          ~pixel:
            (Value.add
               (Value.add (Value.const 2.) (Value.const 2.))
               (Value.add (Value.const 2.) (Value.const 2.)))
          program))
    (Core.Pretty.err_result ~ok:Pp.value ~error:Region_program.pp_error)
    (Region_program.specialize_pixel ~max_size:6 ~max_depth:16 program);
  [%expect
    {|
    specialized: ((2 + 2) + (2 + 2))
    reconstructs: true
    too-small: size exceeds limit 6 |}]

let%expect_test "Region_program leaves Pixel programs untouched" =
  let pixel = Value.add (Value.const 2.) (Value.const 3.) in
  let program = Region_program.pixel pixel in
  let specialized =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.specialize_pixel ~max_size:8 ~max_depth:8 program)
  in
  let mismatch =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.reconstructs ~max_size:8 ~max_depth:8
         ~pixel:(Value.add (Value.const 3.) (Value.const 2.))
         program)
  in
  Fmt.pr "physical: %b@.mismatch: %b@." (pixel == specialized) mismatch;
  [%expect {|
    physical: true
    mismatch: false |}]
