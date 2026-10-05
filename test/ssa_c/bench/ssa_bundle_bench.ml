open Graph_ir
open Loop_ir

(* [ssa_bundle_bench [REPEATS]]: for each graph and each way of making its
   kernels (the Loop path, then the SSA path through each pipeline), the time to
   generate the bundle, to compile it, the first run and the median warm run,
   the C source size and the kernel count. The samples are raw seconds, one per
   repetition; nothing is smoothed. *)

let tensor_of ~salt (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      0.25
      +. 0.05
         *. float_of_int
              (((Vec6.offset sg.Tensor_sig.shape c :> int) + salt) mod 7))

let sig_of_edge (b : Loop_bundle.t) id =
  List.find_map
    (fun (inv : Loop_bundle.invocation) ->
      List.find_map
        (fun ((buf : Loop_buffer.t), edge) ->
          if Tensor_id.equal edge id then Some buf.Loop_buffer.sg else None)
        (List.combine inv.Loop_bundle.program.Loop_program.buffers
           inv.Loop_bundle.edges))
    b.Loop_bundle.invocations
  |> Option.get

let map_of l id = List.assoc_opt id l

let time f =
  let t0 = Unix.gettimeofday () in
  let x = f () in
  (x, Unix.gettimeofday () -. t0)

let median l =
  let a = Array.of_list (List.sort compare l) in
  a.(Array.length a / 2)

let wide_chain () =
  let module F = Native_test.Graph_fixtures in
  F.build "wide_chain"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:56 ~w:56 ~c:32) () in
      let* w =
        constant ~shape:(F.weight_shape ~out_channels:64 ~in_channels:32) ()
      in
      let* bias = constant ~shape:(F.s1c 64) () in
      let* gamma = constant ~shape:(F.s1c 64) () in
      let* beta = constant ~shape:(F.s1c 64) () in
      let* mean = constant ~shape:(F.s1c 64) () in
      let* var = constant ~shape:(F.s1c 64) () in
      let* y = conv2d (F.conv_params ~in_channels:32) ~x ~weight:w ~bias () in
      let* n =
        batch_norm F.bn_params ~x:y ~weight:gamma ~bias:beta ~running_mean:mean
          ~running_var:var ()
      in
      relu n)

let variants =
  let ssa pipeline =
    ( "ssa:" ^ Ssa_backends.Pipeline.name pipeline,
      Some (Ssa_backends.c ~pipeline) )
  in
  ("loop", None)
  :: List.map ssa
       [
         Ssa_backends.Pipeline.Representation;
         Ssa_backends.Pipeline.Exact;
         Ssa_backends.Pipeline.Planned
           {
             numerics = Ssa_ir.Ssa_numerics.Reference_f64;
             target = Ssa_ir.Ssa_target.neon128;
           };
       ]

let measure ~repeats name g (variant, kernel) =
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error
      (Loop_bundle.build ~config:Loop_bundle_c.default_config g)
  in
  let constants =
    List.map
      (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b id)))
      b.Loop_bundle.constants
  in
  let inputs =
    List.map
      (fun id -> (id, tensor_of ~salt:1 (sig_of_edge b id)))
      b.Loop_bundle.inputs
  in
  let generate, compile, first, warm, size, kernels =
    let dir = Loop_c_exec.Proc.temp_dir "ssa_bundle_bench" in
    Fun.protect
      ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
      (fun () ->
        let gen, g_s =
          time (fun () ->
              Err.or_raise ~pp_error:Loop_bundle_c.pp_error
                (Loop_bundle_c.build ?kernel b))
        in
        let p, c_s =
          time (fun () ->
              Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
                (Loop_c_exec.Host.prepare ?kernel ~dir b
                   ~constants:(map_of constants)))
        in
        let run () =
          snd
            (time (fun () ->
                 Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
                   (Loop_c_exec.Host.run p ~bind:(map_of inputs))))
        in
        let first = run () in
        let warm = List.init repeats (fun _ -> run ()) in
        ( g_s,
          c_s,
          first,
          warm,
          String.length gen.Loop_bundle_c.source,
          gen.Loop_bundle_c.stats.Loop_bundle_c.distinct_kernels ))
  in
  Fmt.pr
    "%-10s %-34s generate %.4fs  generate+compile %.3fs  first %.4fs  warm \
     median %.4fs  source %d bytes  %d kernels@."
    name variant generate compile first (median warm) size kernels;
  Fmt.pr "  warm samples: %a@." Fmt.(list ~sep:(any " ") (fmt "%.4f")) warm

let () =
  let repeats =
    if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 5
  in
  Fmt.pr "compiler: %s@." (String.concat " " Loop_c_exec.Host.default_compiler);
  List.iter
    (fun (name, g) ->
      let g = g () in
      List.iter (measure ~repeats name g) variants)
    [ ("chain", Native_test.Graph_fixtures.chain); ("wide_chain", wide_chain) ]
