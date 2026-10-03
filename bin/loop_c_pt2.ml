(* The whole-model C backend on a real downloaded model: lower the archive,
   build the bundle, generate the two C files, pack the weights, compile with the
   host compiler and run every sample through the standalone binary.

   argv: <model.pt2> <inputs.pt> <expected.json> <outputs.pt> [--strict]
         [--shadow] [--poison] [--samples=N] [--keep=DIR] [--cc=CMD]
         [--bench=N] [--reference] [--vector]
         [--numerics=NAME] [--shadow-numeric] [--atol=X] [--rtol=X]

   The default is the performance path (numerics simd_fp32_relaxed, vectorized
   for the host). [--reference] selects the binary64 scalar reference path, which
   every strict gate (--shadow) runs; [--vector] alone vectorizes it strictly.

   [--shadow] also runs [Eval_direct.run] (the per-node reference) on the same
   graph, constants and input and requires every graph output to be bitwise
   equal; without it only the top-5 ranking is checked, which is never a
   correctness gate on its own. [--numerics=NAME] picks the numerical policy
   (Loop_numerics.name; default simd_fp32_relaxed, or reference_f64 under
   --reference): binary32 kernels are not bitwise equal to the reference, so a
   simd_fp32_* policy refuses --shadow and is checked by --shadow-numeric instead,
   which runs
   the same reference and reports absolute, relative, normalized and ULP error
   and every nonfinite cell (Loop_numeric_diff), failing when a cell leaves
   |actual - reference| <= atol + rtol * |reference| (--atol, --rtol; 1e-4
   each by default); the coverage line says how many kernels ran in each
   precision. [--keep=DIR] keeps the artifact there. Timings
   go to stderr. *)

open Loop_ir

type eval =
  [ Native_interp.error
  | Native_predict.error
  | Loop_bundle.error
  | Loop_c_exec.Host.error
  | `Bundle_mismatch of int
  | `Numeric_mismatch of int ]

let pp_eval ppf : eval -> unit = function
  | `Bundle_mismatch n ->
      Format.fprintf ppf
        "C model: %d output(s) differ bitwise from the reference" n
  | `Numeric_mismatch n ->
      Format.fprintf ppf "C model: %d cell(s) outside the numerical tolerance" n
  | #Loop_c_exec.Host.error as e -> Loop_c_exec.Host.pp_error ppf e
  | #Loop_bundle.error as e -> Loop_bundle.pp_error ppf e
  | #Native_predict.error as e -> Native_predict.pp_error ppf e
  | #Native_interp.error as e -> Native_interp.pp_error ppf e

let bits t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  !acc

let flag name argv =
  ( Array.exists (String.equal name) argv,
    Array.of_list (List.filter (fun a -> a <> name) (Array.to_list argv)) )

let valued prefix argv =
  let n = String.length prefix in
  let v = ref None in
  let rest =
    List.filter
      (fun a ->
        if String.length a > n && String.sub a 0 n = prefix then (
          v := Some (String.sub a n (String.length a - n));
          false)
        else true)
      (Array.to_list argv)
  in
  (!v, Array.of_list rest)

let now = Unix.gettimeofday
let ms t0 = (now () -. t0) *. 1000.
let cache = ref None

let prepared ~keep ~compiler ~vector ~numerics archive =
  let open Err.Syntax in
  let map e = (e :> eval) in
  match !cache with
  | Some c -> Err.return c
  | None ->
      let t0 = now () in
      let* lowered = Native_interp.lower_archive archive |> Err.map_error map in
      let g = lowered.Pt2_native_graph.graph in
      let* constants =
        Native_interp.preload archive lowered |> Err.map_error map
      in
      let* b =
        Loop_bundle.build ~config:Loop_bundle_c.default_config g
        |> Err.map_error map
      in
      let t1 = now () in
      let dir =
        match keep with
        | Some d -> d
        | None -> Loop_c_exec.Proc.temp_dir "loop_c_pt2"
      in
      let* p =
        Loop_c_exec.Host.prepare ?vector ~numerics ?compiler ~dir b
          ~constants:(fun id -> Graph_ir.Tensor_id.Map.find_opt id constants)
        |> Err.map_error map
      in
      let c = Loop_c_exec.Host.bundle_c p in
      let st = c.Loop_bundle_c.stats in
      let ws = c.Loop_bundle_c.workspace in
      Printf.eprintf
        "loop_c_pt2: %d invocations, %d distinct kernels, source %d bytes; \
         weights %Ld bytes, workspace %Ld bytes (arena %Ld); lower+bundle %.0f \
         ms, generate+pack+compile %.0f ms; %s; dir %s\n\
         %!"
        st.Loop_bundle_c.invocations st.Loop_bundle_c.distinct_kernels
        st.Loop_bundle_c.source_bytes
        c.Loop_bundle_c.weights.C_payload_layout.length
        (C_workspace_plan.bytes ws)
        (C_workspace_plan.graph_arena_bytes ws)
        (ms t0 -. ms t1)
        (ms t1)
        (Loop_c_exec.Host.compiler_identity p)
        dir;
      Printf.eprintf "loop_c_pt2: %s\n%!"
        (Loop_numerics.coverage ~numerics:st.Loop_bundle_c.numerics
           ~f32_kernels:st.Loop_bundle_c.f32_kernels
           ~kernels:st.Loop_bundle_c.distinct_kernels
           ~f32_invocations:st.Loop_bundle_c.f32_invocations
           ~invocations:st.Loop_bundle_c.invocations
           ~refusals:st.Loop_bundle_c.fp32_refusals);
      let cached = (g, constants, b, p) in
      cache := Some cached;
      Err.return cached

(* [--bench=N]: the phases of one run apart, then N warm repeats inside the
   binary. Fresh-process figures include start-up, mapping and publication;
   the warm ones are the inference unit alone on a dirty workspace. *)
let bench p ~bind n =
  let module H = Loop_c_exec.Host in
  let dir = H.directory p in
  let inputs = Filename.concat dir "bench_in.bin"
  and outputs = Filename.concat dir "bench_out.bin" in
  let t0 = now () in
  (match Err.payload (H.write_inputs p ~bind ~path:inputs) with
  | Ok () -> ()
  | Error e -> Format.eprintf "bench: %a@." H.pp_error e);
  let t_pack = ms t0 in
  let t1 = now () in
  let status =
    Loop_c_exec.Proc.run
      (H.command p ~inputs ~outputs @ [ "--repeat"; string_of_int n ])
  in
  let t_proc = ms t1 in
  let t2 = now () in
  ignore (H.read_outputs p ~path:outputs);
  let t_read = ms t2 in
  (match status with
  | Ok (_, log) ->
      let line =
        List.find_opt
          (fun l -> String.starts_with ~prefix:"model_time_ms:" l)
          (String.split_on_char '\n' log)
      in
      Printf.eprintf
        "loop_c_pt2 bench: pack inputs %.1f ms, fresh process %.1f ms, decode \
         outputs %.1f ms; %s\n\
         %!"
        t_pack t_proc t_read
        (Option.value ~default:"(no timing line)" line)
  | Error m -> Printf.eprintf "bench: %s\n%!" m);
  List.iter
    (fun f -> try Sys.remove f with Sys_error _ -> ())
    [ inputs; outputs ]

let infer ~keep ~compiler ~vector ~numerics ~shadow ~numeric ~poison
    ~bench:bench_n archive image =
  let open Err.Syntax in
  let map e = (e :> eval) in
  let* g, constants, b, p =
    prepared ~keep ~compiler ~vector ~numerics archive
  in
  let* input = Native_interp.tensor_of_pt2 image |> Err.map_error map in
  let input_id = List.hd b.Loop_bundle.inputs in
  let t0 = now () in
  let* outs =
    Loop_c_exec.Host.run ~poison p ~bind:(fun id ->
        if Graph_ir.Tensor_id.equal id input_id then Some input else None)
    |> Err.map_error map
  in
  Printf.eprintf "loop_c_pt2: run (pack, process, decode) %.1f ms\n%!" (ms t0);
  Option.iter
    (bench p ~bind:(fun id ->
         if Graph_ir.Tensor_id.equal id input_id then Some input else None))
    bench_n;
  let* () =
    if not shadow then Err.return ()
    else
      let t1 = now () in
      let* reference =
        Eval_direct.run g
          ~constants:(Graph_ir.Tensor_id.Map.bindings constants)
          ~inputs:[ (input_id, input) ]
        |> Err.map_error map
      in
      let bad =
        List.length
          (List.filter
             (fun (id, t) ->
               bits t <> bits (Graph_ir.Tensor_id.Map.find id reference))
             (List.combine g.Graph_ir.Graph.outputs outs))
      in
      Printf.eprintf
        "loop_c_pt2: shadow %d/%d outputs differ (reference %.0f ms)\n%!" bad
        (List.length outs) (ms t1);
      if bad = 0 then Err.return () else Err.fail (`Bundle_mismatch bad)
  in
  let* () =
    match numeric with
    | None -> Err.return ()
    | Some (atol, rtol) -> (
        let t1 = now () in
        let* reference =
          Eval_direct.run g
            ~constants:(Graph_ir.Tensor_id.Map.bindings constants)
            ~inputs:[ (input_id, input) ]
          |> Err.map_error map
        in
        let per_output =
          Loop_numeric_diff.outputs ~atol ~rtol
            ~reference:(fun id -> Graph_ir.Tensor_id.Map.find id reference)
            (List.combine g.Graph_ir.Graph.outputs outs)
        in
        List.iter
          (fun (id, d) ->
            Format.eprintf "loop_c_pt2: numeric shadow t%d: %a@."
              (Graph_ir.Tensor_id.to_int id)
              Loop_numeric_diff.pp d)
          per_output;
        let total = Loop_numeric_diff.total per_output in
        Printf.eprintf "loop_c_pt2: numeric shadow done (reference %.0f ms)\n%!"
          (ms t1);
        match total with
        | Some d when d.Loop_numeric_diff.failing > 0 ->
            Err.fail (`Numeric_mismatch d.Loop_numeric_diff.failing)
        | _ -> Err.return ())
  in
  let* top = Native_predict.top_predictions outs 5 |> Err.map_error map in
  Err.return (List.map (fun ((c : Dim.index Dim.t), p) -> ((c :> int), p)) top)

let () =
  let shadow, argv = flag "--shadow" Sys.argv in
  let poison, argv = flag "--poison" argv in
  let samples, argv = valued "--samples=" argv in
  let keep, argv = valued "--keep=" argv in
  let bench_n, argv = valued "--bench=" argv in
  let bench_n = Option.map int_of_string bench_n in
  let numeric, argv = flag "--shadow-numeric" argv in
  let atol, argv = valued "--atol=" argv in
  let rtol, argv = valued "--rtol=" argv in
  let numeric =
    if numeric then
      Some
        ( Option.fold ~none:1e-4 ~some:float_of_string atol,
          Option.fold ~none:1e-4 ~some:float_of_string rtol )
    else None
  in
  let vector_flag, argv = flag "--vector" argv in
  let reference, argv = flag "--reference" argv in
  let numerics, argv = valued "--numerics=" argv in
  let row_block, argv = valued "--row-block=" argv in
  (* The default is the performance path: binary32 kernels where the planner
     vectorizes, scheduled sums and contraction ([simd_fp32_relaxed]).
     [--reference] is the binary64 scalar reference path every strict gate runs;
     [--numerics=NAME] picks any policy, and a binary32 one always vectorizes. *)
  let numerics =
    match numerics with
    | None ->
        if reference then Loop_numerics.Reference_f64
        else Loop_numerics.Simd_fp32_relaxed
    | Some n -> (
        match Loop_numerics.of_name n with
        | Some p -> p
        | None ->
            Printf.eprintf "loop_c_pt2: unknown numerics %S (one of %s)\n" n
              (String.concat ", "
                 (List.map Loop_numerics.name Loop_numerics.all));
            exit 2)
  in
  let vector =
    if vector_flag || numerics <> Loop_numerics.Reference_f64 then
      Some
        (match row_block with
        | Some n ->
            Loop_target.with_row_block (int_of_string n) Loop_target.neon128
        | None -> Loop_target.neon128)
    else None
  in
  (match numerics with
  | Loop_numerics.Reference_f64 -> ()
  | _ ->
      if shadow then (
        prerr_endline
          "loop_c_pt2: --shadow is bitwise against the binary64 reference, \
           which binary32 kernels do not match; use --reference, or \
           --shadow-numeric for the default policy";
        exit 2));
  let cc, argv = valued "--cc=" argv in
  let compiler =
    Option.map
      (fun s -> List.filter (fun x -> x <> "") (String.split_on_char ' ' s))
      cc
  in
  match Infer_report.parse_argv argv with
  | Error usage ->
      prerr_endline usage;
      exit 2
  | Ok (paths, options) -> (
      match
        Err.payload
          (Infer_report.run
             ?max_samples:(Option.map int_of_string samples)
             ~now
             ~infer:
               (infer ~keep ~compiler ~vector ~numerics ~shadow ~numeric ~poison
                  ~bench:bench_n)
             paths options)
      with
      | Ok () -> ()
      | Error e ->
          Format.eprintf "loop_c_pt2: %a@." (Infer_report.pp_error pp_eval) e;
          exit 1)
