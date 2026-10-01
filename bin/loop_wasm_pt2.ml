(* The whole-model Wasm backend on a real downloaded model: lower the archive,
   build the bundle, generate the module and pack the weights, then run every
   sample through [model_run] under node.

   argv: <model.pt2> <inputs.pt> <expected.json> <outputs.pt> [--strict]
         [--shadow] [--poison] [--samples=N] [--keep=DIR] [--bench=N]
         [--export=DIR] [--via-c] [--cflags=FLAGS] [--wat=FILE] [--simd] [--simd-forced]

   [--shadow] also runs [Eval_direct.run] (the per-node reference) on the same
   graph, constants and input and requires every graph output to be bitwise
   equal; without it only the top-5 ranking is checked, which is never a
   correctness gate on its own. [--keep=DIR] keeps the artifact there.
   [--bench=N] runs the schedule N more times on the same instance, dirty
   workspace included, requiring identical outputs, and reports the phases.
   [--via-c] runs the C backend's unit compiled to Wasm instead (needs
   MLTORCH_WASI_SYSROOT), [--cflags] replacing its scalar flags. Timings go to
   stderr. *)

open Loop_ir

type eval =
  [ Native_interp.error
  | Native_predict.error
  | Loop_bundle.error
  | Loop_wasm_exec.Host.error
  | Loop_wasm_exec.Via_c.error
  | `Bundle_mismatch of int ]

let pp_eval ppf : eval -> unit = function
  | `Bundle_mismatch n ->
      Format.fprintf ppf
        "Wasm model: %d output(s) differ bitwise from the reference" n
  | #Loop_wasm_exec.Host.error as e -> Loop_wasm_exec.Host.pp_error ppf e
  | #Loop_wasm_exec.Via_c.error as e -> Loop_wasm_exec.Via_c.pp_error ppf e
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

(* One route to a module: the direct emitter, or the C backend's unit compiled
   to Wasm ([--via-c]). Both run the same schedule over the same payload files
   under node. *)
type route = {
  run :
    poison:bool ->
    repeat:int option ->
    bind:(Graph_ir.Tensor_id.t -> Tensor.packed option) ->
    (Tensor.packed list, eval) Err.t;
  timings : unit -> (string * float) list;
  export : string -> input:Tensor.packed -> outs:Tensor.packed list -> unit;
}

let cache = ref None

let write_artifacts dir ~module_file ~weights_file ~template ~identity ~inputs
    ~outputs ~placement ~input ~outs =
  let module Io = Loop_c_exec.Payload_io in
  let copy src dst =
    Loop_c_exec.Proc.write_file (Filename.concat dir dst)
      (Loop_c_exec.Proc.read_file src)
  in
  Io.mkdir_p dir;
  copy module_file "model.wasm";
  copy weights_file "weights.bin";
  copy template "outputs.template";
  let check = function Ok () -> () | Error (`Io m) -> failwith m in
  check
    (Io.write_payload inputs ~identity
       ~path:(Filename.concat dir "inputs.bin")
       [ input ]);
  check
    (Io.write_payload outputs ~identity
       ~path:(Filename.concat dir "outputs.bin")
       outs);
  Loop_c_exec.Proc.write_file (Filename.concat dir "placement.json") placement;
  Printf.eprintf "loop_wasm_pt2: exported to %s\n%!" dir

let placement_json (pl : Loop_wasm_exec.Node.placement) ~total ~identity =
  Printf.sprintf
    "{\"weights\":%d,\"inputs\":%d,\"workspace\":%d,\"outputs\":%d,\"total\":%d,\"workspaceBytes\":%Ld,\"outputsBytes\":%Ld,\"identity\":\"%s\"}\n"
    pl.Loop_wasm_exec.Node.weights pl.Loop_wasm_exec.Node.inputs
    pl.Loop_wasm_exec.Node.workspace pl.Loop_wasm_exec.Node.outputs total
    pl.Loop_wasm_exec.Node.workspace_bytes pl.Loop_wasm_exec.Node.outputs_bytes
    (Digest.to_hex identity)

let direct_route ~dir ~wat ~vector b ~constants =
  let open Err.Syntax in
  let map e = (e :> eval) in
  let module H = Loop_wasm_exec.Host in
  let t1 = now () in
  let* p = H.prepare ?vector ~dir b ~constants |> Err.map_error map in
  let w = H.bundle_wasm p in
  let st = w.Loop_bundle_wasm.stats in
  let ws = w.Loop_bundle_wasm.workspace in
  Printf.eprintf
    "loop_wasm_pt2: direct: %d invocations, %d distinct kernels, module %d \
     bytes; weights %Ld bytes, workspace %Ld bytes (arena %Ld), memory %d \
     bytes; generate+pack %.0f ms; dir %s\n\
     %!"
    st.Loop_bundle_wasm.invocations st.Loop_bundle_wasm.distinct_kernels
    st.Loop_bundle_wasm.module_bytes
    w.Loop_bundle_wasm.weights.C_payload_layout.length
    (C_workspace_plan.bytes ws)
    (C_workspace_plan.graph_arena_bytes ws)
    w.Loop_bundle_wasm.placement.Loop_bundle_wasm.Placement.total (ms t1) dir;
  Printf.printf "module identity %s\n%!"
    (Digest.to_hex w.Loop_bundle_wasm.identity);
  Option.iter
    (fun f ->
      Loop_c_exec.Proc.write_file f
        (Wasm_wat.to_string w.Loop_bundle_wasm.module_))
    wat;
  Err.return
    {
      run =
        (fun ~poison ~repeat ~bind ->
          H.run ~poison ?repeat p ~bind |> Err.map_error map);
      timings = (fun () -> H.timings p);
      export =
        (fun d ~input ~outs ->
          let pl = w.Loop_bundle_wasm.placement in
          write_artifacts d ~module_file:(H.module_path p)
            ~weights_file:(H.weights_path p)
            ~template:(Filename.concat (H.directory p) "outputs.template")
            ~identity:w.Loop_bundle_wasm.identity
            ~inputs:w.Loop_bundle_wasm.inputs
            ~outputs:w.Loop_bundle_wasm.outputs
            ~placement:
              (placement_json
                 {
                   Loop_wasm_exec.Node.weights =
                     pl.Loop_bundle_wasm.Placement.weights;
                   inputs = pl.Loop_bundle_wasm.Placement.inputs;
                   workspace = pl.Loop_bundle_wasm.Placement.workspace;
                   outputs = pl.Loop_bundle_wasm.Placement.outputs;
                   workspace_bytes = C_workspace_plan.bytes ws;
                   outputs_bytes =
                     w.Loop_bundle_wasm.outputs.C_payload_layout.length;
                 }
                 ~total:pl.Loop_bundle_wasm.Placement.total
                 ~identity:w.Loop_bundle_wasm.identity)
            ~input ~outs);
    }

let c_route ~dir ~flags b ~constants =
  let open Err.Syntax in
  let map e = (e :> eval) in
  let module V = Loop_wasm_exec.Via_c in
  let* toolchain =
    match V.toolchain_from_env () with
    | Ok t -> Err.return t
    | Error e -> Err.fail (e :> eval)
  in
  let t1 = now () in
  let* p = V.prepare ?flags ~toolchain ~dir b ~constants |> Err.map_error map in
  let c = V.bundle_c p in
  let st = c.Loop_bundle_c.stats in
  let z = V.sizes p in
  Printf.eprintf
    "loop_wasm_pt2: via C: %d invocations, %d distinct kernels, source %d \
     bytes, module %d bytes, memory %d bytes, vector instructions in the \
     generated code %d; generate+compile+link %.0f ms; dir %s\n\
     %!"
    st.Loop_bundle_c.invocations st.Loop_bundle_c.distinct_kernels
    z.V.source_bytes z.V.module_bytes z.V.memory_bytes z.V.vector_instructions
    (ms t1) dir;
  Err.return
    {
      run =
        (fun ~poison ~repeat ~bind ->
          V.run ~poison ?repeat p ~bind |> Err.map_error map);
      timings = (fun () -> V.timings p);
      export =
        (fun _ ~input:_ ~outs:_ ->
          prerr_endline "--export applies to the direct route only");
    }

let prepared ~keep ~via_c ~cflags ~wat ~vector archive =
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
        Loop_bundle.build ~config:Loop_bundle_wasm.default_config g
        |> Err.map_error map
      in
      Printf.eprintf "loop_wasm_pt2: lower+bundle %.0f ms\n%!" (ms t0);
      let dir =
        match keep with
        | Some d -> d
        | None -> Loop_c_exec.Proc.temp_dir "loop_wasm_pt2"
      in
      let constants_of id = Graph_ir.Tensor_id.Map.find_opt id constants in
      let* route =
        if via_c then c_route ~dir ~flags:cflags b ~constants:constants_of
        else direct_route ~dir ~wat ~vector b ~constants:constants_of
      in
      let cached = (g, constants, b, route) in
      cache := Some cached;
      Err.return cached

let infer ~keep ~via_c ~cflags ~wat ~vector ~export:export_dir ~shadow ~poison
    ~bench:bench_n archive image =
  let open Err.Syntax in
  let map e = (e :> eval) in
  let* g, constants, b, route =
    prepared ~keep ~via_c ~cflags ~wat ~vector archive
  in
  let* input = Native_interp.tensor_of_pt2 image |> Err.map_error map in
  let input_id = List.hd b.Loop_bundle.inputs in
  let t0 = now () in
  let* outs =
    route.run ~poison ~repeat:bench_n ~bind:(fun id ->
        if Graph_ir.Tensor_id.equal id input_id then Some input else None)
  in
  Printf.eprintf "loop_wasm_pt2: run (pack, process, decode) %.1f ms\n%!"
    (ms t0);
  (match route.timings () with
  | [] -> ()
  | l ->
      Printf.eprintf "loop_wasm_pt2: node phases (ms): %s\n%!"
        (String.concat ", "
           (List.map (fun (k, v) -> Printf.sprintf "%s %.2f" k v) l)));
  Option.iter (fun d -> route.export d ~input ~outs) export_dir;
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
        "loop_wasm_pt2: shadow %d/%d outputs differ (reference %.0f ms)\n%!" bad
        (List.length outs) (ms t1);
      if bad = 0 then Err.return () else Err.fail (`Bundle_mismatch bad)
  in
  let* top = Native_predict.top_predictions outs 5 |> Err.map_error map in
  Err.return (List.map (fun ((c : Dim.index Dim.t), p) -> ((c :> int), p)) top)

let () =
  let shadow, argv = flag "--shadow" Sys.argv in
  let poison, argv = flag "--poison" argv in
  let samples, argv = valued "--samples=" argv in
  let keep, argv = valued "--keep=" argv in
  let export_dir, argv = valued "--export=" argv in
  let via_c, argv = flag "--via-c" argv in
  let simd, argv = flag "--simd" argv in
  let forced, argv = flag "--simd-forced" argv in
  let vector =
    if forced then Some (Loop_target.forced Loop_target.wasm128)
    else if simd then Some Loop_target.wasm128
    else None
  in
  let wat, argv = valued "--wat=" argv in
  let cflags, argv = valued "--cflags=" argv in
  let cflags =
    Option.map
      (fun s -> List.filter (fun x -> x <> "") (String.split_on_char ' ' s))
      cflags
  in
  let bench_n, argv = valued "--bench=" argv in
  let bench_n = Option.map int_of_string bench_n in
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
               (infer ~keep ~via_c ~cflags ~wat ~vector ~export:export_dir
                  ~shadow ~poison ~bench:bench_n)
             paths options)
      with
      | Ok () -> ()
      | Error e ->
          Format.eprintf "loop_wasm_pt2: %a@." (Infer_report.pp_error pp_eval) e;
          exit 1)
