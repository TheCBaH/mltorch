(* The whole-model Wasm backend on a real downloaded model: lower the archive,
   build the bundle, generate the module and pack the weights, then run every
   sample through [model_run] under node.

   argv: <model.pt2> <inputs.pt> <expected.json> <outputs.pt> [--strict]
         [--shadow] [--poison] [--samples=N] [--keep=DIR] [--bench=N]

   [--shadow] also runs [Eval_direct.run] (the per-node reference) on the same
   graph, constants and input and requires every graph output to be bitwise
   equal; without it only the top-5 ranking is checked, which is never a
   correctness gate on its own. [--keep=DIR] keeps the artifact there.
   [--bench=N] runs the schedule N more times on the same instance, dirty
   workspace included, requiring identical outputs, and reports the phases.
   Timings go to stderr. *)

open Loop_ir

type eval =
  [ Native_interp.error
  | Native_predict.error
  | Loop_bundle.error
  | Loop_wasm_exec.Host.error
  | `Bundle_mismatch of int ]

let pp_eval ppf : eval -> unit = function
  | `Bundle_mismatch n ->
      Format.fprintf ppf
        "Wasm model: %d output(s) differ bitwise from the reference" n
  | #Loop_wasm_exec.Host.error as e -> Loop_wasm_exec.Host.pp_error ppf e
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

let prepared ~keep archive =
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
      let t1 = now () in
      let dir =
        match keep with
        | Some d -> d
        | None -> Loop_c_exec.Proc.temp_dir "loop_wasm_pt2"
      in
      let* p =
        Loop_wasm_exec.Host.prepare ~dir b ~constants:(fun id ->
            Graph_ir.Tensor_id.Map.find_opt id constants)
        |> Err.map_error map
      in
      let w = Loop_wasm_exec.Host.bundle_wasm p in
      let st = w.Loop_bundle_wasm.stats in
      let ws = w.Loop_bundle_wasm.workspace in
      Printf.eprintf
        "loop_wasm_pt2: %d invocations, %d distinct kernels, module %d bytes; \
         weights %Ld bytes, workspace %Ld bytes (arena %Ld), memory %d bytes; \
         lower+bundle %.0f ms, generate+pack %.0f ms; dir %s\n\
         %!"
        st.Loop_bundle_wasm.invocations st.Loop_bundle_wasm.distinct_kernels
        st.Loop_bundle_wasm.module_bytes
        w.Loop_bundle_wasm.weights.C_payload_layout.length
        (C_workspace_plan.bytes ws)
        (C_workspace_plan.graph_arena_bytes ws)
        w.Loop_bundle_wasm.placement.Loop_bundle_wasm.Placement.total
        (ms t0 -. ms t1)
        (ms t1) dir;
      let cached = (g, constants, b, p) in
      cache := Some cached;
      Err.return cached

let infer ~keep ~shadow ~poison ~bench:bench_n archive image =
  let open Err.Syntax in
  let map e = (e :> eval) in
  let* g, constants, b, p = prepared ~keep archive in
  let* input = Native_interp.tensor_of_pt2 image |> Err.map_error map in
  let input_id = List.hd b.Loop_bundle.inputs in
  let t0 = now () in
  let* outs =
    Loop_wasm_exec.Host.run ~poison ?repeat:bench_n p ~bind:(fun id ->
        if Graph_ir.Tensor_id.equal id input_id then Some input else None)
    |> Err.map_error map
  in
  Printf.eprintf "loop_wasm_pt2: run (pack, process, decode) %.1f ms\n%!"
    (ms t0);
  (match Loop_wasm_exec.Host.timings p with
  | [] -> ()
  | l ->
      Printf.eprintf "loop_wasm_pt2: node phases (ms): %s\n%!"
        (String.concat ", "
           (List.map (fun (k, v) -> Printf.sprintf "%s %.2f" k v) l)));
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
             ~infer:(infer ~keep ~shadow ~poison ~bench:bench_n)
             paths options)
      with
      | Ok () -> ()
      | Error e ->
          Format.eprintf "loop_wasm_pt2: %a@." (Infer_report.pp_error pp_eval) e;
          exit 1)
