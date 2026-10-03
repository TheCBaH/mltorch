(* The whole-model Wasm backend generated AND executed in JavaScript: a real
   downloaded model is lowered, bundled and emitted as a WebAssembly module by
   the js_of_ocaml build of the compiler, instantiated through node's own
   [WebAssembly] and run, with no native process and no external compiler.
   js_of_ocaml only; its native counterpart is bin/loop_wasm_pt2.

   argv: <model.pt2> <inputs.pt> <expected.json> <outputs.pt> [--strict]
         [--shadow] [--samples=N] [--repeat=N]

   Prints, on stdout, the module's identity (the digest of its bytes), so a
   native and a JS run of the same model can be diffed: the two compilers must
   emit the same bytes. [--shadow] also runs [Eval_direct.run] and requires
   every graph output bitwise equal. [--repeat=N] runs the schedule N more
   times on the same instance and requires the same outputs. Timings go to
   stderr. *)

open Js_of_ocaml

type eval =
  [ Native_interp.error
  | Native_predict.error
  | Loop_bundle.error
  | Loop_wasm_host.error
  | `Bundle_mismatch of int
  | `Repeat_mismatch of int ]

let pp_eval ppf : eval -> unit = function
  | `Bundle_mismatch n ->
      Format.fprintf ppf
        "Wasm model: %d output(s) differ bitwise from the reference" n
  | `Repeat_mismatch n ->
      Format.fprintf ppf "Wasm model: run %d differs from the first" n
  | #Loop_wasm_host.error as e -> Loop_wasm_host.pp_error ppf e
  | #Loop_bundle.error as e -> Loop_bundle.pp_error ppf e
  | #Native_predict.error as e -> Native_predict.pp_error ppf e
  | #Native_interp.error as e -> Native_interp.pp_error ppf e

let bits t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  !acc

let strip_flag flag argv =
  ( Array.exists (String.equal flag) argv,
    Array.of_list (List.filter (fun a -> a <> flag) (Array.to_list argv)) )

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

let now () = Sys.time ()
let ms t0 = (now () -. t0) *. 1000.
let cache = ref None

let prepared archive =
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
      let* m =
        Loop_wasm_host.prepare b ~constants:(fun id ->
            Graph_ir.Tensor_id.Map.find_opt id constants)
        |> Err.map_error map
      in
      let w = Loop_wasm_host.bundle_wasm m in
      let st = w.Loop_bundle_wasm.stats in
      Printf.eprintf
        "loop_wasm_pt2_js: %d invocations, %d distinct kernels, module %d \
         bytes, memory %d bytes; lower+bundle %.0f ms, generate+compile+place \
         %.0f ms\n\
         %!"
        st.Loop_bundle_wasm.invocations st.Loop_bundle_wasm.distinct_kernels
        st.Loop_bundle_wasm.module_bytes
        (Loop_wasm_host.memory_bytes m)
        ((t1 -. t0) *. 1000.)
        (ms t1);
      Printf.printf "module identity %s\n%!"
        (Digest.to_hex w.Loop_bundle_wasm.identity);
      let cached = (g, constants, b, m) in
      cache := Some cached;
      Err.return cached

let infer ~shadow ~repeat archive image =
  let open Err.Syntax in
  let map e = (e :> eval) in
  let* g, constants, b, m = prepared archive in
  let* input = Native_interp.tensor_of_pt2 image |> Err.map_error map in
  let input_id = List.hd b.Loop_bundle.inputs in
  let bind id =
    if Graph_ir.Tensor_id.equal id input_id then Some input else None
  in
  let t0 = now () in
  let* outs = Loop_wasm_host.run m ~bind |> Err.map_error map in
  Printf.eprintf "loop_wasm_pt2_js: first run %.1f ms\n%!" (ms t0);
  let first = List.map bits outs in
  let* warm =
    Err.List.map
      (fun k ->
        let t = now () in
        let* again = Loop_wasm_host.run m ~bind |> Err.map_error map in
        let dt = ms t in
        if List.map bits again = first then Err.return dt
        else Err.fail (`Repeat_mismatch (k + 1)))
      (List.init repeat Fun.id)
  in
  if warm <> [] then
    Printf.eprintf "loop_wasm_pt2_js: warm runs ms [%s]\n%!"
      (String.concat "; "
         (List.map (Printf.sprintf "%.1f") (List.sort compare warm)));
  let* () =
    if not shadow then Err.return ()
    else
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
      Printf.eprintf "loop_wasm_pt2_js: shadow %d/%d outputs differ\n%!" bad
        (List.length outs);
      if bad = 0 then Err.return () else Err.fail (`Bundle_mismatch bad)
  in
  let* top = Native_predict.top_predictions outs 5 |> Err.map_error map in
  Err.return (List.map (fun ((c : Dim.index Dim.t), p) -> ((c :> int), p)) top)

let () =
  ignore Js.null;
  let shadow, argv = strip_flag "--shadow" Sys.argv in
  let samples, argv = valued "--samples=" argv in
  let repeat, argv = valued "--repeat=" argv in
  let repeat = Option.fold ~none:0 ~some:int_of_string repeat in
  match Infer_report.parse_argv argv with
  | Error usage ->
      prerr_endline usage;
      exit 2
  | Ok (paths, options) -> (
      match
        Err.payload
          (Infer_report.run
             ~max_samples:(Option.fold ~none:1 ~some:int_of_string samples)
             ~now ~infer:(infer ~shadow ~repeat) paths options)
      with
      | Ok () -> ()
      | Error e ->
          Format.eprintf "loop_wasm_pt2_js: %a@."
            (Infer_report.pp_error pp_eval)
            e;
          exit 1)
