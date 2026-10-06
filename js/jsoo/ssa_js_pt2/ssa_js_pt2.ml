(* A real model's whole bundle through the SSA backend's generated JavaScript,
   in-process under node, checked against the release's own top-5 rankings.

   argv: <model.pt2> <inputs.pt> <expected.json> <outputs.pt>
         [--strict] [--ssa=representation|exact|planned] [--shadow]
         [--reference] [--samples=N]

   [--ssa] picks the pipeline that makes each invocation's kernel ([planned] is
   the scalar binary64 plan: the JavaScript emitter refuses vectors and
   binary32). [--shadow] also runs the same graph through the Loop emitter's
   bundle ([Loop_bundle_exec] with its default kernels, itself checked bitwise
   against [Eval_direct] by loop_js_pt2) and requires every output to be bitwise
   equal; with [--reference] the shadow is [Eval_direct.run] instead, the
   independent evaluator. Like loop_js_pt2, only the first sample the files hold is run unless
   [--samples] asks for more. *)

open Graph_ir
open Loop_ir

type eval =
  [ Native_interp.error
  | Native_predict.error
  | Loop_bundle.error
  | Loop_bundle_exec.error
  | `Bundle_mismatch of int ]

let pp_eval ppf : eval -> unit = function
  | `Bundle_mismatch n ->
      Format.fprintf ppf "ssa bundle: %d output(s) differ bitwise from Loop's" n
  | #Loop_bundle_exec.error as e -> Loop_bundle_exec.pp_error ppf e
  | #Loop_bundle.error as e -> Loop_bundle.pp_error ppf e
  | #Native_predict.error as e -> Native_predict.pp_error ppf e
  | #Native_interp.error as e -> Native_interp.pp_error ppf e

let output_values t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  !acc

(* Lowered, preloaded and prepared once: every sample shares the one graph. *)
let cache = ref None

let prepared ~kernel archive =
  let open Err.Syntax in
  let map e = (e :> eval) in
  match !cache with
  | Some cached -> Err.return (false, cached)
  | None ->
      let* lowered = Native_interp.lower_archive archive |> Err.map_error map in
      let g = lowered.Pt2_native_graph.graph in
      let* constants =
        Native_interp.preload archive lowered |> Err.map_error map
      in
      let* b = Loop_bundle.build g |> Err.map_error map in
      let constants_of id = Tensor_id.Map.find_opt id constants in
      let t0 = Sys.time () in
      let* p =
        Loop_bundle_exec.prepare ~kernel b ~constants:constants_of
        |> Err.map_error map
      in
      let ssa_ms = (Sys.time () -. t0) *. 1000. in
      let* loop =
        Loop_bundle_exec.prepare b ~constants:constants_of |> Err.map_error map
      in
      let cached = (g, b, p, loop, ssa_ms, constants) in
      cache := Some cached;
      Err.return (true, cached)

let infer ~kernel ~shadow ~reference archive image =
  let open Err.Syntax in
  let map e = (e :> eval) in
  let* first, (g, b, p, loop, ssa_ms, constants) = prepared ~kernel archive in
  let* input = Native_interp.tensor_of_pt2 image |> Err.map_error map in
  let input_id = List.hd b.Loop_bundle.inputs in
  let bind id = if Tensor_id.equal id input_id then Some input else None in
  let t0 = Sys.time () in
  let* out = Loop_bundle_exec.run p ~bind |> Err.map_error map in
  let run_ms = (Sys.time () -. t0) *. 1000. in
  if first then
    Printf.eprintf
      "ssa_js_pt2: %d invocations, prepare %.1f ms, first run %.1f ms\n%!"
      (List.length b.Loop_bundle.invocations)
      ssa_ms run_ms;
  let outputs =
    List.map (fun id -> Tensor_id.Map.find id out) g.Graph.outputs
  in
  let* () =
    if not shadow then Err.return ()
    else
      let* reference =
        if reference then
          Eval_direct.run g
            ~constants:(Tensor_id.Map.bindings constants)
            ~inputs:[ (input_id, input) ]
          |> Err.map_error map
        else Loop_bundle_exec.run loop ~bind |> Err.map_error map
      in
      let bad =
        List.length
          (List.filter
             (fun id ->
               output_values (Tensor_id.Map.find id out)
               <> output_values (Tensor_id.Map.find id reference))
             g.Graph.outputs)
      in
      Printf.eprintf "ssa_js_pt2: shadow %d/%d outputs differ\n%!" bad
        (List.length g.Graph.outputs);
      if bad = 0 then Err.return () else Err.fail (`Bundle_mismatch bad)
  in
  let* top = Native_predict.top_predictions outputs 5 |> Err.map_error map in
  Err.return (List.map (fun ((c : Dim.index Dim.t), p) -> ((c :> int), p)) top)

let strip_flag flag argv =
  ( Array.exists (String.equal flag) argv,
    Array.of_list
      (List.filter (fun a -> not (String.equal a flag)) (Array.to_list argv)) )

let strip_valued prefix argv =
  let n = String.length prefix in
  let is a = String.length a >= n && String.sub a 0 n = prefix in
  ( Array.fold_left
      (fun acc a ->
        if is a then Some (String.sub a n (String.length a - n)) else acc)
      None argv,
    Array.of_list (List.filter (fun a -> not (is a)) (Array.to_list argv)) )

let () =
  let shadow, argv = strip_flag "--shadow" Sys.argv in
  let reference, argv = strip_flag "--reference" argv in
  let samples, argv = strip_valued "--samples=" argv in
  let ssa, argv = strip_valued "--ssa=" argv in
  let pipeline =
    match ssa with
    | None | Some "exact" -> Ssa_backends.Pipeline.Exact
    | Some "representation" -> Ssa_backends.Pipeline.Representation
    | Some "planned" ->
        Ssa_backends.Pipeline.Planned
          {
            numerics = Ssa_ir.Ssa_numerics.Reference_f64;
            target = Ssa_ir.Ssa_target.scalar;
          }
    | Some other ->
        Printf.eprintf
          "ssa_js_pt2: unknown --ssa=%S (representation, exact or planned)\n"
          other;
        exit 2
  in
  let kernel inv = Ssa_backends.js ~pipeline inv in
  match Infer_report.parse_argv argv with
  | Error usage ->
      prerr_endline usage;
      exit 2
  | Ok (paths, options) -> (
      let max_samples = Option.fold ~none:1 ~some:int_of_string samples in
      match
        Infer_report.run ~max_samples ~now:Sys.time
          ~infer:(infer ~kernel ~shadow ~reference)
          paths options
      with
      | Ok () -> ()
      | Error e ->
          Format.eprintf "%a@." (Err.Error.pp (Infer_report.pp_error pp_eval)) e;
          exit 1)
