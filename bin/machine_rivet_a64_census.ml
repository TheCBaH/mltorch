(* A model's bundle compiled to AArch64 images through typed Rivet modules and
   run on this CPU. Every invocation is selected, allocated, framed, published
   and loaded ([Mir_model.prepare] on the native route), and every refusal is
   tallied by its reason. Then the first K invocations (all by default) run on
   the archive's own weights and a deterministic synthetic input, and every
   tensor they write is compared with the per-node reference, bit for bit.

   Phases are timed separately and never summed: preparing (SSA, Machine IR,
   selection, allocation, publication, Rivet lowering and layout, mapping the
   image) and running (copying storage into the images, the calls, copying
   back). The run is dominated by the copy and the reference, not the kernels.

   argv: <model.pt2> [--run=K] (K = 0: compile only) [--allocation=reference|scanned]
   [--gnu] [--table [--poison] [--planned=ordered|relaxed]] [--runtime=dependency_free|system_libm] (default dependency_free: a kernel
   calling a C library math helper is refused and tallied) *)

open Loop_ir
module M = Machine_model.Mir_model
module Rt = Machine_rivet_aarch64.Rivet_a64_route

(* GNU assembles and links each invocation's modules where Rivet bound them; a
   disagreement refuses the invocation, so the tally names it. *)
let gnu_check ~entry modules =
  let module G = Machine_rivet_aarch64_gnu.Gnu_coherence in
  match G.check ~entry modules with
  | G.Verdict.Agree _ -> Ok ()
  | v -> Error (Fmt.str "gnu: %a" G.Verdict.pp v)

let bits (Tensor.Tensor t as packed) =
  let acc = ref [] in
  Vec6.iter t.Tensor.shape (fun c ->
      acc := Int64.bits_of_float (Tensor.read packed c) :: !acc);
  List.rev !acc

let prefix (b : Loop_bundle.t) m ~constants ~count =
  let g = b.Loop_bundle.graph in
  let inputs =
    List.map
      (fun id ->
        let sg = Graph_ir.Tensor_id.Map.find id g.Graph_ir.Graph.tensors in
        ( id,
          Tensor.materialize sg.Tensor_sig.shape (fun c ->
              let k = (Vec6.offset sg.Tensor_sig.shape c :> int) in
              float_of_int ((k mod 17) - 8) /. 8.) ))
      b.Loop_bundle.inputs
  in
  let t0 = Unix.gettimeofday () in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g
         ~constants:(Graph_ir.Tensor_id.Map.bindings constants)
         ~inputs)
  in
  Fmt.pr "reference: %.1f s@." (Unix.gettimeofday () -. t0);
  match
    M.Context.create m ~constants:(fun id ->
        Graph_ir.Tensor_id.Map.find_opt id constants)
  with
  | Error s -> Fmt.pr "context: %a@." M.Stop.pp s
  | Ok cx -> (
      let t0 = Unix.gettimeofday () in
      match
        M.Context.run_prefix ~fuel:Int64.max_int cx
          ~inputs:(fun id -> List.assoc_opt id inputs)
          ~count
      with
      | Error s -> Fmt.pr "run: %a@." M.Stop.pp s
      | Ok written ->
          let seconds = Unix.gettimeofday () -. t0 in
          let differing =
            List.filter
              (fun id ->
                match M.Context.tensor cx id with
                | Ok t -> (
                    match Graph_ir.Tensor_id.Map.find_opt id reference with
                    | Some r -> bits t <> bits r
                    | None -> true)
                | Error _ -> true)
              written
          in
          Fmt.pr "first %d invocations: %d tensors, %d differ (%.1f s)@." count
            (List.length written) (List.length differing) seconds;
          List.iter
            (fun id -> Fmt.pr "  differs: %a@." Graph_ir.Tensor_id.pp id)
            differing)

(* The table-bound host: tensors are the context's storage, passed by address.
   The same comparison, with the phases timed apart. *)
let host_run (b : Loop_bundle.t) ~constants ~allocation ~runtime ~count ~poison
    ~gnu ~pipeline ~interp ~bench ~profile =
  let module H = Machine_rivet_aarch64.Rivet_a64_host in
  let n = List.length b.Loop_bundle.invocations in
  let t0 = Unix.gettimeofday () in
  match
    H.prepare
      ?check:(if gnu then Some gnu_check else None)
      ~allocation ~runtime ~pipeline b
  with
  | Error refusals ->
      let tally = Hashtbl.create 16 in
      List.iter
        (fun (r : M.Refusal.t) ->
          let k = Fmt.str "%a" M.Reason.pp r.M.Refusal.reason in
          Hashtbl.replace tally k
            (1 + Option.value ~default:0 (Hashtbl.find_opt tally k)))
        refusals;
      Fmt.pr "%d of %d invocations compiled (%.1f s)@."
        (n - List.length refusals)
        n
        (Unix.gettimeofday () -. t0);
      List.iter
        (fun (k, c) -> Fmt.pr "  %4d  %s@." c k)
        (List.sort
           (fun (a, x) (b, y) ->
             match compare y x with 0 -> compare a b | c -> c)
           (List.of_seq (Hashtbl.to_seq tally)))
  | Ok _ when count = 0 ->
      Fmt.pr "%d of %d invocations compiled (%.1f s)@." n n
        (Unix.gettimeofday () -. t0)
  | Ok host -> (
      Fmt.pr "%d of %d invocations compiled (%.1f s)@." n n
        (Unix.gettimeofday () -. t0);
      let g = b.Loop_bundle.graph in
      let inputs =
        List.map
          (fun id ->
            let sg = Graph_ir.Tensor_id.Map.find id g.Graph_ir.Graph.tensors in
            ( id,
              Tensor.materialize sg.Tensor_sig.shape (fun c ->
                  let k = (Vec6.offset sg.Tensor_sig.shape c :> int) in
                  float_of_int ((k mod 17) - 8) /. 8.) ))
          b.Loop_bundle.inputs
      in
      let t0 = Unix.gettimeofday () in
      let reference =
        Err.or_raise ~pp_error:Eval_direct.pp_error
          (Eval_direct.run g
             ~constants:(Graph_ir.Tensor_id.Map.bindings constants)
             ~inputs)
      in
      Fmt.pr "reference: %.1f s@." (Unix.gettimeofday () -. t0);
      let t0 = Unix.gettimeofday () in
      match
        H.Context.create host ~constants:(fun id ->
            Graph_ir.Tensor_id.Map.find_opt id constants)
      with
      | Error s -> Fmt.pr "context: %a@." M.Stop.pp s
      | Ok cx -> (
          Fmt.pr "context: %.2f s@." (Unix.gettimeofday () -. t0);
          let t0 = Unix.gettimeofday () in
          match
            H.Context.run_prefix ~poison cx
              ~inputs:(fun id -> List.assoc_opt id inputs)
              ~count
          with
          | Error s -> Fmt.pr "run: %a@." M.Stop.pp s
          | Ok written ->
              let seconds = Unix.gettimeofday () -. t0 in
              let exact =
                match pipeline with
                | Ssa_backends.Pipeline.Planned _ -> false
                | _ -> true
              in
              let worst = ref 0. in
              let differing =
                List.filter
                  (fun id ->
                    match H.Context.tensor cx id with
                    | Ok t -> (
                        match Graph_ir.Tensor_id.Map.find_opt id reference with
                        | Some r ->
                            if exact then bits t <> bits r
                            else begin
                              List.iter2
                                (fun x y ->
                                  let x = Int64.float_of_bits x
                                  and y = Int64.float_of_bits y in
                                  let d =
                                    Float.abs (x -. y)
                                    /. Float.max 1. (Float.abs y)
                                  in
                                  worst :=
                                    if Float.is_nan d then Float.infinity
                                    else Float.max !worst d)
                                (bits t) (bits r);
                              false
                            end
                        | None -> true)
                    | Error _ -> true)
                  written
              in
              Fmt.pr "first %d invocations: %d tensors, %d differ (%.2f s)@."
                count (List.length written) (List.length differing) seconds;
              if not exact then
                Fmt.pr "  largest relative difference from binary64: %.3g@."
                  !worst;
              (match bench with
              | None -> ()
              | Some reps ->
                  (* warm calls of the whole schedule: inputs rewritten, every
                     invocation called, outputs left in the context *)
                  let times =
                    List.init reps (fun _ ->
                        let t0 = Unix.gettimeofday () in
                        ignore
                          (H.Context.run_prefix cx
                             ~inputs:(fun id -> List.assoc_opt id inputs)
                             ~count);
                        (Unix.gettimeofday () -. t0) *. 1000.)
                    |> List.sort compare
                  in
                  Fmt.pr
                    "  warm: min %.3f ms, median %.3f ms, max %.3f ms over %d \
                     calls@."
                    (List.hd times)
                    (List.nth times (reps / 2))
                    (List.nth times (reps - 1))
                    reps);
              (match profile with
              | None -> ()
              | Some reps ->
                  (* the cost of each invocation: the cumulative time of the
                     first k invocations, differenced, the minimum of [reps] *)
                  let cumulative k =
                    List.init reps (fun _ ->
                        let t0 = Unix.gettimeofday () in
                        ignore
                          (H.Context.run_prefix cx
                             ~inputs:(fun id -> List.assoc_opt id inputs)
                             ~count:k);
                        Unix.gettimeofday () -. t0)
                    |> List.fold_left Float.min Float.infinity
                  in
                  let total = cumulative count in
                  let previous = ref (cumulative 0) in
                  let costs =
                    List.init count (fun k ->
                        let t = cumulative (k + 1) in
                        let d = t -. !previous in
                        previous := t;
                        (k, d *. 1000.))
                  in
                  Fmt.pr "  profile: whole schedule %.3f ms@." (total *. 1000.);
                  List.iteri
                    (fun rank (k, ms) ->
                      if rank < 12 then begin
                        Fmt.pr "    invocation %d: %.3f ms@." k ms;
                        if rank < 3 then
                          Fmt.pr "%a@." Loop_pp.program
                            (List.nth b.Loop_bundle.invocations k)
                              .Loop_bundle.program
                      end)
                    (List.sort (fun (_, a) (_, b) -> compare b a) costs));
              if interp then
                (* the same planned program on the selected-stage interpreter:
                   what the native code must equal bit for bit *)
                begin match
                  M.prepare
                    ~route:
                      (M.Route.Aarch64
                         Machine_model.Mir_model_route.Stage.Selected) ~pipeline
                    b
                with
                | Error _ -> Fmt.pr "  interpreter route refused@."
                | Ok mi -> (
                    match
                      M.Context.create mi ~constants:(fun id ->
                          Graph_ir.Tensor_id.Map.find_opt id constants)
                    with
                    | Error s ->
                        Fmt.pr "  interpreter context: %a@." M.Stop.pp s
                    | Ok ci -> (
                        let t0 = Unix.gettimeofday () in
                        match
                          M.Context.run_prefix ~fuel:Int64.max_int ci
                            ~inputs:(fun id -> List.assoc_opt id inputs)
                            ~count
                        with
                        | Error s ->
                            Fmt.pr "  interpreter run: %a@." M.Stop.pp s
                        | Ok written ->
                            let differ =
                              List.filter
                                (fun id ->
                                  match
                                    ( M.Context.tensor ci id,
                                      H.Context.tensor cx id )
                                  with
                                  | Ok a, Ok b -> bits a <> bits b
                                  | _ -> true)
                                written
                            in
                            Fmt.pr
                              "  against the selected-stage interpreter: %d \
                               tensors, %d differ (%.1f s)@."
                              (List.length written) (List.length differ)
                              (Unix.gettimeofday () -. t0)))
                end;
              List.iter
                (fun id -> Fmt.pr "  differs: %a@." Graph_ir.Tensor_id.pp id)
                differing))

let () =
  let run = ref None
  and allocation = ref Rt.Allocation.Reference
  and runtime = ref Machine_rivet_aarch64.Rivet_a64_runtime.Dependency_free
  and gnu = ref false
  and table = ref false
  and planned = ref None
  and interp = ref false
  and bench = ref None
  and profile = ref None
  and poison = ref false in
  let args =
    List.filter
      (fun a ->
        match String.split_on_char '=' a with
        | [ "--run"; k ] ->
            run := Some (int_of_string k);
            false
        | [ "--allocation"; "reference" ] ->
            allocation := Rt.Allocation.Reference;
            false
        | [ "--allocation"; "scanned" ] ->
            allocation := Rt.Allocation.Scanned;
            false
        | [ "--planned"; (("ordered" | "relaxed") as n) ] ->
            planned :=
              Some
                (if n = "ordered" then Ssa_ir.Ssa_numerics.Simd_fp32_ordered
                 else Ssa_ir.Ssa_numerics.Simd_fp32_relaxed);
            false
        | [ "--profile"; n ] ->
            profile := Some (int_of_string n);
            false
        | [ "--bench"; n ] ->
            bench := Some (int_of_string n);
            false
        | [ "--against-interp" ] ->
            interp := true;
            false
        | [ "--table" ] ->
            table := true;
            false
        | [ "--poison" ] ->
            poison := true;
            false
        | [ "--gnu" ] ->
            gnu := true;
            false
        | [ "--runtime"; "dependency_free" ] ->
            runtime := Machine_rivet_aarch64.Rivet_a64_runtime.Dependency_free;
            false
        | [ "--runtime"; "system_libm" ] ->
            runtime := Machine_rivet_aarch64.Rivet_a64_runtime.System_libm;
            false
        | _ -> true)
      (List.tl (Array.to_list Sys.argv))
  in
  match args with
  | [ path ] -> (
      let result =
        let open Err.Syntax in
        let lift e =
          (e :> [ Pt2_archive.error | Native_interp.error | Loop_bundle.error ])
        in
        let* archive = Pt2_archive.open_pt2 path |> Err.map_error lift in
        let* lowered =
          Native_interp.lower_archive archive |> Err.map_error lift
        in
        let* constants =
          Native_interp.preload archive lowered |> Err.map_error lift
        in
        let* b =
          Loop_bundle.build lowered.Pt2_native_graph.graph |> Err.map_error lift
        in
        Err.return (b, constants)
      in
      match Err.payload result with
      | Error _ ->
          prerr_endline "census: could not open or lower the archive";
          exit 1
      | Ok (b, constants) -> (
          let n = List.length b.Loop_bundle.invocations in
          if !table then
            host_run b ~constants ~allocation:!allocation ~runtime:!runtime
              ~count:(Option.value ~default:n !run)
              ~poison:!poison ~gnu:!gnu ~interp:!interp ~bench:!bench
              ~profile:!profile
              ~pipeline:
                (match !planned with
                | Some numerics ->
                    Ssa_backends.Pipeline.Planned
                      { numerics; target = Ssa_ir.Ssa_target.neon128 }
                | None -> Ssa_backends.Pipeline.Exact)
          else
            let t0 = Unix.gettimeofday () in
            match
              M.prepare
                ~route:
                  (Rt.route
                     ?check:(if !gnu then Some gnu_check else None)
                     ~allocation:!allocation ~runtime:!runtime ())
                ~pipeline:Ssa_backends.Pipeline.Exact b
            with
            | Ok m ->
                Fmt.pr "%s: %d of %d invocations compiled (%.1f s)@."
                  (Filename.basename path) n n
                  (Unix.gettimeofday () -. t0);
                let count = Option.value ~default:n !run in
                if count > 0 then prefix b m ~constants ~count
            | Error refusals ->
                let tally = Hashtbl.create 16 in
                List.iter
                  (fun (r : M.Refusal.t) ->
                    let k = Fmt.str "%a" M.Reason.pp r.M.Refusal.reason in
                    Hashtbl.replace tally k
                      (1 + Option.value ~default:0 (Hashtbl.find_opt tally k)))
                  refusals;
                Fmt.pr "%s: %d of %d invocations compiled (%.1f s)@."
                  (Filename.basename path)
                  (n - List.length refusals)
                  n
                  (Unix.gettimeofday () -. t0);
                List.iter
                  (fun (k, c) -> Fmt.pr "  %4d  %s@." c k)
                  (List.sort
                     (fun (a, x) (b, y) ->
                       match compare y x with 0 -> compare a b | c -> c)
                     (List.of_seq (Hashtbl.to_seq tally)))))
  | _ ->
      prerr_endline
        "usage: machine_rivet_a64_census <model.pt2> [--run=K] \
         [--allocation=reference|scanned] [--gnu] [--table [--poison] \
         [--planned=ordered|relaxed]] [--runtime=dependency_free|system_libm]";
      exit 2
