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
   [--runtime=dependency_free|system_libm] (default dependency_free: a kernel
   calling a C library math helper is refused and tallied) *)

open Loop_ir
module M = Machine_model.Mir_model
module Rt = Machine_rivet_aarch64.Rivet_a64_route

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

let () =
  let run = ref None
  and allocation = ref Rt.Allocation.Reference
  and runtime = ref Machine_rivet_aarch64.Rivet_a64_runtime.Dependency_free in
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
          let t0 = Unix.gettimeofday () in
          match
            M.prepare
              ~route:(Rt.route ~allocation:!allocation ~runtime:!runtime ())
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
         [--allocation=reference|scanned] \
         [--runtime=dependency_free|system_libm]";
      exit 2
