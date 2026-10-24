(* A model's bundle on a Coral NPU simulator: every invocation planned
   binary32 ([Fp32_ordered]) for Zve32f at VLEN 128, its recognized matrix
   products made generated helper calls (with [--contractions]: on the matrix
   engine where the configuration has one, else the vector unit), built into
   its own image with the tensors in the configuration's RAM, and run once on
   the archive's own weights and a deterministic synthetic input; every graph
   output is compared with the binary32 oracle bit for bit, and the cycles of
   every invocation are reported.

   [--prepack] reads each Linear's weight as a packed copy with contiguous
   columns, built once before the first invocation.

   argv: <model.pt2> --dir=<extracted emulators archive>
   --configuration=<name> [--contractions] [--prepack] *)

open Loop_ir
module M = Machine_model.Mir_model
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral

let bits (Tensor.Tensor t as packed) =
  let acc = ref [] in
  Vec6.iter t.Tensor.shape (fun c ->
      acc := Int64.bits_of_float (Tensor.read packed c) :: !acc);
  List.rev !acc

let same a b =
  List.for_all2
    (fun x y ->
      Core.Float_bits.equal_portable (Int64.float_of_bits x)
        (Int64.float_of_bits y))
    (bits a) (bits b)

let home (s : Machine_rivet_common.Rivet_table.slot) =
  match s.Machine_rivet_common.Rivet_table.section with
  | Machine_model.Mir_artifact.Section.Bound -> Coral_layout.Home.Ram
  | _ -> Coral_layout.Home.Any

let () =
  let dir = ref None
  and configuration = ref None
  and contractions = ref false
  and prepack = ref false in
  let args =
    List.filter
      (fun a ->
        match String.split_on_char '=' a with
        | [ "--dir"; d ] ->
            dir := Some d;
            false
        | [ "--configuration"; c ] ->
            configuration := Some c;
            false
        | [ "--contractions" ] ->
            contractions := true;
            false
        | [ "--prepack" ] ->
            prepack := true;
            false
        | _ -> true)
      (List.tl (Array.to_list Sys.argv))
  in
  match (args, !dir, !configuration) with
  | [ path ], Some dir, Some configuration -> (
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
        Err.return (b, constants, lowered.Pt2_native_graph.graph)
      in
      match Err.payload result with
      | Error _ ->
          prerr_endline "coral model: could not open or lower the archive";
          exit 1
      | Ok (b, constants, graph) -> (
          let config =
            {
              C.Config.dir;
              configuration;
              deadline = 3600.;
              flags = [ "--cycles=2000000000" ];
            }
          in
          let runs = ref [] in
          let on_run (o : C.Run.t) = runs := o :: !runs in
          let pipeline =
            Ssa_backends.Pipeline.Planned
              {
                numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
                target = Ssa_ir.Ssa_target.zve32f128;
              }
          in
          let t0 = Unix.gettimeofday () in
          match
            M.prepare
              ~route:(C.route ~home ~on_run config)
              ~contractions:!contractions ~prepack:!prepack ~pipeline b
          with
          | Error refusals ->
              List.iter
                (fun (r : M.Refusal.t) ->
                  Fmt.pr "refused: %a@." M.Reason.pp r.M.Refusal.reason)
                refusals;
              exit 1
          | Ok m -> (
              Fmt.pr "%s on %s: %d invocations compiled (%.1f s)@."
                (Filename.basename path) configuration
                (List.length b.Loop_bundle.invocations)
                (Unix.gettimeofday () -. t0);
              let inputs =
                List.map
                  (fun id ->
                    let sg =
                      Graph_ir.Tensor_id.Map.find id
                        graph.Graph_ir.Graph.tensors
                    in
                    ( id,
                      Tensor.materialize sg.Tensor_sig.shape (fun c ->
                          let k = (Vec6.offset sg.Tensor_sig.shape c :> int) in
                          float_of_int ((k mod 17) - 8) /. 8.) ))
                  b.Loop_bundle.inputs
              in
              let constant id = Graph_ir.Tensor_id.Map.find_opt id constants in
              let oracle =
                Err.or_raise ~pp_error:Loop_bundle_oracle.pp_error
                  (Loop_bundle_oracle.run b ~constants:constant
                     ~inputs:(fun id -> List.assoc_opt id inputs))
              in
              match M.Context.create m ~constants:constant with
              | Error s ->
                  Fmt.pr "context: %a@." M.Stop.pp s;
                  exit 1
              | Ok cx -> (
                  let t0 = Unix.gettimeofday () in
                  match
                    M.Context.run cx ~inputs:(fun id ->
                        List.assoc_opt id inputs)
                  with
                  | Error s ->
                      Fmt.pr "run: %a@." M.Stop.pp s;
                      exit 1
                  | Ok outs ->
                      let runs = List.rev !runs in
                      let differ =
                        List.length
                          (List.filter not (List.map2 same outs oracle))
                      in
                      List.iteri
                        (fun i (o : C.Run.t) ->
                          Fmt.pr "invocation %d: %Ld cycles@." i o.C.Run.mcycle)
                        runs;
                      Fmt.pr
                        "%d runs, %Ld cycles; %d outputs, %d differ from the \
                         oracle (%.1f s)@."
                        (List.length runs)
                        (List.fold_left
                           (fun a (o : C.Run.t) -> Int64.add a o.C.Run.mcycle)
                           0L runs)
                        (List.length outs) differ
                        (Unix.gettimeofday () -. t0);
                      if differ > 0 then exit 1))))
  | _ ->
      prerr_endline
        "usage: machine_rivet_coral_model <model.pt2> --dir=DIR \
         --configuration=NAME [--contractions] [--prepack]";
      exit 2
