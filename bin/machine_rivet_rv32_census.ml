(* A model's bundle compiled to RV32IMF processes through typed Rivet modules
   and run under qemu-riscv32. Every invocation is planned binary32
   ([Fp32_ordered]) for the scalar target, selected onto ILP32, allocated,
   framed, published and built into its own static ELF; every refusal is
   tallied by its reason. Then the whole schedule runs once on the archive's
   own weights and a deterministic synthetic input, one process per
   invocation, and every graph output is compared with the binary32 oracle
   (the Loop interpreter at binary32 over the bundle), bit for bit.

   [--contractions] lists the matrix products the contraction recognizer
   finds in each invocation's scalar binary32 form, with a tally of why it
   refuses every other ordered sum, and stops there.

   [--planned=zve32f] plans for Zve32f at VLEN 128 instead of the scalar
   target, so vector loops run as RVV forms; [--vlen=N] runs the processes at
   that VLEN, tail and masked-off elements filled with ones.

   argv: <model.pt2> [--run=0] (compile only) [--allocation=reference|scanned]
   [--planned=zve32f] [--vlen=N] [--contractions] *)

open Loop_ir
module M = Machine_model.Mir_model
module Rt = Machine_rivet_riscv32_mir.Rv32_mir_route

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

(* How many selected forms are vector forms, over every invocation. *)
let vector_census m =
  let module S = Machine_target_riscv32.Rv32_select in
  let vec = ref 0 and all = ref 0 in
  List.iter
    (fun g ->
      match
        Err.payload
          (S.program ~unlisted:Machine_ir.Mir_failure.Unlisted.Unreachable g)
      with
      | Error _ -> ()
      | Ok r ->
          let p =
            (Machine_target_riscv32.Rv32_stage.Sel.Verified.selected
               r.S.selected)
              .Machine_target_riscv32.Rv32_stage.Sel.program
          in
          List.iter
            (fun (f : (_, _) Machine_ir.Mir_func.t) ->
              List.iter
                (fun (b : (_, _) Machine_ir.Mir_block.t) ->
                  List.iter
                    (fun (i : _ Machine_ir.Mir_instr.t) ->
                      match i.Machine_ir.Mir_instr.op with
                      | Machine_ir.Mir_sel.Op.Machine op -> (
                          incr all;
                          match op with
                          | Machine_target_riscv32.Rv32_op.Vec _ -> incr vec
                          | _ -> ())
                      | _ -> ())
                    b.Machine_ir.Mir_block.body)
                f.Machine_ir.Mir_func.blocks)
            p.Machine_ir.Mir_program.funcs)
    (M.generic m);
  Fmt.pr "selected forms: %d, of which vector %d@." !all !vec

(* The contraction census: each found product, then the refusals by reason. *)
let contractions (b : Loop_bundle.t) =
  let tally = Hashtbl.create 8 and found = ref 0 in
  List.iteri
    (fun i inv ->
      match Ssa_backends.program Ssa_backends.Pipeline.Exact inv with
      | Error e -> Fmt.pr "invocation %d: %s@." i e
      | Ok p ->
          List.iter
            (function
              | Ssa_ir.Ssa_contract.Outcome.Found f ->
                  incr found;
                  Fmt.pr "invocation %d: %a@." i Contraction.pp
                    f.Ssa_ir.Ssa_contract.Found.desc
              | Ssa_ir.Ssa_contract.Outcome.Refused r ->
                  let k = Ssa_ir.Ssa_contract.Refusal.name r in
                  Hashtbl.replace tally k
                    (1 + Option.value ~default:0 (Hashtbl.find_opt tally k)))
            (Ssa_ir.Ssa_contract.census ~numerics:Contraction.Numerics.Ordered
               (Ssa_ir.Ssa_plan.binary32_scalar p)))
    b.Loop_bundle.invocations;
  Fmt.pr "contractions found: %d@." !found;
  List.iter
    (fun (k, n) -> Fmt.pr "  refused %4d  %s@." n k)
    (List.sort compare (List.of_seq (Hashtbl.to_seq tally)))

let run_model (b : Loop_bundle.t) ~(graph : Graph_ir.graph) m ~constants =
  let inputs =
    List.map
      (fun id ->
        let sg = Graph_ir.Tensor_id.Map.find id graph.Graph_ir.Graph.tensors in
        ( id,
          Tensor.materialize sg.Tensor_sig.shape (fun c ->
              let k = (Vec6.offset sg.Tensor_sig.shape c :> int) in
              float_of_int ((k mod 17) - 8) /. 8.) ))
      b.Loop_bundle.inputs
  in
  let t0 = Unix.gettimeofday () in
  let oracle =
    Err.or_raise ~pp_error:Loop_bundle_oracle.pp_error
      (Loop_bundle_oracle.run b
         ~constants:(fun id -> Graph_ir.Tensor_id.Map.find_opt id constants)
         ~inputs:(fun id -> List.assoc_opt id inputs))
  in
  Fmt.pr "binary32 oracle: %.1f s@." (Unix.gettimeofday () -. t0);
  match
    M.Context.create m ~constants:(fun id ->
        Graph_ir.Tensor_id.Map.find_opt id constants)
  with
  | Error s ->
      Fmt.pr "context: %a@." M.Stop.pp s;
      exit 1
  | Ok cx -> (
      let t0 = Unix.gettimeofday () in
      match M.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs) with
      | Error s ->
          Fmt.pr "run: %a@." M.Stop.pp s;
          exit 1
      | Ok outs ->
          let differ =
            List.length (List.filter not (List.map2 same outs oracle))
          in
          Fmt.pr
            "qemu: %d invocations, %d outputs, %d differ from the oracle (%.1f \
             s)@."
            (List.length b.Loop_bundle.invocations)
            (List.length outs) differ
            (Unix.gettimeofday () -. t0);
          if differ > 0 then exit 1)

let () =
  let run = ref true
  and allocation = ref Rt.Allocation.Scanned
  and target = ref Ssa_ir.Ssa_target.scalar
  and cpu = ref None
  and census = ref false in
  let args =
    List.filter
      (fun a ->
        match String.split_on_char '=' a with
        | [ "--planned"; "zve32f" ] ->
            target := Ssa_ir.Ssa_target.zve32f128;
            false
        | [ "--vlen"; n ] ->
            cpu :=
              Some
                (Fmt.str
                   "rv32,d=false,zve32f=true,vlen=%s,elen=32,rvv_ta_all_1s=true,rvv_ma_all_1s=true"
                   n);
            false
        | [ "--contractions" ] ->
            census := true;
            false
        | [ "--run"; "0" ] ->
            run := false;
            false
        | [ "--allocation"; "reference" ] ->
            allocation := Rt.Allocation.Reference;
            false
        | [ "--allocation"; "scanned" ] ->
            allocation := Rt.Allocation.Scanned;
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
        Err.return (b, constants, lowered.Pt2_native_graph.graph)
      in
      match Err.payload result with
      | Error _ ->
          prerr_endline "census: could not open or lower the archive";
          exit 1
      | Ok (b, _, _) when !census -> contractions b
      | Ok (b, constants, graph) -> (
          let n = List.length b.Loop_bundle.invocations in
          let pipeline =
            Ssa_backends.Pipeline.Planned
              { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target = !target }
          in
          let t0 = Unix.gettimeofday () in
          match
            M.prepare
              ~route:(Rt.route ?cpu:!cpu ~allocation:!allocation ())
              ~pipeline b
          with
          | Ok m ->
              Fmt.pr "%s: %d of %d invocations compiled (%.1f s)@."
                (Filename.basename path) n n
                (Unix.gettimeofday () -. t0);
              vector_census m;
              if !run then run_model b ~graph m ~constants
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
                   (List.of_seq (Hashtbl.to_seq tally)));
              exit 1))
  | _ ->
      prerr_endline
        "usage: machine_rivet_rv32_census <model.pt2> [--run=0] \
         [--allocation=reference|scanned] [--planned=zve32f] [--vlen=N] \
         [--contractions]";
      exit 2
