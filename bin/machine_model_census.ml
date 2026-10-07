(* Which invocations of a model's bundle the Machine IR adapter admits: every
   invocation's placed kernel lowered through the chosen SSA pipeline and then
   to generic Machine IR ([Machine_model.Mir_model.prepare]), and every refusal
   tallied by its reason. Lowering only: interpreting a whole model is not the
   point of this report.

   [--run=K] then runs the first K invocations — an explicit invocation
   sequence — on the archive's own weights and a deterministic synthetic input,
   in the generic interpreter, and compares every tensor they write with the
   per-node reference, bit for bit.

   [--measure=aarch64|x86_64] times every back-end stage of every invocation
   separately — SSA lowering, Machine IR lowering, selection, reference
   allocation, frame realization, publication (re-verification and checking
   included) — and the packing of the model's constants into bytes. Printing,
   assembly, loading and native calls are downstream of the artifact and are
   reported as such, never estimated.

   argv: <model.pt2> [--ssa=exact|representation] [--run=K]
   [--measure=aarch64|x86_64] *)

open Loop_ir
module M = Machine_model.Mir_model

let bits (Tensor.Tensor t as packed) =
  let acc = ref [] in
  Vec6.iter t.Tensor.shape (fun c ->
      acc := Int64.bits_of_float (Tensor.read packed c) :: !acc);
  List.rev !acc

(* Every invocation through every back-end stage, each timed on its own. *)
let stages (b : Loop_bundle.t) ~pipeline ~constants route =
  let now = Unix.gettimeofday in
  let timed f =
    let t0 = now () in
    let r = f () in
    (r, now () -. t0)
  in
  let packed, packing =
    timed (fun () ->
        Graph_ir.Tensor_id.Map.fold
          (fun _ t n ->
            n + String.length (Machine_model.Mir_tensor_bytes.to_string t))
          constants 0)
  in
  let totals = Array.make 6 0. and relocations = ref 0 and refused = ref 0 in
  let add k x = totals.(k) <- totals.(k) +. x in
  List.iter
    (fun (inv : Loop_bundle.invocation) ->
      let p, ssa = timed (fun () -> Ssa_backends.program pipeline inv) in
      add 0 ssa;
      match p with
      | Error _ -> incr refused
      | Ok p -> (
          let planning =
            Machine_ir.Mir_planning.make
              ~subject:(Machine_lower.Mir_lower.subject p)
              ~policy:"reference_f64" ~schedule:"scalar"
              ~precision:Machine_ir.Mir_planning.Precision.F64
              ~lanes:(Machine_ir.Mir_type.Lanes.of_int 1)
              ~fma:Machine_ir.Mir_planning.Fma.Forbidden ~capabilities:[]
          in
          let l, lower =
            timed (fun () ->
                Err.payload
                  (Machine_lower.Mir_lower.program ~planning:(Some planning) p))
          in
          add 1 lower;
          match l with
          | Error _ -> incr refused
          | Ok l -> (
              match
                Machine_model.Mir_model_route.measure route ~now
                  ~sites:(M.sites inv) l.Machine_lower.Mir_lower.program
              with
              | Error _ -> incr refused
              | Ok t ->
                  let open Machine_model.Mir_model_route.Timing in
                  add 2 t.select;
                  add 3 t.allocate;
                  add 4 t.realize;
                  add 5 t.publish;
                  relocations := !relocations + t.relocations)))
    b.Loop_bundle.invocations;
  Fmt.pr
    "%s: lowering ssa %.2f s, machine %.2f s; selection %.2f s; allocation \
     %.2f s; realization %.2f s; publication %.2f s; %d relocations; %d \
     refused@."
    (M.Route.name route) totals.(0) totals.(1) totals.(2) totals.(3) totals.(4)
    totals.(5) !relocations !refused;
  Fmt.pr "packing: %d constant bytes in %.2f s@." packed packing;
  Fmt.pr
    "printing, assembly, loading, first and warm calls: downstream of the \
     artifact, not measured here@."

(* The first [count] invocations against the reference, tensor by tensor. *)
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
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g
         ~constants:(Graph_ir.Tensor_id.Map.bindings constants)
         ~inputs)
  in
  let lookup m id = Graph_ir.Tensor_id.Map.find_opt id m in
  match M.Context.create m ~constants:(lookup constants) with
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
          Fmt.pr
            "first %d invocations: %d tensors, %d differ from the reference \
             (%.1f s)@."
            count (List.length written) (List.length differing) seconds;
          List.iter
            (fun id -> Fmt.pr "  differs: %a@." Graph_ir.Tensor_id.pp id)
            differing)

let () =
  let pipeline = ref Ssa_backends.Pipeline.Exact in
  let run = ref None in
  let measure = ref None in
  let args =
    List.filter
      (fun a ->
        match String.split_on_char '=' a with
        | [ "--ssa"; "exact" ] -> false
        | [ "--ssa"; "representation" ] ->
            pipeline := Ssa_backends.Pipeline.Representation;
            false
        | [ "--run"; k ] ->
            run := Some (int_of_string k);
            false
        | [ "--measure"; "aarch64" ] ->
            measure := Some (M.Route.Aarch64 M.Stage.Realized);
            false
        | [ "--measure"; "x86_64" ] ->
            measure := Some (M.Route.X86_64 M.Stage.Realized);
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
          match Machine_model.Mir_model.prepare ~pipeline:!pipeline b with
          | Ok m -> (
              Fmt.pr "%s: %d of %d invocations admitted@."
                (Filename.basename path) n n;
              (match !measure with
              | None -> ()
              | Some route -> stages b ~pipeline:!pipeline ~constants route);
              match !run with
              | None -> ()
              | Some count -> prefix b m ~constants ~count)
          | Error refusals ->
              let tally = Hashtbl.create 16 in
              List.iter
                (fun (r : Machine_model.Mir_model.Refusal.t) ->
                  let k =
                    Fmt.str "%a" Machine_model.Mir_model.Reason.pp
                      r.Machine_model.Mir_model.Refusal.reason
                  in
                  Hashtbl.replace tally k
                    (1 + Option.value ~default:0 (Hashtbl.find_opt tally k)))
                refusals;
              Fmt.pr "%s: %d of %d invocations admitted@."
                (Filename.basename path)
                (n - List.length refusals)
                n;
              List.iter
                (fun (k, c) -> Fmt.pr "  %4d  %s@." c k)
                (List.sort
                   (fun (a, x) (b, y) ->
                     match compare y x with 0 -> compare a b | c -> c)
                   (List.of_seq (Hashtbl.to_seq tally)))))
  | _ ->
      prerr_endline "usage: machine_model_census <model.pt2> [--ssa=NAME]";
      exit 2
