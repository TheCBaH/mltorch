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
   separately — SSA lowering, Machine IR lowering, selection, sink scheduling,
   reference and linear-scan allocation, frame realization, publication of the
   reference allocation (re-verification and checking included) — and the
   packing of the model's constants into bytes. Printing, assembly, loading
   and native calls are downstream of the artifact and are reported as such,
   never estimated.

   [--pressure=aarch64|x86_64] reports what the production pipeline (sink
   scheduling, linear scan, frames) leaves under register pressure, summed
   over the invocations: the largest peak of live values per bank, the spill
   stores and reloads in hot loops, the largest frame and the helper calls;
   with [--blocking=feedback] each invocation's SSA output blocking is chosen
   by that pressure first, and the choices are tallied.

   [--traffic=aarch64|x86_64] runs the invocations (the first K with
   [--run=K], else all) on that target's production pipeline in the physical
   interpreter, compares them like [--run], and reports what they executed:
   instructions, register moves, frame stores and reloads and
   rematerializations, in total and by operation; then the executed
   instructions by origin role and the most executed role and mnemonic
   pairs.

   argv: <model.pt2> [--ssa=exact|representation] [--run=K]
   [--measure=aarch64|x86_64] [--pressure=aarch64|x86_64]
   [--blocking=feedback] [--traffic=aarch64|x86_64] *)

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
  let totals = Array.make 8 0. and relocations = ref 0 and refused = ref 0 in
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
                  add 3 t.schedule;
                  add 4 t.allocate;
                  add 5 t.scan;
                  add 6 t.realize;
                  add 7 t.publish;
                  relocations := !relocations + t.relocations)))
    b.Loop_bundle.invocations;
  Fmt.pr
    "%s: lowering ssa %.2f s, machine %.2f s; selection %.2f s; scheduling \
     %.2f s; allocation reference %.2f s, linear scan %.2f s; realization %.2f \
     s; publication %.2f s; %d relocations; %d refused@."
    (M.Route.name route) totals.(0) totals.(1) totals.(2) totals.(3) totals.(4)
    totals.(5) totals.(6) totals.(7) !relocations !refused;
  Fmt.pr "packing: %d constant bytes in %.2f s@." packed packing;
  Fmt.pr
    "printing, assembly, loading, first and warm calls: downstream of the \
     artifact, not measured here@."

(* What the production pipeline leaves under register pressure, over every
   invocation: [reports] holds one per invocation. *)
let pressure_summary route reports ~choices =
  let module P = Machine_alloc.Mir_pressure in
  let ok = List.filter_map Result.to_option reports in
  let merge f l =
    List.fold_left
      (fun acc (b, n) ->
        let m = Option.value ~default:0 (List.assoc_opt b acc) in
        (b, f m n) :: List.remove_assoc b acc)
      [] l
    |> List.sort compare
  in
  let banks fmt = function
    | [] -> Fmt.string fmt "none"
    | l ->
        Fmt.(
          list ~sep:(any ", ") (fun fmt (b, n) ->
              Fmt.pf fmt "%s %d" (Machine_ir.Mir_target.Bank.name b) n))
          fmt l
  in
  Fmt.pr
    "%s pressure: %d invocations (%d unmeasured); largest peak %a; hot stores \
     %a, loads %a, in %d invocations; largest frame %Ld bytes; %d helper \
     calls@."
    (M.Route.name route) (List.length reports)
    (List.length reports - List.length ok)
    banks
    (merge max (List.concat_map (fun (p : P.t) -> p.P.peak) ok))
    banks
    (merge ( + ) (List.concat_map (fun (p : P.t) -> p.P.hot_stores) ok))
    banks
    (merge ( + ) (List.concat_map (fun (p : P.t) -> p.P.hot_loads) ok))
    (List.length (List.filter (fun p -> P.hot_spills p > 0) ok))
    (List.fold_left
       (fun m (p : P.t) -> max m (Option.value ~default:0L p.P.frame))
       0L ok)
    (List.fold_left (fun n (p : P.t) -> n + p.P.helper_calls) 0 ok);
  Fmt.pr "%s hot spills %a@." (M.Route.name route) P.pp_depths
    ( merge ( + ) (List.concat_map (fun (p : P.t) -> p.P.by_depth) ok),
      List.fold_left (fun n (p : P.t) -> n + p.P.innermost) 0 ok );
  match choices with
  | [] -> ()
  | l ->
      Fmt.pr "feedback chose: %a@."
        Fmt.(
          list ~sep:(any ", ") (fun fmt (g, n) -> Fmt.pf fmt "group %d x%d" g n))
        (merge ( + ) (List.map (fun g -> (g, 1)) l))

(* What the invocations run so far executed of their allocation, in total and
   by operation, largest stores and reloads first. *)
let traffic_summary (b : Loop_bundle.t) m route =
  let module Tr = Machine_interp.Mir_phys_interp.Traffic in
  let ran =
    List.filter_map
      (fun (i, t) ->
        match t with
        | Some (t : Tr.t) when Int64.compare t.Tr.instructions 0L > 0 ->
            Some (i, t)
        | _ -> None)
      (M.traffic m)
  in
  let op (inv : Loop_bundle.invocation) =
    match
      List.find_opt
        (fun (n : _ Graph_common.Node.t) ->
          n.Graph_common.Node.id = inv.Loop_bundle.node)
        b.Loop_bundle.graph.Graph_common.Graph.nodes
    with
    | Some n ->
        let s =
          Fmt.str "%a"
            (Graph_ir.pp_op b.Loop_bundle.graph)
            n.Graph_common.Node.op
        in
        List.hd
          (String.split_on_char ' '
             (String.map (function '\n' -> ' ' | c -> c) s))
    | None -> "?"
  in
  let by_op = Hashtbl.create 16 in
  List.iter
    (fun (inv, t) ->
      let k = op inv in
      Hashtbl.replace by_op k
        (Tr.add t (Option.value ~default:Tr.zero (Hashtbl.find_opt by_op k))))
    ran;
  Fmt.pr "%s traffic: %a@." (M.Route.name route) Tr.pp
    (List.fold_left (fun acc (_, t) -> Tr.add acc t) Tr.zero ran);
  Hashtbl.fold (fun k t acc -> (k, t) :: acc) by_op []
  |> List.sort (fun (_, (a : Tr.t)) (_, (b : Tr.t)) ->
      compare
        (Int64.add b.Tr.stores b.Tr.reloads)
        (Int64.add a.Tr.stores a.Tr.reloads))
  |> List.iter (fun (k, t) -> Fmt.pr "  %s: %a@." k Tr.pp t);
  (* what the executed instructions were: by origin role, then the most
     executed role and mnemonic pairs *)
  let total = List.fold_left (fun acc (_, t) -> Tr.add acc t) Tr.zero ran in
  let pct n =
    100. *. Int64.to_float n
    /. Float.max 1. (Int64.to_float total.Tr.instructions)
  in
  let roles = Hashtbl.create 8 in
  Tr.Ops.iter
    (fun k n ->
      let role = List.hd (String.split_on_char ' ' k) in
      Hashtbl.replace roles role
        (Int64.add n (Option.value ~default:0L (Hashtbl.find_opt roles role))))
    total.Tr.ops;
  Fmt.pr "by role:@.";
  Hashtbl.fold (fun k n acc -> (k, n) :: acc) roles []
  |> List.sort (fun (_, a) (_, b) -> Int64.compare b a)
  |> List.iter (fun (k, n) -> Fmt.pr "  %s: %Ld (%.1f%%)@." k n (pct n));
  Fmt.pr "by operation:@.";
  Tr.Ops.bindings total.Tr.ops
  |> List.sort (fun (_, a) (_, b) -> Int64.compare b a)
  |> List.filteri (fun i _ -> i < 25)
  |> List.iter (fun (k, n) -> Fmt.pr "  %s: %Ld (%.1f%%)@." k n (pct n))

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
  let pressure = ref None in
  let feedback = ref false in
  let traffic = ref None in
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
        | [ "--pressure"; "aarch64" ] ->
            pressure := Some (M.Route.Aarch64 M.Stage.Selected);
            false
        | [ "--pressure"; "x86_64" ] ->
            pressure := Some (M.Route.X86_64 M.Stage.Selected);
            false
        | [ "--blocking"; "feedback" ] ->
            feedback := true;
            false
        | [ "--traffic"; "aarch64" ] ->
            traffic := Some (M.Route.Aarch64 M.Stage.Scanned);
            false
        | [ "--traffic"; "x86_64" ] ->
            traffic := Some (M.Route.X86_64 M.Stage.Scanned);
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
          let route =
            match (!traffic, !pressure) with
            | Some r, _ | None, Some r -> r
            | None, None -> M.Route.Generic
          in
          let blocking =
            if !feedback then Machine_model.Mir_blocking.Policy.Feedback
            else Machine_model.Mir_blocking.Policy.Unblocked
          in
          match
            Machine_model.Mir_model.prepare ~route ~blocking ~pipeline:!pipeline
              b
          with
          | Ok m -> (
              Fmt.pr "%s: %d of %d invocations admitted@."
                (Filename.basename path) n n;
              (match !measure with
              | None -> ()
              | Some route -> stages b ~pipeline:!pipeline ~constants route);
              (match !pressure with
              | None -> ()
              | Some route ->
                  let decisions = List.map snd (M.blocking m) in
                  let reports =
                    if !feedback then
                      List.map
                        (fun (d : Machine_model.Mir_blocking.Decision.t) ->
                          (List.find
                             (fun (c : Machine_model.Mir_blocking.Candidate.t)
                                ->
                               c.Machine_model.Mir_blocking.Candidate.group
                               = d.Machine_model.Mir_blocking.Decision.chosen)
                             d.Machine_model.Mir_blocking.Decision.candidates)
                            .Machine_model.Mir_blocking.Candidate.pressure)
                        decisions
                    else
                      List.map2
                        (fun inv g ->
                          Machine_model.Mir_model_route.pressure route
                            ~sites:(M.sites inv) g)
                        b.Loop_bundle.invocations (M.generic m)
                  in
                  pressure_summary route reports
                    ~choices:
                      (List.map
                         (fun (d : Machine_model.Mir_blocking.Decision.t) ->
                           d.Machine_model.Mir_blocking.Decision.chosen)
                         decisions));
              match (!run, !traffic) with
              | None, None -> ()
              | Some count, None -> prefix b m ~constants ~count
              | count, Some route ->
                  prefix b m ~constants ~count:(Option.value ~default:n count);
                  traffic_summary b m route)
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
