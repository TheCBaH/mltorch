(* Where a model's loop iterations go under the strict vectorizer: every
   invocation program of the model's bundle is vectorized for the chosen target,
   and the leaf loops are tallied by outcome, by loop count and by dynamic
   iterations (a loop's trip count times the constant trip counts of the loops
   around it; a non-constant bound counts as one, so the figures are lower
   bounds).

   argv: <model.pt2> [--target=wasm128|neon128] [--f32]

   [--f32] plans against the target's binary32 description ([Loop_target.f32]):
   sixteen lanes, binary32 prices. *)

open Loop_ir

let target_of = function
  | "neon128" -> Loop_target.neon128
  | _ -> Loop_target.wasm128

let () =
  let target = ref Loop_target.wasm128 in
  let show = ref None in
  let f32 = Array.exists (String.equal "--f32") Sys.argv in
  let args =
    List.filter
      (fun a ->
        match String.split_on_char '=' a with
        | [ "--target"; t ] ->
            target := target_of t;
            false
        | [ "--show"; r ] ->
            show := Some r;
            false
        | _ -> a <> "--f32")
      (List.tl (Array.to_list Sys.argv))
  in
  if f32 then target := Loop_target.f32 !target;
  match args with
  | [ path ] -> (
      let result =
        let open Err.Syntax in
        let* archive =
          Pt2_archive.open_pt2 path
          |> Err.map_error (fun e ->
              (e
                :> [ Pt2_archive.error
                   | Native_interp.error
                   | Loop_bundle.error ]))
        in
        let* lowered =
          Native_interp.lower_archive archive
          |> Err.map_error (fun e ->
              (e
                :> [ Pt2_archive.error
                   | Native_interp.error
                   | Loop_bundle.error ]))
        in
        Loop_bundle.build ~config:Loop_bundle_wasm.default_config
          lowered.Pt2_native_graph.graph
        |> Err.map_error (fun e ->
            (e
              :> [ Pt2_archive.error | Native_interp.error | Loop_bundle.error ]))
      in
      match Err.payload result with
      | Error _ ->
          prerr_endline "census: could not open or lower the archive";
          exit 1
      | Ok b ->
          let all =
            List.concat_map
              (fun (inv : Loop_bundle.invocation) ->
                snd
                  (Loop_vectorize.program ~target:!target
                     inv.Loop_bundle.program))
              b.Loop_bundle.invocations
          in
          (match !show with
          | None -> ()
          | Some reason -> (
              (* The invocation whose loop of this outcome covers the most
                 iterations. *)
              let best = ref None in
              List.iter
                (fun (inv : Loop_bundle.invocation) ->
                  let _, r =
                    Loop_vectorize.program ~target:!target
                      inv.Loop_bundle.program
                  in
                  List.iter
                    (fun (d : Loop_vectorize.Decision.t) ->
                      let name =
                        match d.Loop_vectorize.Decision.outcome with
                        | Loop_vectorize.Decision.Kept_scalar r ->
                            Loop_vectorize.Reason.name r
                        | Loop_vectorize.Decision.Vectorized -> "vectorized"
                      in
                      let w =
                        Int64.mul d.Loop_vectorize.Decision.work
                          d.Loop_vectorize.Decision.executions
                      in
                      if name = reason then
                        match !best with
                        | Some (bw, _) when bw >= w -> ()
                        | _ -> best := Some (w, inv))
                    r)
                b.Loop_bundle.invocations;
              match !best with
              | None -> ()
              | Some (w, inv) ->
                  Fmt.pr "heaviest (%Ld iterations):@.%a@." w Loop_pp.program
                    inv.Loop_bundle.program));
          (* Sums: how many reduction loops the optimized programs hold, how many
             the planner can recover, and the iterations they cover. *)
          let candidates = ref 0 and recovered = ref 0 in
          let rec loops (s : Loop_stmt.t) =
            match s with
            | Loop_stmt.For
                { body = Loop_stmt.Mark Loop_mark.Reduction :: _; _ } -> (
                incr candidates;
                match s with
                | Loop_stmt.For { body; _ } -> List.iter loops body
                | _ -> ())
            | Loop_stmt.For { body; _ } -> List.iter loops body
            | Loop_stmt.If (_, a, b) ->
                List.iter loops a;
                List.iter loops b
            | _ -> ()
          in
          List.iter
            (fun (inv : Loop_bundle.invocation) ->
              let p = inv.Loop_bundle.program in
              List.iter loops p.Loop_program.body;
              recovered := !recovered + Loop_sum.count (Loop_sum.recover p))
            b.Loop_bundle.invocations;
          Printf.printf
            "sums: %d reduction loops, %d recovered as structured sums\n"
            !candidates !recovered;
          (* Shape of the recovered sums: statements in the body before the term,
             and constant trip counts. *)
          let shapes = Hashtbl.create 8 in
          let shown = ref false in
          let rec shape_stmt (s : Loop_stmt.t) =
            match s with
            | Loop_stmt.Reduce_sum { lo; hi; body; _ } ->
                if
                  Sys.getenv_opt "CENSUS_SHOW_BODY" <> None
                  && List.length body = 6
                  && not !shown
                then (
                  shown := true;
                  Fmt.pr "example body:@.%a@." Loop_pp.stmts body);
                let trips =
                  match (lo, hi) with
                  | Loop_index.Const a, Loop_index.Const b ->
                      string_of_int (b - a)
                  | _ -> "?"
                in
                let key =
                  Printf.sprintf "body=%d trips=%s" (List.length body) trips
                in
                Hashtbl.replace shapes key
                  (1 + Option.value ~default:0 (Hashtbl.find_opt shapes key));
                List.iter shape_stmt body
            | Loop_stmt.For { body; _ } -> List.iter shape_stmt body
            | Loop_stmt.If (_, a, b) ->
                List.iter shape_stmt a;
                List.iter shape_stmt b
            | _ -> ()
          in
          List.iter
            (fun (inv : Loop_bundle.invocation) ->
              List.iter shape_stmt
                (Loop_sum.recover inv.Loop_bundle.program).Loop_program.body)
            b.Loop_bundle.invocations;
          Hashtbl.fold (fun k v acc -> (k, v) :: acc) shapes []
          |> List.sort compare
          |> List.iter (fun (k, v) -> Printf.printf "  sum %-24s x%d\n" k v);
          let tally = Loop_vectorize.tally all in
          let total_iter =
            List.fold_left (fun acc (_, (_, w)) -> Int64.add acc w) 0L tally
          in
          Printf.printf
            "target %s, %d invocations, %d leaf loops, %Ld loop iterations\n"
            !target.Loop_target.name
            (List.length b.Loop_bundle.invocations)
            (List.length all) total_iter;
          List.iter
            (fun (name, (n, w)) ->
              Printf.printf "  %-34s %6d loops %14Ld iterations %5.1f%%\n" name
                n w
                (100. *. Int64.to_float w /. Int64.to_float (max 1L total_iter)))
            (List.sort (fun (_, (_, a)) (_, (_, b)) -> compare b a) tally))
  | _ ->
      prerr_endline
        "usage: loop_vector_census <model.pt2> [--target=wasm128|neon128]";
      exit 2
