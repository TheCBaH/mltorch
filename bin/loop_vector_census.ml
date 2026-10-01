(* Where a model's loop iterations go under the strict vectorizer: every
   invocation program of the model's bundle is vectorized for the chosen target,
   and the leaf loops are tallied by outcome, by loop count and by dynamic
   iterations (a loop's trip count times the constant trip counts of the loops
   around it; a non-constant bound counts as one, so the figures are lower
   bounds).

   argv: <model.pt2> [--target=wasm128|neon128]  *)

open Loop_ir

let target_of = function
  | "neon128" -> Loop_target.neon128
  | _ -> Loop_target.wasm128

let () =
  let target = ref Loop_target.wasm128 in
  let show = ref None in
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
        | _ -> true)
      (List.tl (Array.to_list Sys.argv))
  in
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
          | Some reason ->
              let found = ref false in
              List.iter
                (fun (inv : Loop_bundle.invocation) ->
                  if not !found then
                    let _, r =
                      Loop_vectorize.program ~target:!target
                        inv.Loop_bundle.program
                    in
                    if
                      List.exists
                        (fun (d : Loop_vectorize.Decision.t) ->
                          match d.Loop_vectorize.Decision.outcome with
                          | Loop_vectorize.Decision.Kept_scalar r ->
                              Loop_vectorize.Reason.name r = reason
                          | Loop_vectorize.Decision.Vectorized ->
                              reason = "vectorized")
                        r
                    then (
                      found := true;
                      Fmt.pr "%a@." Loop_pp.program inv.Loop_bundle.program))
                b.Loop_bundle.invocations);
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
