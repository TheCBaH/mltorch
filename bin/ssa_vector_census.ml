(* What the binary32 vector planners take of a model: every invocation of its
   bundle through each [Ssa_backends.Coverage] column (the SSA vectorizer on
   the narrowed and on the exact binary32 form, and the Loop planner under
   Fp32_ordered, for neon128 and wasm128), tallied by outcome: loops, and
   innermost iterations covered (a loop's trips times the constant trip counts
   around it). Planning only, nothing runs.

   argv: <model.pt2> *)

open Loop_ir

let () =
  match List.tl (Array.to_list Sys.argv) with
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
        Loop_bundle.build lowered.Pt2_native_graph.graph |> Err.map_error lift
      in
      match Err.payload result with
      | Error _ ->
          prerr_endline "ssa_vector_census: could not open or lower the archive";
          exit 1
      | Ok b ->
          Fmt.pr "%s: %d invocations@." (Filename.basename path)
            (List.length b.Loop_bundle.invocations);
          List.iter
            (fun (c, t, refused) ->
              Fmt.pr "  %s (%d refused): %a@."
                (Ssa_backends.Coverage.name c)
                refused Ssa_backends.Coverage.pp_tally t)
            (Ssa_backends.Coverage.bundle Ssa_backends.Coverage.columns b))
  | _ ->
      prerr_endline "usage: ssa_vector_census <model.pt2>";
      exit 2
