(* The binary64 census of a whole model's C as the embedded CompCert makes it.
   Every invocation of the model's Loop IR bundle is emitted as C in the
   Compcert_scalar dialect under the chosen numerics, compiled in process, and
   the assembly (plus the [__compcert_i64_*] helpers CompCert links for it)
   counted by role: what still runs on a double-precision register.

   argv: <model.pt2> [--numerics=NAME]    (default reference_f64) *)

open Loop_ir
module E = Loop_c_embed.Make (Loop_c_embed_compcert.Compcert_loader)
module T = Loop_c_embed_compcert.Compcert_embed_target

let census name text =
  match Err.payload (Asm_census.of_text Asm_census.Isa.Aarch64 text) with
  | Ok c ->
      Fmt.pr "%s: %a%a@." name Asm_census.pp c
        Fmt.(list ~sep:nop (fun ppf (m, n) -> pf ppf "; %s x%d" m n))
        (Asm_census.mnemonics c)
  | Error e ->
      Fmt.pr "%s: %a@." name Asm_census.pp_error e;
      exit 1

let () =
  let numerics = ref Loop_numerics.Reference_f64 in
  let args =
    List.filter
      (fun a ->
        match String.split_on_char '=' a with
        | [ "--numerics"; n ] -> (
            match Loop_numerics.of_name n with
            | Some p ->
                numerics := p;
                false
            | None ->
                prerr_endline ("asm_census_pt2: unknown numerics " ^ n);
                exit 2)
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
        Loop_bundle.build ~config:Loop_bundle_c.default_config
          lowered.Pt2_native_graph.graph
        |> Err.map_error lift
      in
      match Err.payload result with
      | Error _ ->
          prerr_endline "asm_census_pt2: could not open or lower the archive";
          exit 1
      | Ok bundle -> (
          match Err.payload (E.prepare ~numerics:!numerics bundle) with
          | Error e ->
              Fmt.epr "asm_census_pt2: %a@." Loop_c_embed.pp_error e;
              exit 1
          | Ok p -> (
              let source = E.source p in
              E.close p;
              match T.compile_to_asm ~name:"model.c" source with
              | Error e ->
                  Fmt.epr "asm_census_pt2: %a@." Compcert_embed.pp_error e;
                  exit 1
              | Ok asm -> (
                  census (Filename.basename path ^ " model") asm;
                  match T.runtime_units () with
                  | Ok units ->
                      List.iter
                        (fun (n, text) -> census ("  helper " ^ n) text)
                        units
                  | Error e ->
                      Fmt.pr "  helpers: %a@." Compcert_embed.pp_error e))))
  | _ ->
      prerr_endline "usage: asm_census_pt2 <model.pt2> [--numerics=NAME]";
      exit 2
