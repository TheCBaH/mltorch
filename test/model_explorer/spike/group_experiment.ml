(* Exports a model twice, once untouched and once through
   [Me_group_constants.apply Grouped], so the two can be screenshotted
   through the same pinned Model Explorer element (see
   [test/model_explorer/spike/gen_fixture.ml]'s harness, e.g.
   [web/src/session.html]) and compared by eye. Not wired into any dune
   alias or `make` target: a rendering judgment call, not a check a build
   can make. Caught the `parameters`-fallback defect no unit test alone
   would have -- the layout consequence only shows up in the renderer. *)

module ME = Model_explorer

let read_all path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic;
  s

let write path text =
  let oc = open_out_bin path in
  output_string oc text;
  close_out oc

let to_cli pp r =
  Err.export ~pos:__POS__ r |> Result.map_error (Core.Pretty.to_string pp)

let encode session =
  match Jsont_bytesrw.encode_string Me_session.Session.jsont session with
  | Ok s -> s
  | Error e -> failwith e

let () =
  let model = Sys.argv.(1) in
  let out_explicit = Sys.argv.(2) in
  let out_grouped = Sys.argv.(3) in
  let limits = Me_limits.Limits.untrusted in
  let bytes = read_all model in
  let name = Filename.remove_extension (Filename.basename model) in
  match
    to_cli Me_export.pp_error
      (Me_export.session ~limits
         ~options:
           {
             Me_export.Options.stages = Me_session.Capability.all_stages;
             fold = false;
             verify_symbolic = None;
             name;
             source_bytes = Int64.of_int (String.length bytes);
             source_sha256 = None;
           }
         ~bytes)
  with
  | Error e ->
      prerr_endline e;
      exit 1
  | Ok session ->
      write out_explicit (encode session);
      let grouped =
        {
          session with
          Me_session.Session.graph_collections =
            List.map
              (fun (c : ME.GraphCollection.t) ->
                {
                  c with
                  ME.GraphCollection.graphs =
                    List.map
                      (Me_group_constants.apply Me_group_constants.Grouped)
                      c.ME.GraphCollection.graphs;
                })
              session.Me_session.Session.graph_collections;
        }
      in
      (match
         to_cli Me_session.Session.pp_error
           (Me_session.Session.validate ~limits grouped)
       with
      | Ok () -> ()
      | Error e ->
          prerr_endline e;
          exit 1);
      write out_grouped (encode grouped)
