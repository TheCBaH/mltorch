(* Split out of native_graph.ml, see native_graph_common.ml, and
   native_graph_args.ml. *)

open Cmdliner
open Native_graph_common
open Native_graph_args

let read_bounded limits path =
  let ic = open_in_bin path in
  let size = in_channel_length ic in
  let peek = really_input_string ic (min size 4) in
  seek_in ic 0;
  let ceiling =
    match Me_export.detect ~bytes:peek with
    | Ok Me_export.Model_json -> limits.Me_limits.Limits.max_json_bytes
    | Ok Me_export.Pt2_archive -> limits.Me_limits.Limits.max_pt2_bytes
    (* Matching the payload, not a guard: [detect]'s error row is closed, so
       this arm still fails to compile the day a second failure is added --
       which a [when Err.Error.kind e = `Unrecognised_format] guard would
       not. *)
    | Error e -> (
        match Err.Error.kind e with
        | `Unrecognised_format -> limits.Me_limits.Limits.max_pt2_bytes)
  in
  if Int64.compare (Int64.of_int size) ceiling > 0 then begin
    close_in ic;
    Error
      (Printf.sprintf "%s is %d bytes, over the %Ld-byte ceiling" path size
         ceiling)
  end
  else begin
    let s = really_input_string ic size in
    close_in ic;
    Ok s
  end

let visualize model limits output format fold verify_symbolic constants :
    (unit, string) result =
  (* Size is rejected BEFORE the rest of the file is read, so an oversized
     file never becomes an OCaml string. Only a filesystem caller can do
     that, which is why it is here and not in the library. *)
  let* bytes = read_bounded limits model in
  let size = Int64.of_int (String.length bytes) in
  let collection_label = Filename.remove_extension (Filename.basename model) in
  let* session =
    to_cli Me_export.pp_error
      (Me_export.session ~limits
         ~options:
           {
             (* Every stage: the CLI exports one complete, standalone
                document, unlike the worker's on-demand requests, so
                "everything the pipeline reaches" is the only sensible
                default and there is no flag to narrow it. *)
             Me_export.Options.stages = Me_session.Capability.all_stages;
             fold;
             verify_symbolic;
             name = collection_label;
             source_bytes = size;
             (* [None]: no expected digest was supplied, and there is nothing
                to verify a locally chosen file against. *)
             source_sha256 = None;
           }
         ~bytes)
  in
  (* Presentation-only, applied after the export the browser bridge's own
     [groupConstants] also starts from -- see [Me_group_constants]. [Explicit]
     is the default and the identity, left untouched and unvalidated a second
     time so every existing golden and every caller that never asks for this
     stays byte-identical to before this flag existed. [Grouped] is
     re-validated for the same reason the bridge does: the transform can only
     ever reuse or shorten an already-valid namespace, but a cheap check is
     what actually distinguishes "safe to show" from "the transform did not
     raise". *)
  let* session =
    match constants with
    | Me_group_constants.Explicit -> Ok session
    | Me_group_constants.Grouped ->
        let grouped =
          {
            session with
            Me_session.Session.graph_collections =
              List.map
                (fun (c : Model_explorer.GraphCollection.t) ->
                  {
                    c with
                    Model_explorer.GraphCollection.graphs =
                      List.map
                        (Me_group_constants.apply Me_group_constants.Grouped)
                        c.Model_explorer.GraphCollection.graphs;
                  })
                session.Me_session.Session.graph_collections;
          }
        in
        let* () =
          to_cli Me_session.Session.pp_error
            (Me_session.Session.validate ~limits grouped)
        in
        Ok grouped
  in
  let* text =
    match format with
    | `Session ->
        to_cli Me_export.pp_error
          (Me_export.encode_bounded
             ~max_bytes:limits.Me_limits.Limits.max_session_bytes
             Me_session.Session.jsont session)
    | `Collections ->
        prerr_endline
          "warning: --format collections is lossy; comparisons, capabilities \
           and the flow are discarded";
        Result.map_error
          (fun e -> "encode: " ^ e)
          (Jsont_bytesrw.encode_string
             (Jsont.list Model_explorer.GraphCollection.jsont)
             session.Me_session.Session.graph_collections)
  in
  (match output with
  | None -> print_string text
  | Some path ->
      let oc = open_out_bin path in
      output_string oc text;
      close_out oc);
  Ok ()

(* The on-demand half, as its own command rather than a flag on [visualize]:
   they produce different documents with different lifetimes -- a session is a
   whole model, a delta is one value's expression merged into a session someone
   already has -- and one command emitting either would have to be told which. *)
let detail model limits output parent value : (unit, string) result =
  (* Same bounded read as [visualize]: rejected on size before the rest of
     the file is read, not after. *)
  let* bytes = read_bounded limits model in
  let size = Int64.of_int (String.length bytes) in
  let* key =
    to_cli Me_request.Detail_key.pp_invalid
      (Err.map_error
         (fun (`Invalid_detail_key e) -> e)
         (Me_request.Detail_key.create ~limits ~parent_graph:parent
            ~value:(Graph_ir.Tensor_id.of_int value)))
  in
  let* delta =
    to_cli Me_export.pp_error
      (Me_export.detail ~limits
         ~options:
           {
             (* Unread by [detail], which never reaches [lowered_shape] -- see
                [Options.stages]'s doc. *)
             Me_export.Options.stages = Me_session.Capability.all_stages;
             fold = false;
             verify_symbolic = None;
             name = Filename.remove_extension (Filename.basename model);
             source_bytes = size;
             source_sha256 = None;
           }
         ~key ~bytes)
  in
  let* text =
    to_cli Me_export.pp_error
      (Me_export.encode_bounded
         ~max_bytes:limits.Me_limits.Limits.max_detail_bytes
         Me_detail.Delta.jsont delta)
  in
  (match output with
  | None -> print_string text
  | Some path ->
      let oc = open_out_bin path in
      output_string oc text;
      close_out oc);
  Ok ()

let detail_cmd =
  let doc =
    "Export one kernel value's expression as a Model Explorer detail delta."
  in
  Cmd.v (Cmd.info "detail" ~doc)
    Term.(
      const detail $ model_arg $ limits_arg $ output_arg $ parent_arg
      $ value_arg)

let visualize_cmd =
  let doc =
    "Export a Model Explorer session for a PT2 archive or an exported \
     model.json."
  in
  Cmd.v
    (Cmd.info "visualize" ~doc)
    Term.(
      const visualize $ model_arg $ limits_arg $ output_arg $ format_arg
      $ fold_arg $ verify_symbolic_arg $ constants_arg)
