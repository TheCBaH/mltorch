(* The lowering inspector's command line: what a backend family makes of one
   node of the canonical graph. See native_graph_common.ml for [to_cli] and
   [( let* )], and native_graph_visualize.ml for [read_bounded]. *)

open Cmdliner
open Native_graph_common
open Native_graph_args
module Key = Lowering_key
module Artifact = Lowering_view.Artifact

(* A node is spelled as the canonical view prints it ([n12]); the bare number
   is accepted too. Bounded before it becomes an [int] so a 32-bit target
   cannot wrap it into another node. *)
let node_conv =
  let parse s =
    let digits =
      if String.length s > 1 && s.[0] = 'n' then
        String.sub s 1 (String.length s - 1)
      else s
    in
    match int_of_string_opt digits with
    | Some n
      when n >= 0 && String.for_all (fun c -> c >= '0' && c <= '9') digits ->
        Ok (Graph_ir.Node_id.of_int n)
    | _ -> Error (`Msg (Printf.sprintf "%S is not a node id (expected nN)" s))
  in
  Arg.conv (parse, Graph_ir.Node_id.pp)

(* The spelling's own parser is the converter, so an unknown family or segment
   is a usage error naming the part that was refused. *)
let variant_conv =
  let parse s =
    Result.map_error
      (fun e -> `Msg (Core.Pretty.to_string Key.Parse_error.pp e))
      (Key.of_name s)
  in
  Arg.conv (parse, Key.pp)

let node_arg =
  let doc = "The node to lower, spelled as the canonical view prints it." in
  Arg.(required & opt (some node_conv) None & info [ "node" ] ~doc ~docv:"ID")

let artifact_arg =
  let doc =
    "What to show: $(i,family)[:$(i,variant)], e.g. $(b,loop), $(b,loop:raw), \
     $(b,js), $(b,js:ssa), $(b,c), $(b,wat), $(b,ssa)."
  in
  Arg.(
    value & opt (some variant_conv) None & info [ "artifact" ] ~doc ~docv:"SPEC")

let compare_arg =
  let doc = "A second artifact of the same node; print a unified diff." in
  Arg.(
    value & opt (some variant_conv) None & info [ "compare" ] ~doc ~docv:"SPEC")

let list_arg =
  let doc =
    "Print the node's invocation and whether each family's default emits or \
     refuses."
  in
  Arg.(value & flag & info [ "list" ] ~doc)

(* The canonical bundle is built granting nothing ([Loop_bundle.build]'s
   default), and a node that needed granting fails the build, so no node of
   it is granted. *)
let nobody (_ : Graph_ir.Node_id.t) = false

let refused reason =
  Fmt.epr "refused: %a@." Artifact.Refusal.pp reason;
  exit 1

let text_of variant bundle node =
  let a = Lowering_view.of_node ~granted:nobody bundle node variant in
  match a.Artifact.body with
  | Artifact.Text { text; _ } -> Ok (a, text)
  | Artifact.Refused r -> Error r

let list_node (inv : Loop_ir.Loop_bundle.invocation) =
  Fmt.pr "node %a: %d outputs, %d edges, %d synthetics@." Graph_ir.Node_id.pp
    inv.node (List.length inv.outputs) (List.length inv.edges)
    (List.length inv.synthetics);
  List.iter
    (fun family ->
      let v = Key.default family in
      let a = Lowering_view.of_invocation ~granted:nobody inv v in
      match a.Artifact.body with
      | Artifact.Text { text; _ } ->
          Fmt.pr "%-10s emits %d lines@." (Key.name v)
            (List.length (Lowering_diff.lines text))
      | Artifact.Refused r ->
          Fmt.pr "%-10s refused: %a@." (Key.name v) Artifact.Refusal.pp r)
    Key.families

let lower model limits node artifact compare list : (unit, string) result =
  let* bytes = Native_graph_visualize.read_bounded limits model in
  let* bundle =
    to_cli Me_export.pp_error
      (Me_export.canonical_bundle ~limits
         ~options:
           {
             Me_export.Options.stages = Me_session.Capability.all_stages;
             fold = false;
             verify_symbolic = None;
             name = Filename.remove_extension (Filename.basename model);
             source_bytes = Int64.of_int (String.length bytes);
             source_sha256 = None;
           }
         ~bytes)
  in
  let* inv =
    match
      List.find_opt
        (fun (i : Loop_ir.Loop_bundle.invocation) ->
          Graph_ir.Node_id.equal i.node node)
        bundle.Loop_ir.Loop_bundle.invocations
    with
    | Some i -> Ok i
    | None ->
        Error (Fmt.str "no invocation for node %a" Graph_ir.Node_id.pp node)
  in
  if list then Ok (list_node inv)
  else
    match (artifact, compare) with
    | None, _ -> Error "--artifact is required unless --list"
    | Some v, None -> (
        match text_of v bundle node with
        | Ok (a, _) -> Ok (Fmt.pr "%a@." Artifact.pp a)
        | Error r -> refused r)
    | Some a, Some b -> (
        match (text_of a bundle node, text_of b bundle node) with
        | Ok (_, ta), Ok (_, tb) ->
            let d = Lowering_diff.diff ta tb in
            if Lowering_diff.identical d then Fmt.pr "identical@."
            else
              print_string
                (Lowering_diff.unified ~context:3 ~old_label:(Key.name a)
                   ~new_label:(Key.name b) d);
            Ok ()
        | Error r, _ | _, Error r -> refused r)

let lower_cmd =
  let doc = "Show what a backend family makes of one node of the model." in
  Cmd.v (Cmd.info "lower" ~doc)
    Term.(
      const lower $ model_arg $ limits_arg $ node_arg $ artifact_arg
      $ compare_arg $ list_arg)
