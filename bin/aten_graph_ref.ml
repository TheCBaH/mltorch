(* ATen reference producer for the pure native graph evaluator. *)

let run model input reference =
  let open Err.Syntax in
  let* archive =
    Pt2_archive.open_pt2 model |> Err.map_error (fun e -> `Archive e)
  in
  let* input =
    Pt2_archive.load_first_pt_tensor input
    |> Err.map_error (fun e -> `Archive e)
  in
  let* result =
    Interp.run archive input |> Err.map_error (fun e -> `Interp e)
  in
  let* output =
    Tensor_bridge.of_aten result |> Err.map_error (fun e -> `Tensor_bridge e)
  in
  let+ json =
    Graph_json.encode_tensor ~format:Jsont.Indent output
    |> Err.map_error (fun e -> `Encode e)
  in
  Out_channel.with_open_bin reference (fun oc ->
      Out_channel.output_string oc json)

let pp_error ppf = function
  | `Archive e -> Pt2_archive.pp_error ppf e
  | `Interp e -> Interp.pp_error ppf e
  | `Tensor_bridge e -> Tensor_bridge.pp_error ppf e
  | `Encode e -> Graph_json.pp_error ppf e

let () =
  match Sys.argv with
  | [| _; model; input; output |] -> (
      match run model input output with
      | Ok () -> ()
      | Error e ->
          Format.eprintf "aten_graph_ref: %a@." (Err.Error.pp pp_error) e;
          exit 1)
  | _ ->
      prerr_endline "usage: aten_graph_ref MODEL.pt2 INPUTS.pt OUTPUT.json";
      exit 1
