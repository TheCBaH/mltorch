(* ATen reference producer for the pure native graph evaluator. *)

let run model input reference =
  let open Err.Syntax in
  let* archive =
    Pt2_archive.open_pt2 model |> Err.map_error (fun e -> `Archive e)
  in
  let* input =
    Pt2_archive.load_pt input |> Err.map_error (fun e -> `Archive e)
  in
  let* result =
    Interp.run archive input |> Err.map_error (fun e -> `Interp e)
  in
  match Tensor_bridge.of_aten result with
  | Error message -> Err.fail (`Tensor_bridge message)
  | Ok output -> (
      match Graph_json.encode_tensor ~format:Jsont.Indent output with
      | Error message -> Err.fail (`Encode message)
      | Ok json ->
          Out_channel.with_open_bin reference (fun oc ->
              Out_channel.output_string oc json);
          Err.return ())

let pp_error ppf = function
  | `Archive e -> Pt2_archive.pp_error ppf e
  | `Interp e -> Interp.pp_error ppf e
  | `Tensor_bridge message | `Encode message ->
      Format.pp_print_string ppf message

let () =
  match Sys.argv with
  | [| _; model; input; output |] -> (
      match run model input output with
      | Ok () -> ()
      | Error e ->
          Format.eprintf "aten_graph_ref: %a@." (Err.Error.pp pp_error) e;
          exit 1)
  | _ ->
      prerr_endline "usage: aten_graph_ref MODEL.pt2 INPUT.pt OUTPUT.json";
      exit 1
