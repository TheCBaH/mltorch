Parse the two selected functional model fixtures and print schema version, node count, and
op-type histogram sorted by frequency (high to low).

  $ cat > parse_models.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > #use "model_test_utils.ml";;
  > let models = [
  >   "mobilenetv2_050", "mobilenetv2_050_model.json";
  >   "test_convnext2", "test_convnext2_model.json";
  > ]
  > let () =
  >   List.iter (fun (name, file) ->
  >     let json = In_channel.with_open_bin file In_channel.input_all in
  >     Format.printf "%s: %a@." name
  >       (Format.pp_print_result ~ok:pp_model ~error:Format.pp_print_string)
  >       (Jsont_bytesrw.decode_string ExportedProgram.jsont json)
  >   ) models
  > EOF
  $ ocaml parse_models.ml 2>/dev/null
  mobilenetv2_050: schema=8.20 nodes=152
    torch.ops.aten._native_batch_norm_legit_no_training.default: 52
    torch.ops.aten.conv2d.default: 52
    torch.ops.aten.hardtanh.default: 35
    torch.ops.aten.add.Tensor: 10
    torch.ops.aten.adaptive_avg_pool2d.default: 1
    torch.ops.aten.view.default: 1
    torch.ops.aten.linear.default: 1
  test_convnext2: schema=8.20 nodes=71
    torch.ops.aten.permute.default: 18
    torch.ops.aten.linear.default: 9
    torch.ops.aten.clone.default: 9
    torch.ops.aten.layer_norm.default: 9
    torch.ops.aten.conv2d.default: 8
    torch.ops.aten.view.default: 5
    torch.ops.aten.gelu.default: 4
    torch.ops.aten.mul.Tensor: 4
    torch.ops.aten.add.Tensor: 4
    torch.ops.aten.adaptive_avg_pool2d.default: 1
