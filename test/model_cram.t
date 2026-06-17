Parse the full exported resnet18 model (model.json) and print summary stats.

  $ cat > parse_model.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > #use "model_test_utils.ml";;
  > let () =
  >   let json = In_channel.with_open_bin "model.json" In_channel.input_all in
  >   Format.printf "%a@."
  >     (Format.pp_print_result ~ok:pp_model ~error:Format.pp_print_string)
  >     (Jsont_bytesrw.decode_string ExportedProgram.jsont json)
  > EOF
  $ ocaml parse_model.ml 2>/dev/null
  schema=8.20 nodes=70
    torch.ops.aten.convolution.default: 20
    torch.ops.aten._native_batch_norm_legit_no_training.default: 20
    torch.ops.aten.relu.default: 17
    torch.ops.aten.add.Tensor: 8
    torch.ops.aten.view.default: 1
    torch.ops.aten.mean.dim: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.addmm.default: 1
    torch.ops.aten.max_pool2d_with_indices.default: 1
