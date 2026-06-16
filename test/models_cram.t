Parse every exported model and print schema version, node count, and
op-type histogram sorted by frequency (high to low).

  $ cat > parse_models.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #use "schema_pytorch.ml";;
  > #use "model_test_utils.ml";;
  > let models = [
  >   "efficientnet_b0",  "efficientnet_b0_model.json";
  >   "efficientnet_b1",  "efficientnet_b1_model.json";
  >   "efficientnet_b2",  "efficientnet_b2_model.json";
  >   "efficientnet_b3",  "efficientnet_b3_model.json";
  >   "efficientnet_b4",  "efficientnet_b4_model.json";
  >   "efficientnet_b5",  "efficientnet_b5_model.json";
  >   "efficientnet_b6",  "efficientnet_b6_model.json";
  >   "efficientnet_b7",  "efficientnet_b7_model.json";
  >   "efficientnet_v2_l","efficientnet_v2_l_model.json";
  >   "efficientnet_v2_m","efficientnet_v2_m_model.json";
  >   "efficientnet_v2_s","efficientnet_v2_s_model.json";
  >   "mobilenet_v2",     "mobilenet_v2_model.json";
  >   "mobilenet_v3_large","mobilenet_v3_large_model.json";
  >   "mobilenet_v3_small","mobilenet_v3_small_model.json";
  >   "resnet18",         "resnet18_model.json";
  >   "resnet34",         "resnet34_model.json";
  >   "resnet50",         "resnet50_model.json";
  >   "resnet101",        "resnet101_model.json";
  >   "resnet152",        "resnet152_model.json";
  > ]
  > let () =
  >   List.iter (fun (name, file) ->
  >     let json = In_channel.with_open_bin file In_channel.input_all in
  >     Format.printf "%s: %a@." name
  >       (Format.pp_print_result ~ok:pp_model ~error:Format.pp_print_string)
  >       (Jsont_bytesrw.decode_string ExportedProgram.jsont json)
  >   ) models
  > EOF
  $ ocaml schema_runtime.cma parse_models.ml 2>/dev/null
  efficientnet_b0: schema=8.20 nodes=290
    torch.ops.aten.convolution.default: 81
    torch.ops.aten.sigmoid.default: 65
    torch.ops.aten.mul.Tensor: 65
    torch.ops.aten._native_batch_norm_legit_no_training.default: 49
    torch.ops.aten.mean.dim: 17
    torch.ops.aten.add.Tensor: 9
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  efficientnet_b1: schema=8.20 nodes=412
    torch.ops.aten.convolution.default: 115
    torch.ops.aten.sigmoid.default: 92
    torch.ops.aten.mul.Tensor: 92
    torch.ops.aten._native_batch_norm_legit_no_training.default: 69
    torch.ops.aten.mean.dim: 24
    torch.ops.aten.add.Tensor: 16
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  efficientnet_b2: schema=8.20 nodes=412
    torch.ops.aten.convolution.default: 115
    torch.ops.aten.sigmoid.default: 92
    torch.ops.aten.mul.Tensor: 92
    torch.ops.aten._native_batch_norm_legit_no_training.default: 69
    torch.ops.aten.mean.dim: 24
    torch.ops.aten.add.Tensor: 16
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  efficientnet_b3: schema=8.20 nodes=466
    torch.ops.aten.convolution.default: 130
    torch.ops.aten.sigmoid.default: 104
    torch.ops.aten.mul.Tensor: 104
    torch.ops.aten._native_batch_norm_legit_no_training.default: 78
    torch.ops.aten.mean.dim: 27
    torch.ops.aten.add.Tensor: 19
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  efficientnet_b4: schema=8.20 nodes=574
    torch.ops.aten.convolution.default: 160
    torch.ops.aten.sigmoid.default: 128
    torch.ops.aten.mul.Tensor: 128
    torch.ops.aten._native_batch_norm_legit_no_training.default: 96
    torch.ops.aten.mean.dim: 33
    torch.ops.aten.add.Tensor: 25
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  efficientnet_b5: schema=8.20 nodes=696
    torch.ops.aten.convolution.default: 194
    torch.ops.aten.sigmoid.default: 155
    torch.ops.aten.mul.Tensor: 155
    torch.ops.aten._native_batch_norm_legit_no_training.default: 116
    torch.ops.aten.mean.dim: 40
    torch.ops.aten.add.Tensor: 32
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  efficientnet_b6: schema=8.20 nodes=804
    torch.ops.aten.convolution.default: 224
    torch.ops.aten.sigmoid.default: 179
    torch.ops.aten.mul.Tensor: 179
    torch.ops.aten._native_batch_norm_legit_no_training.default: 134
    torch.ops.aten.mean.dim: 46
    torch.ops.aten.add.Tensor: 38
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  efficientnet_b7: schema=8.20 nodes=980
    torch.ops.aten.convolution.default: 273
    torch.ops.aten.sigmoid.default: 218
    torch.ops.aten.mul.Tensor: 218
    torch.ops.aten._native_batch_norm_legit_no_training.default: 163
    torch.ops.aten.mean.dim: 56
    torch.ops.aten.add.Tensor: 48
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  efficientnet_v2_l: schema=8.20 nodes=1223
    torch.ops.aten.convolution.default: 339
    torch.ops.aten.sigmoid.default: 264
    torch.ops.aten.mul.Tensor: 264
    torch.ops.aten._native_batch_norm_legit_no_training.default: 217
    torch.ops.aten.add.Tensor: 73
    torch.ops.aten.mean.dim: 62
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  efficientnet_v2_m: schema=8.20 nodes=884
    torch.ops.aten.convolution.default: 245
    torch.ops.aten.sigmoid.default: 191
    torch.ops.aten.mul.Tensor: 191
    torch.ops.aten._native_batch_norm_legit_no_training.default: 157
    torch.ops.aten.add.Tensor: 51
    torch.ops.aten.mean.dim: 45
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  efficientnet_v2_s: schema=8.20 nodes=614
    torch.ops.aten.convolution.default: 170
    torch.ops.aten.sigmoid.default: 132
    torch.ops.aten.mul.Tensor: 132
    torch.ops.aten._native_batch_norm_legit_no_training.default: 110
    torch.ops.aten.add.Tensor: 35
    torch.ops.aten.mean.dim: 31
    torch.ops.aten.view.default: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  mobilenet_v2: schema=8.20 nodes=154
    torch.ops.aten.convolution.default: 52
    torch.ops.aten._native_batch_norm_legit_no_training.default: 52
    torch.ops.aten.hardtanh.default: 35
    torch.ops.aten.add.Tensor: 10
    torch.ops.aten.view.default: 1
    torch.ops.aten.mean.dim: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.clone.default: 1
    torch.ops.aten.addmm.default: 1
  mobilenet_v3_large: schema=8.20 nodes=297
    torch.ops.aten.convolution.default: 62
    torch.ops.aten.clamp.default: 58
    torch.ops.aten._native_batch_norm_legit_no_training.default: 46
    torch.ops.aten.add.Tensor: 39
    torch.ops.aten.div.Tensor: 29
    torch.ops.aten.mul.Tensor: 29
    torch.ops.aten.relu.default: 19
    torch.ops.aten.mean.dim: 9
    torch.ops.aten.permute.default: 2
    torch.ops.aten.addmm.default: 2
    torch.ops.aten.view.default: 1
    torch.ops.aten.clone.default: 1
  mobilenet_v3_small: schema=8.20 nodes=262
    torch.ops.aten.clamp.default: 56
    torch.ops.aten.convolution.default: 52
    torch.ops.aten._native_batch_norm_legit_no_training.default: 34
    torch.ops.aten.add.Tensor: 34
    torch.ops.aten.div.Tensor: 28
    torch.ops.aten.mul.Tensor: 28
    torch.ops.aten.relu.default: 14
    torch.ops.aten.mean.dim: 10
    torch.ops.aten.permute.default: 2
    torch.ops.aten.addmm.default: 2
    torch.ops.aten.view.default: 1
    torch.ops.aten.clone.default: 1
  resnet18: schema=8.20 nodes=70
    torch.ops.aten.convolution.default: 20
    torch.ops.aten._native_batch_norm_legit_no_training.default: 20
    torch.ops.aten.relu.default: 17
    torch.ops.aten.add.Tensor: 8
    torch.ops.aten.view.default: 1
    torch.ops.aten.mean.dim: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.addmm.default: 1
    torch.ops.aten.max_pool2d_with_indices.default: 1
  resnet34: schema=8.20 nodes=126
    torch.ops.aten.convolution.default: 36
    torch.ops.aten._native_batch_norm_legit_no_training.default: 36
    torch.ops.aten.relu.default: 33
    torch.ops.aten.add.Tensor: 16
    torch.ops.aten.view.default: 1
    torch.ops.aten.mean.dim: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.addmm.default: 1
    torch.ops.aten.max_pool2d_with_indices.default: 1
  resnet50: schema=8.20 nodes=176
    torch.ops.aten.convolution.default: 53
    torch.ops.aten._native_batch_norm_legit_no_training.default: 53
    torch.ops.aten.relu.default: 49
    torch.ops.aten.add.Tensor: 16
    torch.ops.aten.view.default: 1
    torch.ops.aten.mean.dim: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.addmm.default: 1
    torch.ops.aten.max_pool2d_with_indices.default: 1
  resnet101: schema=8.20 nodes=346
    torch.ops.aten.convolution.default: 104
    torch.ops.aten._native_batch_norm_legit_no_training.default: 104
    torch.ops.aten.relu.default: 100
    torch.ops.aten.add.Tensor: 33
    torch.ops.aten.view.default: 1
    torch.ops.aten.mean.dim: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.addmm.default: 1
    torch.ops.aten.max_pool2d_with_indices.default: 1
  resnet152: schema=8.20 nodes=516
    torch.ops.aten.convolution.default: 155
    torch.ops.aten._native_batch_norm_legit_no_training.default: 155
    torch.ops.aten.relu.default: 151
    torch.ops.aten.add.Tensor: 50
    torch.ops.aten.view.default: 1
    torch.ops.aten.mean.dim: 1
    torch.ops.aten.permute.default: 1
    torch.ops.aten.addmm.default: 1
    torch.ops.aten.max_pool2d_with_indices.default: 1
