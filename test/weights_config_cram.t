Parse model_weights_config.json for all 17 models and print weight count and
the lexicographically first weight's key, path_name, and is_param flag.

  $ for name in \
  >   mobilenetv2_050; do
  >   ./parse_weights_config.exe "$name" "${name}_weights_config.json"
  > done
  mobilenetv2_050: weights=314 first=blocks.0.0.bn1.bias path_name=weight_5 is_param=true
