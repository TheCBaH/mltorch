  $ ../bin/aten_ops_gen.exe native_functions.yaml aten_op_config.ml > aten_op_config.ml
  $ sed -n '/^let all /,/^let find /p' aten_op_config.ml | sed -n 's/^  ("\([^"]*\)".*/\1/p' | LC_ALL=C sort -c
