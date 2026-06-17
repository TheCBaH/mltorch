Test that struct fields with non-None defaults (bool False, dict {}, list [])
are decoded correctly: absent keys use the schema default, present keys use
the JSON value.

  $ cat > default_test.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let () =
  >   let json = {|{"inputs":[],"outputs":[],"nodes":[],"tensor_values":{},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":false,"custom_obj_values":{},"sym_float_values":{}}|} in
  >   Format.printf "%a@."
  >     (Format.pp_print_result
  >        ~ok:(fun ppf g -> Format.fprintf ppf "nodes=%d" (List.length g.Graph_Type.nodes))
  >        ~error:Format.pp_print_string)
  >     (Jsont_bytesrw.decode_string Graph.jsont json)
  > EOF
  $ ocaml default_test.ml 2>/dev/null
  nodes=0
