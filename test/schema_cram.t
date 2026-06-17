Run the code generator on the PyTorch export schema and verify the result
loads cleanly in the OCaml toplevel.

schema_runtime.cma and schema_runtime.cmi are copied into this directory's
build tree (see test/dune) and therefore land in the cram sandbox.  The OCaml
batch interpreter includes '.' in its search path, so Schema_runtime's
interface is found without an explicit -I flag.

  $ cat > load_schema.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let () = print_endline "schema_pytorch loaded ok"
  > EOF
  $ ocaml load_schema.ml 2>/dev/null
  schema_pytorch loaded ok
