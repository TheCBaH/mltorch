Incremental JSON parsing tests for relevant portions of the generated schema.
Each test defines a pp function and uses Format.pp_print_result to print the
decode outcome.

-- SchemaVersion: plain struct with two int fields --

  $ cat > parse1.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let pp ppf v =
  >   Format.fprintf ppf "%d.%d" v.SchemaVersion.major v.SchemaVersion.minor
  > let () =
  >   Format.printf "%a@."
  >     (Format.pp_print_result ~ok:pp ~error:Format.pp_print_string)
  >     (Jsont_bytesrw.decode_string SchemaVersion.jsont {|{"major":8,"minor":14}|})
  > EOF
  $ ocaml parse1.ml 2>/dev/null
  8.14

-- TensorArgument: struct with a single string field --

  $ cat > parse2.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let pp ppf v =
  >   Format.fprintf ppf "name=%s" v.TensorArgument.name
  > let () =
  >   Format.printf "%a@."
  >     (Format.pp_print_result ~ok:pp ~error:Format.pp_print_string)
  >     (Jsont_bytesrw.decode_string TensorArgument.jsont {|{"name":"x"}|})
  > EOF
  $ ocaml parse2.ml 2>/dev/null
  name=x

-- ArgumentKind: enum decoded from an integer --

  $ cat > parse3.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let pp ppf = function
  >   | ArgumentKind.UNKNOWN    -> Format.fprintf ppf "UNKNOWN"
  >   | ArgumentKind.POSITIONAL -> Format.fprintf ppf "POSITIONAL"
  >   | ArgumentKind.KEYWORD    -> Format.fprintf ppf "KEYWORD"
  > let () =
  >   Format.printf "%a@."
  >     (Format.pp_print_result ~ok:pp ~error:Format.pp_print_string)
  >     (Jsont_bytesrw.decode_string ArgumentKind.jsont {|1|})
  > EOF
  $ ocaml parse3.ml 2>/dev/null
  POSITIONAL

-- SymIntArgument: union decoded from a single-key object --

  $ cat > parse4.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let pp ppf = function
  >   | SymIntArgument.Name s -> Format.fprintf ppf "Name %s" s
  >   | SymIntArgument.Int  i -> Format.fprintf ppf "Int %d" i
  > let () =
  >   Format.printf "%a@."
  >     (Format.pp_print_result ~ok:pp ~error:Format.pp_print_string)
  >     (Jsont_bytesrw.decode_string SymIntArgument.jsont {|{"as_name":"sym0"}|})
  > EOF
  $ ocaml parse4.ml 2>/dev/null
  Name sym0

-- RangeConstraint: struct with optional int fields; absent key -> None --

  $ cat > parse5.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let pp_opt ppf = function
  >   | None   -> Format.fprintf ppf "none"
  >   | Some n -> Format.fprintf ppf "%d" n
  > let pp ppf v =
  >   Format.fprintf ppf "min=%a max=%a"
  >     pp_opt v.RangeConstraint.min_val
  >     pp_opt v.RangeConstraint.max_val
  > let () =
  >   Format.printf "%a@."
  >     (Format.pp_print_result ~ok:pp ~error:Format.pp_print_string)
  >     (Jsont_bytesrw.decode_string RangeConstraint.jsont {|{"min_val":0}|})
  > EOF
  $ ocaml parse5.ml 2>/dev/null
  min=0 max=none

-- Node: explicit null on an Optional field is decoded as None --

  $ cat > parse6.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let pp_opt_bool ppf = function
  >   | None   -> Format.fprintf ppf "none"
  >   | Some b -> Format.fprintf ppf "%b" b
  > let pp ppf v =
  >   Format.fprintf ppf "target=%s is_hop=%a"
  >     v.Node.target
  >     pp_opt_bool v.Node.is_hop_single_tensor_return
  > let () =
  >   let json = {|{"target":"t","inputs":[],"outputs":[],"metadata":{},"is_hop_single_tensor_return":null}|} in
  >   Format.printf "%a@."
  >     (Format.pp_print_result ~ok:pp ~error:Format.pp_print_string)
  >     (Jsont_bytesrw.decode_string Node.jsont json)
  > EOF
  $ ocaml parse6.ml 2>/dev/null
  target=t is_hop=none

-- ComplexValue: float fields decode from plain JSON numbers --

  $ cat > parse7.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let pp ppf v =
  >   Format.fprintf ppf "real=%g imag=%g" v.ComplexValue.real v.ComplexValue.imag
  > let () =
  >   Format.printf "%a@."
  >     (Format.pp_print_result ~ok:pp ~error:Format.pp_print_string)
  >     (Jsont_bytesrw.decode_string ComplexValue.jsont {|{"real":1.5,"imag":-2.0}|})
  > EOF
  $ ocaml parse7.ml 2>/dev/null
  real=1.5 imag=-2

-- ComplexValue: the special strings decode to non-finite floats --
-- (PyTorch serializes inf/-inf/nan as "Infinity"/"-Infinity"/"NaN") --

  $ cat > parse8.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let pp ppf v =
  >   Format.fprintf ppf "real=%g imag=%g" v.ComplexValue.real v.ComplexValue.imag
  > let decode s =
  >   Format.printf "%a@."
  >     (Format.pp_print_result ~ok:pp ~error:Format.pp_print_string)
  >     (Jsont_bytesrw.decode_string ComplexValue.jsont s)
  > let () =
  >   decode {|{"real":"Infinity","imag":"-Infinity"}|};
  >   decode {|{"real":"NaN","imag":3.0}|}
  > EOF
  $ ocaml parse8.ml 2>/dev/null
  real=inf imag=-inf
  real=nan imag=3

-- ComplexValue: an unknown float string is a decode error --

  $ cat > parse9.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let pp ppf v =
  >   Format.fprintf ppf "real=%g imag=%g" v.ComplexValue.real v.ComplexValue.imag
  > let () =
  >   Format.printf "%a@."
  >     (Format.pp_print_result ~ok:pp ~error:(fun ppf _ -> Format.fprintf ppf "ERROR"))
  >     (Jsont_bytesrw.decode_string ComplexValue.jsont {|{"real":"inf","imag":0.0}|})
  > EOF
  $ ocaml parse9.ml 2>/dev/null
  ERROR

-- float_jsont: non-finite floats encode back to the special strings, finite --
-- ones to plain JSON numbers (direct round-trip of the codec) --

  $ cat > parse10.ml << 'EOF'
  > #use "topfind";;
  > #require "jsont";;
  > #require "jsont.bytesrw";;
  > #load "schema_runtime.cma";;
  > #use "schema_pytorch.ml";;
  > let enc v =
  >   match Jsont_bytesrw.encode_string float_jsont v with Ok s -> s | Error e -> e
  > let () =
  >   List.iter (fun v -> print_endline (enc v))
  >     [ Float.infinity; Float.neg_infinity; Float.nan; 1.5 ]
  > EOF
  $ ocaml parse10.ml 2>/dev/null
  "Infinity"
  "-Infinity"
  "NaN"
  1.5
