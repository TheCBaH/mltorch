Nominal boundaries: what the type checker must reject.

Each semantic ID, digest, name and text role is its own abstract type. Every
negative case is paired with a control that must still compile; without the
controls a broken harness would reject everything and look like a pass.

  $ R=../../..
  $ check() { sh $R/test/inference_compare/type_boundaries.sh $R/test/inference_compare/schema_probe.exe "$1" "$2"; }

Identifiers of different roles do not unify.

  $ check "compare unit id with case id" "ignore (Id.Unit.equal u c)"
  compare unit id with case id: rejected
  $ check "compare unit id with itself" "ignore (Id.Unit.equal u u)"
  compare unit id with itself: COMPILES

A string is not an identifier; entry is [of_string], which validates.

  $ check "pass a string as a unit id" "ignore (Id.Unit.equal u s)"
  pass a string as a unit id: rejected
  $ check "unit id as a string by coercion" "ignore (u : string)"
  unit id as a string by coercion: rejected
  $ check "unit id as a string by to_string" "ignore (Id.Unit.to_string u)"
  unit id as a string by to_string: COMPILES
  $ check "treat of_string as total" "ignore (Id.Unit.equal u (Id.Unit.of_string s))"
  treat of_string as total: rejected

Digests of different domains do not unify, and a digest is not a string.

  $ check "compare graph digest with payload digest" "ignore (Digest.Graph.equal g p)"
  compare graph digest with payload digest: rejected
  $ check "compare graph digest with itself" "ignore (Digest.Graph.equal g g)"
  compare graph digest with itself: COMPILES
  $ check "graph digest as a string by coercion" "ignore (g : string)"
  graph digest as a string by coercion: rejected

Prose is neither an identifier nor other prose.

  $ check "description for a diagnostic text" "ignore (Prose.Diagnostic_text.equal x d)"
  description for a diagnostic text: rejected
  $ check "description for a unit id" "ignore (Id.Unit.equal u d)"
  description for a unit id: rejected
  $ check "description with itself" "ignore (Prose.Description.equal d d)"
  description with itself: COMPILES

Operand and result positions are different types.

  $ check "input port for an output port" "ignore (Output_port.equal o i)"
  input port for an output port: rejected
  $ check "input port with itself" "ignore (Input_port.equal i i)"
  input port with itself: COMPILES
  $ check "port as an int by coercion" "ignore (i : int)"
  port as an int by coercion: rejected
  $ check "port as an int by to_index" "ignore (Input_port.to_index i)"
  port as an int by to_index: COMPILES

Node and tensor references are different types even with equal fields.

  $ check "tensor ref compared as a node ref" "ignore (Node_ref.equal nr tr)"
  tensor ref compared as a node ref: rejected
  $ check "node ref with itself" "ignore (Node_ref.equal nr nr)"
  node ref with itself: COMPILES
