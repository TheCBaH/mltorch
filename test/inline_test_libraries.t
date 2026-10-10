Discovery preserves modes, quoted atoms and comments, and filters gates on
both libraries and inline runners. No interpreter outside OCaml is needed.

  $ mkdir fixture
  $ cat > fixture/dune <<'EOF'
  > ; fake (library (name ignored) (inline_tests))
  > (library (name "default_runner") (inline_tests))
  > (library (name both) (inline_tests (modes best js)))
  > (library (name not_inline))
  > (library (name wasm) (enabled_if (= %{env:MLTORCH_WASM=0} 1)) (inline_tests))
  > (library (name compcert) (inline_tests (enabled_if (= %{env:MLTORCH_COMPCERT=0} 1))))
  > (library (name comment) (inline_tests) ; ignored )
  > )
  > EOF
  $ ../bin/inline_test_libraries.exe --file fixture/dune | tr '\t' ':'
  fixture:default_runner:best
  fixture:both:best
  fixture:both:js
  fixture:comment:best

Malformed stanzas fail discovery before timing/building an incomplete corpus.

  $ echo '(library (name broken)' > fixture/dune
  $ ../bin/inline_test_libraries.exe --file fixture/dune
  inline-test discovery: unclosed list
  [2]
  $ echo '(library (name "broken)' > fixture/dune
  $ ../bin/inline_test_libraries.exe --file fixture/dune
  inline-test discovery: unterminated quoted atom
  [2]
