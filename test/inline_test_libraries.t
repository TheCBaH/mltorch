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

Dune's evaluated workspace excludes disabled local libraries, even if an
installed library has the same name. Explicit modes of enabled libraries remain.

  $ cat > workspace.sexp <<'EOF'
  > ((root /ignored)
  >  (build_context _build/default)
  >  (library ((name default_runner) (local true) (source_dir _build/default/fixture)))
  >  (library ((name both) (local true) (source_dir _build/default/fixture)))
  >  (library ((name comment) (local false) (source_dir /installed/comment)))
  >  (library ((name comment) (local true) (source_dir _build/default/other))))
  > EOF
  $ ../bin/inline_test_libraries.exe --file fixture/dune --workspace workspace.sexp | tr '\t' ':'
  fixture:default_runner:best
  fixture:both:best
  fixture:both:js

Real Dune conditions keep the host's library and exclude another architecture
and an unavailable package without building their runners.

  $ mkdir -p gated/runners
  $ echo '(lang dune 3.20)' > gated/dune-project
  $ cat > gated/runners/dune <<'EOF'
  > (library (name matching_host) (modules) (inline_tests)
  >  (preprocess (pps ppx_expect))
  >  (enabled_if (= %{architecture} %{architecture})))
  > (library (name other_host) (modules) (inline_tests)
  >  (preprocess (pps ppx_expect))
  >  (enabled_if (= %{architecture} unavailable_architecture)))
  > (library (name missing_package) (modules) (inline_tests)
  >  (preprocess (pps ppx_expect))
  >  (enabled_if %{lib-available:inline_test_discovery_missing_package}))
  > EOF
  $ (cd gated && dune describe workspace --lang 0.1 --format sexp --no-recursive runners > workspace.sexp)
  $ (cd gated && ../../bin/inline_test_libraries.exe --file runners/dune --workspace workspace.sexp) | tr '\t' ':'
  runners:matching_host:best

An invalid workspace description fails rather than silently selecting no tests.

  $ echo '()' > workspace.sexp
  $ ../bin/inline_test_libraries.exe --file fixture/dune --workspace workspace.sexp
  inline-test discovery: workspace description has no build_context
  [2]

Malformed stanzas fail discovery before timing/building an incomplete corpus.

  $ echo '(library (name broken)' > fixture/dune
  $ ../bin/inline_test_libraries.exe --file fixture/dune
  inline-test discovery: unclosed list
  [2]
  $ echo '(library (name "broken)' > fixture/dune
  $ ../bin/inline_test_libraries.exe --file fixture/dune
  inline-test discovery: unterminated quoted atom
  [2]
