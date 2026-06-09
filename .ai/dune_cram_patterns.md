# Dune Build Patterns and Cram Test Conventions

## Cross-Directory Artifacts via Shell Rules

Dune cannot reach into git submodules excluded from its scan (submodules listed in
`.gitignore`-equivalent exclusions). The workaround is a shell rule using
`%{project_root}`:

```dune
(rule
 (target schema.yaml)
 (action
  (with-stdout-to %{target}
   (run cat %{project_root}/../../modules/pytorch/torch/_export/serde/schema.yaml))))
```

`%{project_root}` expands to `_build/default/` (the build root). `../../` steps back
to the workspace root, reaching the actual source tree file. This gives dune a build
target it can track, but there is **no dune dep-tracking for the source file** — if
`schema.yaml` changes, run `dune build` manually or `touch data/dune` to invalidate.

The same pattern is used for each `*_model.json` file in `test/dune`.

## Generated Source Files in Cram Sandboxes

Cram tests run in an isolated sandbox directory. Files listed in `(deps ...)` are
copied into the sandbox. The typical setup for a cram test that uses the generated
schema parser:

```dune
; 1. Generate schema_pytorch.ml from schema.yaml
(rule
 (target schema_pytorch.ml)
 (deps (file ../data/schema.yaml))
 (action
  (with-stdout-to %{target}
   (run %{exe:../bin/schema_gen.exe} %{dep:../data/schema.yaml}))))

; 2. Copy schema_runtime.cma + .cmi into this dir's build tree
(rule
 (target schema_runtime.cma)
 (action (copy ../lib/schema_runtime/schema_runtime.cma %{target})))

(rule
 (target schema_runtime.cmi)
 (action (copy
   ../lib/schema_runtime/.schema_runtime.objs/byte/schema_runtime.cmi
   %{target})))

; 3. Declare cram test with its deps
(cram
 (applies_to my_cram)
 (deps schema_pytorch.ml schema_runtime.cma schema_runtime.cmi))
```

The `.cmi` file matters: when `ocaml schema_runtime.cma ...` loads the archive,
the toplevel needs to resolve `open Schema_runtime`. The `.cmi` must be in the
sandbox's `.` directory so OCaml's implicit include path finds it.

The `.cma` path (`schema_runtime.cma`) is straightforward. The `.cmi` lives in
dune's hidden objs directory: `.schema_runtime.objs/byte/schema_runtime.cmi`
(convention: `.<libname>.objs/byte/`).

## Source Files Shared Between Cram Tests

Source files that exist in the source tree (e.g., `test/model_test_utils.ml`) are
directly available to dune as deps — **do not add a copy rule**. Adding a copy rule
when the file already exists in the source tree causes a conflict:

```
Error: Multiple rules generated for _build/default/test/model_test_utils.ml
```

Just list the file in `(deps ...)` and dune will copy it into the cram sandbox.

## Loading schema_runtime in the Toplevel

The OCaml batch interpreter (`ocaml`) expects `.cma` files to be loaded explicitly
and `.cmi` files to be found via the include path. The cram tests use:

```sh
ocaml schema_runtime.cma parse_model.ml 2>/dev/null
```

Stderr is suppressed because `#use "topfind"` emits "findlib" messages to stderr.
`jsont` and `jsont.bytesrw` are required via `#require` (findlib/topfind).

The `2>/dev/null` pattern is intentional: all interesting output goes to stdout, all
noise goes to stderr.

## Cram Test File Conventions

### Prose vs expected output indentation

Cram uses 2-space indentation to mark lines as part of the test block:
- Lines starting with `  $ ` are commands
- Lines starting with `  > ` are heredoc continuation (inside `<< 'EOF'`)
- Lines starting with `  ` (2 spaces, no `$`) are expected output

**Pitfall**: Prose before a `$` command that is indented 2 spaces looks like orphaned
expected output. Cram strips it silently. Prose must be at column 0.

### Blank lines in heredocs

A blank line inside a heredoc in a cram test must be written as `  > ` (two spaces,
`>`, and a trailing space), not `  >` alone. The trailing space is significant.
Workaround: restructure the script to avoid blank lines in heredocs.

### Generating expected output

If the expected output section of a cram test is empty (or wrong), run:
```sh
opam exec -- dune runtest test/my_cram.t   # shows diff
opam exec -- dune promote test/my_cram.t   # accepts actual output
```

`dune promote` writes the actual output back into the source `.t` file. Then re-run
the test to confirm it passes with no diff.

## `(applies_to ...)` Naming Convention

The `(applies_to name)` stanza in a `(cram ...)` block refers to `name.t` in the
same directory. The name must match without the `.t` extension.

## `%{exe:...}` vs `%{dep:...}`

- `%{exe:../bin/schema_gen.exe}` — references a dune-built executable; dune builds it
  as a dep automatically
- `%{dep:../data/schema.yaml}` — declares a dependency on a file and expands to its
  path in the build tree

Both can appear in the same `(run ...)` action.

## Directory Targets for Dynamic Codegen

When a rule emits a *large or dynamically-sized* file set (e.g. `torchgen` produces
dozens of headers whose exact names you don't want to enumerate), declare a **directory
target** instead of listing files:

```dune
(rule
 (targets (dir gen))
 (deps run_codegen.sh)
 (action (run bash run_codegen.sh %{project_root}/../../modules/pytorch)))
```

This requires opting in once, in `dune-project`:

```dune
(using directory-targets 0.1)
```

A consuming rule depends on the produced directory with a recursive glob — **not**
`(source_tree ...)`, which targets *source* dirs and does not create the build-ordering
edge against another rule's output:

```dune
(deps build_archive.sh (glob_files_rec gen/*) (glob_files_rec inc/*))
```

## `%{project_root}` Resolves Differently in Actions vs Deps

`%{project_root}/../../modules/pytorch` reaches the submodule **only inside an
`(action ...)`**, where the working directory is the rule's build dir
(`_build/default/<rule-dir>`), so the `../../` climbs out of `_build/default` to the
real workspace root — and this holds at any rule-directory depth.

In `(deps ...)` the same string is resolved relative to the dune file's **source**
directory, so the climb lands in the wrong place. Therefore submodule inputs are passed
as **action arguments**, never as tracked deps. As with the `schema.yaml` rule above,
this means **no dune dep-tracking** on those sources: `touch` the rule or `dune clean`
to force a rebuild when the submodule checkout changes.

## Keeping bash out of the sexp

For multi-step shell actions (configure a header, run a generator, compile + archive
many files), put the bash in a committed `*.sh` script listed in `(deps ...)` and invoke
it with `(run bash script.sh <args>)`. The script stays independently runnable/testable
and the dune file stays readable. See [aten_core_build.md](aten_core_build.md).
