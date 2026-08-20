# Working in a git worktree on this project

Hard-won notes for anyone (human or agent) doing isolated work in a
`.claude/worktrees/<name>` checkout of tfview. Several things bite, in order.

Background agents are pushed into a worktree by an isolation guard: edits to the
shared `/workspaces/tfview` checkout are **rejected** until you `EnterWorktree`
(the guard names the worktree path to retry under). So the worktree is not
optional — it's the only place edits land. The rest of this file is what you then
have to set up by hand.

## 1. Base the worktree on your working branch, not `origin/master`

The default `EnterWorktree` base ref is `fresh` → it branches from
`origin/<default-branch>` (here `origin/master`). This project's active work
lives on **`devel`**, which is far ahead of `master` (e.g. the whole `lib/native`
engine doesn't exist on `master`). A `fresh` worktree therefore looks *empty* of
recent work and any edits would be against stale code.

**Fixes (either one):**

- **Recommended:** set `worktree.baseRef: "head"` in `.claude/settings.json` so
  `EnterWorktree` branches from your current local HEAD (see §"Setup" below). This
  is the one-time fix that prevents the whole problem.
- Or create the worktree manually off HEAD and enter it by path:
  ```sh
  git worktree add .claude/worktrees/<name> -b <branch> HEAD
  # then: EnterWorktree path=.claude/worktrees/<name>
  ```

Confirm you're on the right base before editing: `git log --oneline -2` should
show your recent commits, and `ls lib/native` should exist.

## 2. Worktrees don't bring submodules — init them

`git worktree add` does **not** populate submodules. Two vendored libraries are
git submodules and **both** are needed before any dune command works:

```sh
git submodule update --init --recursive vendored/opickle vendored/ocaml-yamlt
```

- **`vendored/ocaml-yamlt`** provides the `yamlt` library. Missing → the very
  first `dune` invocation that touches it dies with `Library "yamlt" not found`
  (`pytorch_schema`/`aten_schema` depend on it, and `aten_schema` feeds
  `bin/aten_ops_gen` → the whole `lib/aten` build chain). It's a plain submodule,
  no nesting.
- **`vendored/opickle`** is in-tree itself but contains *nested* submodules
  (`serde-pickle`, `cpython`) — hence `--recursive`. dune reads *every* `dune`
  file at rule-setup time, and `vendored/opickle/test/dune` globs
  `…/serde-pickle/test/data/*.pickle`; if that's missing, **any** dune command
  fails with `Cannot find directory: …serde-pickle/test/data`, even one that
  never touches opickle.

The two big content submodules — `modules/pytorch` (the ATen C++ sources) and
`modules/devcontainer.pytorch-image-models` (real model weights for the gated cram tests) — are
*not* initialized this way; see §5 for how to make the aten build / model tests
see them without re-cloning gigabytes.

## 3. dune commands need `--root .` inside the worktree

The worktree lives under `.claude/`, a dot-dir that dune's source scan **ignores**.
Run from the worktree without `--root` and dune walks up, roots at the *main*
checkout (`/workspaces/tfview`), and builds **that** tree — silently ignoring your
worktree edits. Always pass `--root .`:

```sh
opam exec -- dune build   --root . lib/native
opam exec -- dune runtest --root . test/native
```

Corollary: `make build` / `make runtest` (run from the repo root) operate on the
main checkout and will **not** see worktree edits. Use the `dune … --root .` forms
above while working in a worktree; merge the branch before relying on `make`.

## 4. dune isn't on PATH — go through opam

`dune` is not directly on PATH here; invoke it as `opam exec -- dune …` (the
Makefile does the same). Do not run Dune commands concurrently in the same
checkout/build root: serialize `dune build`, `dune runtest`, `dune promote`,
`dune fmt`, and Makefile targets that invoke Dune. Concurrent invocations contend
for `_build` state and can produce confusing lock waits or partial promotion
results. Formatting:

```sh
opam exec -- dune build --root . @fmt --auto-promote   # apply
opam exec -- dune build --root . @fmt                  # verify clean (exit 0)
```

## 5. Building the ATen C++ library (`lib/aten`) in a worktree

`lib/native` and most tests are pure OCaml and build with just §2's submodules.
But `lib/aten` compiles a 70 MB static archive (`libaten_core.a`) from the
**`modules/pytorch`** sources via `build_archive.sh`, and a worktree's
`modules/pytorch` is an empty gitlink. Re-initializing that submodule means
cloning a huge tree *plus* its `third_party/{fmt,cpuinfo}` — don't. Instead
**symlink** to the already-populated submodule in the main checkout:

```sh
rmdir modules/pytorch && ln -s /workspaces/tfview/modules/pytorch modules/pytorch
# same for the gated model tests, if you run them:
rmdir modules/devcontainer.pytorch-image-models && \
  ln -s /workspaces/tfview/modules/devcontainer.pytorch-image-models modules/devcontainer.pytorch-image-models
```

This works because the `lib/aten`/`data` dune rules reach the sources as
`%{project_root}/../../modules/pytorch`, which from the worktree's build tree
resolves to `<worktree>/modules/pytorch` — i.e. the symlink. (`ccache` is the
compiler launcher; a cold worktree archive build is the full ~190-object
compile, but on 24 cores it's a couple of minutes and warms ccache for reruns.)

Run a *single* library's inline tests without dragging in unrelated test libs
(which would need every submodule) via its generated alias, e.g.:

```sh
opam exec -- dune build --root . @test/runtest-aten_ops_test --force
```

`@test/runtest` (no suffix) builds *all* test libraries — only use it once every
submodule above is in place.

## 6. git + `dune promote` gotchas with the §5 symlinks

- **`dune promote` rejects `--root`.** To accept an expect/cram diff in a
  worktree, use `dune runtest --root . <path> --auto-promote` instead of the
  bare `dune promote <path>` from CLAUDE.md.
- **git refuses to operate while a submodule path is a symlink.** With the §5
  symlinks in place, even `git status` aborts with
  `expected submodule path 'modules/pytorch' not to be a symbolic link`. Before
  any git command (status/add/commit), restore them to empty dirs — the build is
  done by then, so you don't need the sources anymore:
  ```sh
  for p in modules/pytorch modules/devcontainer.pytorch-image-models; do
    [ -L "$p" ] && { rm "$p"; mkdir "$p"; }
  done
  ```
  The emptied gitlinks then show as clean (uninitialized submodules), so only
  your real source edits appear in `git status`.

## Quick start (copy-paste)

```sh
# from the worktree, once:
git submodule update --init --recursive vendored/opickle vendored/ocaml-yamlt
# only if you build lib/aten or run model tests (see §5):
rmdir modules/pytorch && ln -s /workspaces/tfview/modules/pytorch modules/pytorch
# thereafter:
opam exec -- dune build   --root . lib/native
opam exec -- dune runtest --root . test/native
opam exec -- dune build   --root . @fmt --auto-promote
# before committing, if you made the §5 symlinks: restore them (see §6)
```

## Setup

This file is pointed to from `CLAUDE.md`, so it's loaded each session — read it
before starting worktree work.

Recommended one-time setting (not yet committed — opt in if you want it):

```jsonc
// .claude/settings.json
{ "worktree": { "baseRef": "head" } }
```

`baseRef: "head"` makes new worktrees branch from the current branch instead of
`origin/master` (§1). It's a behavior change to `EnterWorktree`/`--worktree`, so
it's left for a human to enable deliberately.
