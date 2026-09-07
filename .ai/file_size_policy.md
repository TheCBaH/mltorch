# Source file size policy

Status: **implemented**. Keeps authored source and test files small enough to
review, navigate, and test independently. Applies to the main repository only —
not submodules, generated files, downloaded model fixtures, build output, or
third-party sources.

## Size policy

Line count is the primary measure; byte/character count is a secondary signal
for unusually dense files.

| Size | Policy |
| --- | --- |
| At most 750 LOC | Preferred maximum for a new authored file. |
| 751–1,000 LOC | Do not grow without a short justification in the change. |
| 1,001–1,500 LOC | A split is expected; additions need a concrete decomposition plan. |
| More than 1,500 LOC | Exception only. Record why the file cannot reasonably be split. |

Review a file at the same time when it exceeds roughly 60 KB, even if it is
under the line limit. Generated sources must be marked as generated and are
not subject to this policy.

Snapshot (Cram, `.t`) tests have one narrow exception: a single command's
required expected output may exceed 1,500 lines. Such tests must still be one
command or model per file; unrelated snapshots must not be combined merely for
convenience. This exception isn't mechanically checkable the same way as a
line count, so Cram files are out of scope for the enforcement script below.

## Enforcement

`scripts/check-file-size.sh`, wired into CI as the `file size check` step in
`.github/workflows/build.yml` (before the devcontainer step; it uses Bash, git,
awk, and wc) and runnable locally via `make check.file-size`. Two checks,
deliberately different in mechanism:

* **Tree check (1000-line cap):** tracked `.ml`/`.mli` files in the current
  checkout, excluding `vendored/`, `modules/`, `lib/generated/`, and listed
  exceptions — no diff against a base ref. This sidesteps the
  shallow-checkout problem (the default `actions/checkout` fetch-depth of 1 has
  no history to diff against) and is a property of the tree, not of any one
  commit.
* **New-file check (750-line cap):** `.ml`/`.mli` files in the same scope with
  status `A` in `git diff-tree HEAD^ HEAD`, i.e. added by the tip commit. This needs one
  parent commit, so the checkout step's `fetch-depth` is 2, not the default 1.
  Checking only the tip commit (not the merge-base with `main`) means a
  multi-commit PR can add a 751–1000-line file in an earlier commit and leave
  it unchanged in the last one without triggering either check. This is an
  intentional simplicity tradeoff, since the tree check still catches the file if it stays over 1000
  lines. A squash commit is checked as one commit only when the script runs
  against that commit. If `HEAD^` is unavailable, the script reports that it
  is skipping the new-file check; the tree check still runs.

Both checks count working-tree contents, so local unstaged edits affect the
result. The script does not enforce the byte-size guideline, other source
languages, or the review dates in the exception list. Marking a file as
generated does not automatically exempt it outside the excluded paths.

Both checks share `scripts/file-size-exceptions.txt` (`path|review-by|reason`,
one exempt path per line, `#`-comments allowed). A file listed there is exempt
from both checks. Splitting a listed file back under the threshold should
remove its line in the same change; an entry that reaches its `review-by`
date should be re-justified or removed.

## Splitting a production module: the recurring pitfall

When decomposing an oversized module that has no `.mli` (`op_bridge.ml`,
`pointwise.ml`, `conv.ml`, `walk_meta.ml`, ...), the file's own name is often
part of its external surface — callers reference `Pointwise.Add`,
`Conv.Conv2d`, etc. **Grep every external reference to the module before
splitting**, and leave a facade file (the original filename) that re-exports
the split-out pieces under manifest aliases, so the external surface stays
source-compatible. For a module with a `.mli`, check which private helpers the
extracted code needs: a sibling `open` only sees the public interface. Move
shared private helpers into an internal module when necessary, keeping the
public interface focused on its existing contract.

A naive "next entry's line minus one" or grep-based split of comments/helpers
is not reliable — verify each candidate's real call sites (not just grep hit
counts) and confirm the split by building, not by counting.
