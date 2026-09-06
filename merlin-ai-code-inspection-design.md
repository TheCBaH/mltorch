# Merlin-based OCaml code inspection for AI agents

Status: design proposal  
Date: 2026-09-06  
Target repository: `mltorch`  
Proposed command name: `ocaml-inspect`

## 1. Summary

This document proposes a small semantic code-inspection tool for AI coding agents working on Dune-based OCaml projects. The tool presents a stable, compact interface over Merlin and Dune, supplies source context with semantic answers, makes the freshness and completeness of every answer explicit, and can be exposed either as a command-line program or as an MCP/native agent tool.

The tool is intended to complement, not replace, textual search. A good default discovery loop is:

1. Use `rg` to find likely files or symbols cheaply.
2. Use `ocaml-inspect outline` to understand the selected file.
3. Use `ocaml-inspect inspect` at a location to obtain the enclosing construct, inferred type, definition, documentation, and nearby uses.
4. Use `definition` and `references` when exact navigation is required.
5. Request a detailed typed browse tree only as a debugging escape hatch.

The first implementation should be a shell-based executable specification around `ocamlmerlin` and Dune. Its output schema, fixtures, and evaluation harness should be treated as the durable product. A persistent OCaml backend using `merlin-lib` can replace the query engine later if benchmarks show meaningful latency or integration gains.

The central design rule is that semantic confidence must never be implicit. If a project index is absent, stale, unsupported, or failed to rebuild, the tool must return a useful partial result and label it `partial`; it must not present buffer-local occurrences as exhaustive project references.

## 2. Motivation

AI agents frequently inspect code through `rg`, `find`, `sed`, and whole-file reads. Those tools are universal and excellent for lexical discovery, but they cannot reliably answer questions such as:

- Which declaration does this identifier resolve to?
- What type did the compiler infer here?
- Is this occurrence a definition, read, write, or unrelated textual match?
- Which expression or declaration encloses this cursor?
- Does an interface hide or alter the implementation signature?
- Is the apparent module name an alias introduced by wrapping?
- Are all project references represented, or only those in the current buffer?

Merlin already answers most of these questions using the compiler's typed representation. Dune already knows the build context, flags, generated sources, preprocessors, dependencies, and which compiler artifacts need rebuilding. The missing component is a narrow AI-oriented interface that:

- combines the useful queries;
- manages Dune-owned freshness;
- normalizes version-sensitive Merlin output;
- attaches minimal source evidence;
- emits bounded, token-efficient results;
- exposes uncertainty and degraded operation;
- produces instrumentation suitable for controlled evaluation.

## 3. Findings from the current repository

### 3.1 Verified toolchain

The investigation used the active opam switch with OCaml 4.14.3 and installed:

- `merlin-lib` 4.19-414;
- `merlin` 4.19-414;
- `dot-merlin-reader` 4.19-414;
- the `ocamlmerlin` executable available through `opam exec`.

The normal one-shot invocation is:

```sh
opam exec -- ocamlmerlin single COMMAND -filename path/to/file.ml < path/to/file.ml
```

`single` performs one query. Merlin also provides a reusable server protocol. Results are returned in a JSON envelope with a response class, value, and notifications.

The installed command set includes structural, navigation, completion, type, diagnostic, and refactoring queries. The most relevant commands are:

- `outline` for a named hierarchy of a source file;
- `dump browse` for a detailed typed syntax tree;
- `type-enclosing` and `type-expression` for inferred types;
- `locate` and `locate-type` for definitions;
- `occurrences buffer|project` for references;
- `enclosing`, `shape`, `jump`, and `phrase` for structural navigation;
- `complete-prefix`, `search-by-type`, and environment queries for discovery;
- `document` for attached documentation;
- `errors` for diagnostics;
- `expand-ppx` for preprocessor debugging.

Merlin versions differ. A wrapper must discover supported commands at runtime and expose capabilities instead of assuming that every installed version has newer operations such as richer rename, extraction, locate-type, or inlay-hint variants.

### 3.2 Example results in `mltorch`

Representative queries established that Merlin is useful on this codebase:

- `outline` on [`lib/expr/expr_api.ml`](../lib/expr/expr_api.ml) found the large top-level signature and its nested modules, types, constructors, labels, and values. It exposed modules including `Eval`, `Rewrite`, `Fold`, and `Check` without reading the entire file.
- `dump browse` on [`lib/expr/expr.ml`](../lib/expr/expr.ml) returned a detailed typed, AST-like hierarchy.
- At a use of `Rewrite.alpha_normalize` in [`test/expr/rewrite_test.ml`](../test/expr/rewrite_test.ml), `type-enclosing` returned `Value.t -> Value.t`.
- `locate` resolved that use to `lib/expr/expr_api.ml:640:4`.
- completion for `Rewrite.a` included `alpha_normalize`.
- project occurrences at that location returned six occurrences in the test file under the available configuration.
- `errors` on the inspected interface returned no errors.

There are also important limitations. A locate query through an `Expr_internal` alias in the wrapped/private implementation did not always reach the underlying `.ml` definition. Aliases, wrapped libraries, hidden modules, generated code, and version-specific behavior therefore need explicit fixtures and graceful fallback.

### 3.3 Dune and `.ocaml-index`

The OCaml index database is a Dune build artifact under `_build`; the wrapper must not own or invent it. Dune owns the dependency graph and output locations, while the separate `ocaml-index` executable is used by the corresponding rules.

The relevant build targets are:

```sh
dune build @check
dune build @ocaml-index
dune build --watch @ocaml-index
```

`@check` builds compiler artifacts such as `.cmi`, `.cmt`, and `.cmti` that editor queries need. `@ocaml-index` includes the required typed artifacts and builds project indexes for project-wide occurrence queries. An ordinary `dune build` is not a guarantee that the index alias has been materialized.

In this checkout, `dune ocaml merlin dump-config lib/expr` reported intended `INDEX .../cctx.ocaml-index` paths, but no `*.ocaml-index` files were present under `_build/default`. Building `@ocaml-index` failed because this switch lacks a compatible `ocaml-index` executable; currently available package versions require a newer OCaml/Merlin combination. Other unrelated build failures may also prevent a full alias build.

Consequently, project-reference results in the present switch cannot be assumed exhaustive. The tool must detect this condition and return a partial result with an actionable warning.

## 4. Goals and non-goals

### 4.1 Goals

The tool should:

- answer common semantic code-discovery questions with one bounded call;
- work from a file and a precise source position;
- preserve the active opam/Dune build context;
- use Dune to refresh compiler and index artifacts incrementally;
- continue with partial local answers when refresh or indexing fails;
- expose freshness, scope, completeness, and provenance in every result;
- return stable, compact, source-grounded output for language models;
- support human-readable debugging and canonical machine-readable output;
- be safe to invoke on paths and identifiers supplied by an agent;
- be measurable against `rg` in reproducible end-to-end agent tasks;
- remain useful across multiple Merlin versions through capability detection.

### 4.2 Non-goals

The first version will not:

- replace Dune, ocaml-lsp, an editor, or general shell tools;
- construct a perfect whole-repository semantic graph;
- promise complete references when no valid project index exists;
- expose the raw Merlin browse tree by default;
- implement code mutation or refactoring in the initial release;
- hide build or analysis failures behind lexical guesses;
- normalize all compiler-version-dependent OCaml syntax into a new AST;
- solve generated-source mapping for every PPX on day one.

## 5. User model and primary workflows

The primary caller is an AI coding agent. Humans should also be able to run and debug the command directly.

### 5.1 Locate and understand a symbol

```text
rg "alpha_normalize" .
ocaml-inspect inspect test/expr/rewrite_test.ml:91:25
```

The second command should answer, in one result:

- the selected symbol and inferred type;
- the enclosing declaration or expression;
- the resolved declaration and implementation locations, if known;
- a short declaration/source excerpt;
- a bounded set of references;
- whether those references are buffer-local or project-complete;
- warnings about aliases, stale artifacts, or failed refresh.

### 5.2 Understand a large file

```text
ocaml-inspect outline lib/expr/expr_api.ml --depth 2
ocaml-inspect outline lib/expr/expr_api.ml --under Rewrite --depth 3
```

The default output should be shallow. The caller can expand only the relevant subtree rather than ingesting a thousand-line interface or full typed AST.

### 5.3 Find usages before an edit

```text
ocaml-inspect references lib/expr/expr_api.ml:640:4 --project --limit 40
```

If the index is unavailable, the response remains useful but starts with a partial-completeness marker and explains what was searched. An optional lexical fallback must be labeled as lexical, not silently merged into semantic results.

### 5.4 Diagnose a broken query

```text
ocaml-inspect doctor
ocaml-inspect inspect path/to/file.ml:12:8 --trace --format jsonl
```

Machine output remains on stdout. Diagnostic trace data goes only to stderr.

## 6. Proposed command-line interface

```text
ocaml-inspect capabilities [--format ai|jsonl|json|raw]
ocaml-inspect status       [--format ai|jsonl|json]
ocaml-inspect doctor       [--format text|json]

ocaml-inspect outline FILE [--depth N] [--under NAME] [--limit N]
ocaml-inspect inspect LOCATION [--references N] [--context N]
ocaml-inspect type LOCATION
ocaml-inspect definition LOCATION [--follow-aliases N]
ocaml-inspect references LOCATION [--buffer|--project] [--limit N]
ocaml-inspect errors FILE [--limit N]
ocaml-inspect browse FILE [--at LOCATION] [--depth N] [--limit N]

ocaml-inspect refresh [--check|--index|--all]
```

`LOCATION` is `FILE:LINE:COLUMN`. Lines are one-based. Columns are zero-based UTF-8 byte offsets unless a particular Merlin version establishes a different interpretation; the effective convention is reported by `capabilities` and encoded in structured metadata.

Common options:

```text
--format ai|jsonl|json|raw
--refresh auto|always|never
--workspace PATH
--context LINES
--limit N
--timeout DURATION
--trace
```

Defaults:

- `--format ai` for direct shell use by an agent;
- `--refresh auto`;
- a small source context, normally one declaration or three lines around a use;
- bounded outlines and at most 20 references unless requested otherwise;
- no raw Merlin payload or timing fields on stdout.

### 6.1 Exit codes

Suggested stable exit codes are:

| Code | Meaning |
| ---: | --- |
| 0 | Query succeeded, including explicitly marked partial results |
| 2 | Invalid arguments, location, or path |
| 3 | Workspace or Dune configuration unavailable |
| 4 | Merlin unavailable or incompatible |
| 5 | Query failed with no usable result |
| 6 | Timeout |
| 7 | Internal/schema error |

Partial semantic coverage is data, not necessarily process failure. This lets an agent consume a buffer result even when `@ocaml-index` cannot build. Callers that require completeness inspect the `status` and `completeness` fields.

## 7. Output contract

### 7.1 Design principles

Every response should make six facts easy to recover:

1. What entity was selected?
2. What is its type and enclosing structure?
3. Where is it declared or implemented?
4. Where is it used?
5. How fresh are the inputs?
6. How complete and trustworthy is the answer?

The output contract is versioned independently of Merlin. Paths are workspace-relative when inside the workspace. Empty and null fields are omitted. Result order is deterministic.

### 7.2 Canonical response model

The logical response has these fields:

```json
{
  "schema_version": 1,
  "command": "inspect",
  "status": "partial",
  "target": {
    "file": "test/expr/rewrite_test.ml",
    "position": { "line": 91, "column": 25 },
    "column_encoding": "utf8-byte"
  },
  "completeness": {
    "definitions": "project",
    "references": "buffer"
  },
  "freshness": {
    "buffer": "current",
    "compiler_artifacts": "dune-checked",
    "project_index": "missing"
  },
  "result": {
    "symbol": {
      "kind": "value",
      "name": "Rewrite.alpha_normalize",
      "type": "Value.t -> Value.t"
    },
    "definition": {
      "location": "lib/expr/expr_api.ml:640:4-640:43",
      "source": "val alpha_normalize : Value.t -> Value.t"
    },
    "references": {
      "total": 6,
      "returned": 6,
      "truncated": false,
      "items": []
    }
  },
  "warnings": [
    {
      "code": "project_index_unavailable",
      "message": "References are not project-complete."
    }
  ]
}
```

The actual result fields vary by command, but the envelope does not.

### 7.3 AI text format

The default AI format is a compact, line-oriented projection of the canonical model:

```text
@meta inspect partial defs=project refs=buffer index=missing
@sym s0 value Rewrite.alpha_normalize :: Value.t -> Value.t
@file f0 lib/expr/expr_api.ml
@file f1 test/expr/rewrite_test.ml
@def s0 f0:640:4-640:43 | val alpha_normalize : Value.t -> Value.t
@ref s0 f1:91:14-91:37 | let normalized = Rewrite.alpha_normalize value in
@warn project_index_unavailable | references are buffer-local
```

This representation is intentionally easy to scan, quote, and parse. Files and symbols are interned once, repeated locations are compact, and metadata is expressed only when useful.

The format must still have a grammar and golden tests. It should not become an ad hoc prose mode whose wording changes between releases.

### 7.4 JSON Lines

JSONL is the canonical streaming and batch interchange format. It works well because:

- each record can be consumed before a large query completes;
- a failed or truncated operation still leaves valid preceding records;
- shell pipelines can filter records incrementally;
- tool events and benchmark traces fit the same storage model;
- schema validation can be applied record by record.

JSONL is not automatically token-efficient. Repeated keys, absolute paths, nulls, and verbose nested locations can make it larger than compact text. Token savings come from normalization and selection:

- emit a metadata record once;
- intern files and symbols;
- use compact range strings in the AI projection;
- omit empty, null, default, timing, cache, and raw fields;
- cap and rank results;
- collapse adjacent source excerpts;
- return shallow outlines by default;
- provide continuation tokens for additional pages.

A JSONL result can use records such as `meta`, `file`, `symbol`, `definition`, `reference`, `warning`, and `end`. The final `end` record contains `total`, `returned`, `truncated`, and a continuation token when applicable.

### 7.5 Source evidence

Semantic answers without source context often force an immediate file-read call. Each definition and reference should therefore carry just enough evidence:

- one declaration or signature line for definitions;
- one matching line for references;
- a caret or explicit range for the focused occurrence;
- a short enclosing declaration for `inspect`;
- no duplicated full-file excerpts.

Generated or external paths should be labeled. When a source location cannot be mapped back from `_build`, the result should preserve the build path and state that mapping is unresolved.

### 7.6 Provenance and fallback

Semantic and lexical evidence must remain distinguishable:

```json
{
  "method": "textual-fallback",
  "semantic": false,
  "reason": "project_index_unavailable"
}
```

Lexical matches may supplement a partial semantic result, but they must not change its completeness to `project`. Duplicate semantic and textual hits can be merged only while retaining both provenance markers.

### 7.7 Token budgets

Initial output targets are:

| Query | Default output target |
| --- | ---: |
| `inspect` | 300–800 model tokens |
| shallow `outline` | at most about 1,500 tokens |
| `definition` or `type` | normally under 300 tokens |
| `references` | first 20 results, expandable to 50 |
| raw `browse` | never implicit; always bounded |

These are budgets, not truncation by byte slicing. The encoder should rank and select complete records, declare truncation, and provide a continuation mechanism.

## 8. Freshness and index management

### 8.1 Dune is the authority

The tool should not decide freshness by comparing source and artifact timestamps. Timestamp checks miss generated files, PPX dependencies, compiler flags, interfaces, libraries, and Dune rule changes. Instead, it asks Dune to build the smallest relevant alias and relies on Dune's incremental dependency engine.

### 8.2 Refresh policy

`--refresh auto` chooses a refresh class from the query:

| Query class | Required state | Automatic action |
| --- | --- | --- |
| outline, local type, local errors | current buffer and Merlin config | no global build unless configuration/artifacts are absent |
| cross-file definition | typed compiler artifacts | `dune build @check` or a narrower derivable target |
| project references | project index | `dune build @ocaml-index` |
| raw browse | current buffer and config | no project index |

`--refresh always` reruns the corresponding Dune request even if a cached process believes it is current. Because Dune is incremental, a no-change refresh should be cheap.

`--refresh never` is useful for controlled benchmarks and debugging. The response still reports the observed artifact state.

### 8.3 Persistent/watch mode

For interactive sessions, a supervisor may run:

```sh
dune build --watch @ocaml-index
```

or maintain a Dune RPC connection. The query frontend must not assume that a watcher exists. It should detect the service, reuse it when healthy, and fall back to a one-shot Dune request.

### 8.4 Freshness state machine

Suggested states are:

- `current`: the in-memory buffer supplied to Merlin is the requested content;
- `dune-checked`: Dune successfully evaluated the relevant typed-artifact rules;
- `indexed`: Dune successfully evaluated the index alias and index files are discoverable;
- `stale-allowed`: refresh was disabled and existing artifacts were used;
- `missing`: required artifacts do not exist;
- `failed`: refresh was attempted and failed;
- `unsupported`: the active toolchain cannot build the artifact.

A refresh failure does not suppress a useful query result. It changes `status` to `partial`, records the failed component, and emits a concise diagnostic with a remediation hint. Full build logs remain on stderr or in a trace file, not in normal AI output.

### 8.5 Index discovery

Index locations should be obtained from Dune/Merlin configuration, not guessed from a single hard-coded `_build/default` layout. `dune ocaml merlin dump-config` exposes `INDEX` entries for contexts. Discovery must handle:

- multiple build contexts;
- nested workspaces;
- vendored libraries;
- installed dependencies and source trees;
- absent index files despite configured paths;
- version incompatibility between OCaml, Merlin, Dune, and `ocaml-index`.

`status` should summarize configured and materialized indexes separately.

## 9. Architecture

```text
                 +-----------------------+
source buffer -->| Merlin query engine   |--+
                 +-----------------------+  |
                                              v
                 +-----------------------+  normalize  +----------------+
Dune workspace ->| config/refresh/index  |------------>| response model |
                 +-----------------------+              +----------------+
                                                                |
                                      +-------------------------+------------------+
                                      v                         v                  v
                                compact AI text              JSON/JSONL           raw/debug
                                      |                         |
                                      +------------+------------+
                                                   v
                                           shell agent or MCP
```

The system has four separable layers:

1. **Workspace layer:** validates paths, finds the Dune root, selects the opam switch, reads Merlin configuration, and requests refreshes.
2. **Query layer:** executes supported Merlin operations against the exact buffer.
3. **Normalization layer:** converts version-specific payloads into the stable response model, adds source evidence, ranks results, and marks completeness.
4. **Adapters:** render compact text, JSONL, JSON, or an MCP/native tool response and emit traces.

Keeping the response model independent of the backend permits a shell prototype and OCaml service to coexist and be compared on identical fixtures.

### 9.1 Proposed MCP/native surface

Agents use a semantic tool more reliably when its parameters are typed and its output is structured. A single discriminated operation avoids flooding the agent with many nearly identical tool descriptions:

```json
{
  "action": "inspect | outline | definition | references | type | errors",
  "file": "lib/expr/expr_api.ml",
  "line": 640,
  "column": 4,
  "scope": "buffer | project",
  "depth": 2,
  "limit": 20,
  "refresh": "auto | always | never"
}
```

The MCP adapter should call the same core as the CLI. It should not duplicate freshness or normalization logic. The shell command remains valuable for portability, manual diagnosis, replay, and use by agents without MCP configuration.

## 10. Shell implementation versus an OCaml implementation

### 10.1 Shell-based first version

A shell frontend around `opam exec -- ocamlmerlin single` is appropriate for an executable specification because it:

- uses the exact Merlin executable from the active switch;
- isolates compiler/version compatibility naturally through opam;
- is quick to inspect, modify, and deploy in arbitrary repositories;
- makes Dune commands and failure modes visible;
- provides a dependable fallback even if a service cannot start.

Its costs are repeated process startup, repeated parsing and typing across aggregated queries, JSON transformation complexity, and weaker internal types. The implementation should keep shell limited to orchestration and use a robust JSON processor or a small helper for normalization; it must not parse JSON with regular expressions.

### 10.2 OCaml backend using `merlin-lib`

An OCaml service can construct a pipeline once and dispatch multiple `Query_commands` against it. This is attractive for `inspect`, which otherwise performs several near-identical one-shot invocations. Advantages include:

- reuse of parsing, typing, configuration, and caches;
- direct typed access without serializing intermediate Merlin JSON;
- typed response construction and error handling;
- more natural persistent sessions and Dune RPC integration;
- lower latency for repeated and batched queries.

The major constraint is compatibility. Merlin explicitly describes its library API as unstable, and ocaml-lsp vendors Merlin because it relies on those internals. An OCaml backend may therefore need a build variant per OCaml/Merlin switch, pinned dependencies, and adaptation work at upgrades. A one-shot OCaml binary that rebuilds the pipeline for every request captures little of the main benefit.

### 10.3 Recommendation

Use a staged implementation:

1. Define the response schema, compact grammar, exit codes, fixtures, and benchmarks.
2. Implement a shell CLI that establishes behavior and works through `opam exec`.
3. Keep the shell CLI as a compatibility fallback.
4. Implement a persistent OCaml backend only after measurements identify repeated Merlin startup/typechecking or Dune integration as material bottlenecks.
5. Run the same golden and agent-level evaluations against both backends.

The implementation language does not materially determine model-token cost. Ranking, source selection, interning, truncation, and schema design do.

## 11. Safety and robustness

The command will often receive agent-generated paths and identifiers. It must:

- resolve and validate workspace-relative paths;
- reject traversal outside the workspace unless explicitly enabled;
- pass arguments as argument vectors and quote shell values correctly;
- never use `eval` or interpolate a generated command string;
- impose time, record, byte, and recursion-depth limits;
- keep stdout valid for the selected machine format;
- redact environment values and credentials from traces;
- avoid including arbitrary build logs in model-facing results;
- treat source text as data, never as terminal control or shell syntax;
- preserve the original file unless unsaved buffer content is explicitly passed;
- use stable error codes and structured warning codes.

The first release should be read-only. Any future refactoring operation needs a separate mutation contract, preview/diff support, and explicit authorization semantics.

## 12. Observability and debugging

### 12.1 Trace output

`--trace` writes structured diagnostic events to stderr. It records:

- tool, OCaml, Merlin, Dune, and `ocaml-index` versions;
- workspace and normalized target, without secret environment values;
- capability discovery and selected query strategy;
- Dune target and refresh result;
- configured/materialized index state;
- duration of `config`, `refresh`, `merlin`, `source_context`, `normalize`, `encode`, and `total` stages;
- result counts before and after deduplication/ranking;
- truncation, output bytes, and estimated output tokens;
- cache hit/miss information for a persistent backend.

Trace event names and fields should be versioned so benchmark tooling can consume them.

### 12.2 Doctor command

`ocaml-inspect doctor` runs a bounded diagnostic sequence:

1. identify workspace and opam switch;
2. report exact tool versions;
3. confirm Merlin configuration for a representative source file;
4. exercise a local query;
5. check `@check` without forcing unrelated targets where possible;
6. enumerate configured and materialized index paths;
7. determine whether `@ocaml-index` is supported;
8. exercise cross-file locate and project occurrences when fixtures exist;
9. provide remediation commands.

Doctor output should distinguish a broken tool from a project that already has compilation errors.

## 13. Test strategy

### 13.1 Unit and schema tests

- parse and normalize every supported Merlin envelope class;
- validate all JSON and JSONL records against a checked-in schema;
- round-trip AI-format escaping for spaces, Unicode, tabs, and delimiters;
- verify deterministic ordering and path interning;
- verify caps, continuation, and truncation declarations;
- verify warning and exit-code mappings;
- ensure stdout contains no trace/log contamination.

### 13.2 Semantic fixture project

The fixture should cover:

- wrapped and unwrapped libraries;
- `.mli` declarations and `.ml` implementations;
- module aliases, includes, functors, and private modules;
- local opens and shadowing;
- polymorphic and inferred types;
- PPX-generated code and source mapping;
- deliberately broken buffers;
- generated modules;
- Unicode identifiers/comments and paths containing spaces;
- a symbol with enough references to exercise pagination;
- interface and dependency changes that invalidate downstream artifacts.

For every fixture location, record an oracle for selected entity, type, definition, expected semantic references, and expected completeness under each index state.

### 13.3 Integration scenarios

Run at least:

- cold workspace with no `_build`;
- warm typed artifacts but no index;
- fully indexed workspace;
- no-change automatic refresh;
- one implementation-file edit;
- one interface edit with downstream invalidation;
- deliberately stale artifacts with `--refresh never`;
- broken build where a buffer-local answer remains possible;
- missing and incompatible `ocaml-index` executables;
- large outline/reference output;
- repeated queries against a persistent backend.

Correctness invariants matter more than latency: no false completeness, stable locations, correct position conventions, declared truncation, and preserved provenance.

## 14. Evaluating usefulness against `rg`

Because language-model behavior is probabilistic and the tool will evolve, evaluation must separate three questions:

1. **Tool correctness:** does the command return the right semantic facts?
2. **Retrieval efficiency:** does its output contain useful evidence per token and per unit of latency?
3. **Agent effectiveness:** does access to it improve completion quality or reduce effort on realistic tasks?

### 14.1 Conditions

Compare at least:

- `rg` plus ordinary file reads;
- Merlin tool only where applicable;
- hybrid `rg` plus Merlin tool;
- previous and candidate tool versions.

The hybrid condition is likely the practical winner and should be treated as a first-class condition, not contamination.

For a fair tool comparison, expose `rg` and `ocaml-inspect` through instrumented interfaces with equivalent visibility into arguments, output size, failures, and duration. Otherwise a model can hide multiple searches in one shell command, making raw tool-call counts misleading.

### 14.2 Task corpus

Store tasks as immutable JSONL records tied to a fixed project commit:

```json
{
  "task_id": "locate-alpha-normalize",
  "repo_commit": "<sha>",
  "prompt_file": "tasks/locate-alpha-normalize.md",
  "oracle_file": "oracles/locate-alpha-normalize.json",
  "allowed_tools": ["read", "rg", "ocaml-inspect"],
  "budget": { "turns": 30, "seconds": 900 }
}
```

Include navigation, explanation, bug diagnosis, targeted edits, interface changes, alias-heavy cases, and misleading textual matches. Keep a hidden suite to discourage tuning to visible examples. Add synthetic mutations for breadth while retaining manually reviewed realistic tasks.

### 14.3 Experimental design

- Pin the repository snapshot, container image digest, CLI version, requested full model identifier, effort level, prompt, tool schema, and tool build.
- Use the same task prompt and budgets across conditions.
- Run 5–10 independent trials per task and condition.
- Randomize or interleave condition order to reduce time-of-day and subscription-limit bias.
- Use single-attempt success as the primary outcome; report pass@k only as a secondary statistic.
- Pair conditions by task and bootstrap confidence intervals over tasks.
- Record failures and rate limits rather than silently retrying until success.

A useful initial study is 30 tasks × 3 conditions × 5 repetitions = 450 trials. It is large enough to reveal substantial effects while remaining operationally understandable.

### 14.4 Metrics

Primary metrics:

- task success under hidden tests or a blinded rubric;
- success within a fixed token/call/time budget;
- input, cached input, output, and reasoning tokens as separately reported;
- unique tool calls and tool operations by type;
- wall-clock time and time to first useful evidence;
- output bytes and estimated tokens returned by each tool;
- build, refresh, and query failures.

Derived metrics:

- success per 100,000 reported tokens;
- useful oracle facts per 100 tool-output tokens;
- tool calls and operations per successful task;
- cost or quota consumption per successful task when the provider exposes it;
- p50, p95, and maximum query latency;
- reference precision/recall and definition accuracy;
- fraction of partial results that the agent interpreted correctly.

Token totals from different providers are not directly comparable because tokenizers, hidden prompts, and accounting rules differ. Use token comparisons primarily within a provider. Across providers, emphasize success, elapsed time, reported quota/cost where available, and observable operations.

### 14.5 Versioning results

Every run manifest should hash or identify:

- project snapshot;
- task and oracle;
- container image;
- model and agent CLI;
- system/repository instructions;
- tool schema and implementation commit;
- Merlin, OCaml, Dune, and index versions;
- build-cache condition;
- network policy and resource limits.

Never compare tool revisions under the same informal label. Generate a report keyed by these identities, and retain raw event logs for re-analysis.

## 15. Hermetic Codex and Claude CLI execution

### 15.1 Isolation model

Each trial should run in a fresh disposable container or VM with:

- an immutable project snapshot copied to a fixed `/workspace` path;
- a fresh user home, agent state directory, temporary directory, and process namespace;
- fixed locale and timezone;
- a pinned OCI image digest and exact tool versions;
- controlled CPU/memory/time limits;
- networking disabled unless the provider CLI requires it, and otherwise restricted to required provider endpoints where practical;
- no mounted developer home, shell history, SSH agent, IDE socket, prior agent sessions, or shared writable build directory.

For the strongest repository isolation, create the project from `git archive` or a release tarball and initialize a new single-commit repository. Do not use a worktree sharing the original `.git` directory.

Build-state conditions are part of the experiment and must be prepared explicitly: cold, warm, indexed, or deliberately stale. Never accidentally share writable `_build` state between trials.

### 15.2 Codex CLI

Codex supports non-interactive execution with JSONL events. A representative trial command is:

```sh
codex exec \
  --ephemeral \
  --ignore-user-config \
  --ignore-rules \
  --json \
  --color never \
  --model "$MODEL_ID" \
  --sandbox workspace-write \
  --cd /workspace \
  - < /eval/task.txt \
  > /results/events.jsonl \
  2> /results/stderr.log
```

Do not use `resume` or `last`. `--ephemeral` prevents session persistence; ignoring user configuration and rules prevents accidental behavioral leakage from the host. Supply the benchmark's own fixed instructions explicitly.

The event stream includes lifecycle and item records. The terminal turn record reports fields including input, cached-input, output, and reasoning-output tokens. Count command/tool invocations by unique event/item identity rather than counting both start and completion records.

### 15.3 Claude CLI

A representative subscription-authenticated trial command is:

```sh
DISABLE_AUTOUPDATER=1 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude \
  --print \
  --safe-mode \
  --no-session-persistence \
  --strict-mcp-config \
  --mcp-config /eval/mcp.json \
  --model "$MODEL_ID" \
  --effort medium \
  --max-turns 30 \
  --output-format stream-json \
  < /eval/task.txt \
  > /results/events.jsonl \
  2> /results/stderr.log
```

Do not use `resume`, `continue`, or `fork-session`. In the currently inspected CLI, `--bare` accepts only API-key or credential-helper authentication, so subscription-based runs should use `--safe-mode` with an otherwise fresh home and explicit configuration.

Claude tool calls should be counted by unique `tool_use.id`. As with Codex, also count backend operations because a single shell tool call may contain several searches or file reads.

### 15.4 Result layout

```text
runs/TASK/CONDITION/TRIAL/
  manifest.json
  events.jsonl
  stderr.log
  final.txt
  patch.diff
  metrics.json
  outcome.json
```

A separate grader container, without model credentials, applies the patch to a clean snapshot and runs hidden tests or rubric checks. This prevents the agent from seeing evaluation oracles and keeps grading failures distinct from execution failures.

### 15.5 What reproducibility means

Hosted models remain probabilistic and may change behind a provider identifier. The target is therefore statistical reproducibility, not byte-for-byte replay. A valid experiment fixes observable inputs, isolates state, preserves event streams, repeats trials, and reports distributions and uncertainty.

Cached prompt tokens are a provider optimization, not evidence that one trial can read another trial's semantic conversation. Memory leakage is controlled through session, filesystem, process, configuration, and repository isolation.

## 16. Subscription authentication without API keys

Both CLIs can be authenticated once interactively with a paid user subscription and then run non-interactively. Credentials must be treated as passwords.

### 16.1 Codex

Authenticate outside the benchmark:

```sh
codex login --device-auth -c 'cli_auth_credentials_store="file"'
codex login status
```

With file storage, the reusable credential is stored under the Codex state directory, commonly `~/.codex/auth.json`. It contains refreshable access material and must have restrictive permissions.

### 16.2 Claude

Authenticate outside the benchmark:

```sh
claude auth login --claudeai
claude auth status
```

Claude Pro/Max authentication is then reused by the CLI. The exact local storage is an implementation detail and should be discovered by the harness rather than assumed forever; in the inspected Linux installation it is under `~/.claude/.credentials.json`.

### 16.3 Injection pattern

Do not mount the full host home. Maintain a credential-only seed directory outside the repository and results tree:

1. Mount the seed read-only at `/run/credentials`.
2. At trial startup, copy only the required credential file into the fresh trial user's state directory.
3. Set owner-only permissions.
4. Run the CLI with a fresh, otherwise empty home.
5. Discard the trial copy afterward.

OAuth credentials may refresh. The simplest reproducible policy discards refreshed trial credentials and repeats interactive login when the seed expires. For long suites, serialize trials with a lock, copy the seed into one trial, validate the refreshed credential after success, atomically copy only that credential back to the seed, and release the lock. Do not launch parallel trials from independent copies of a rotating refresh token unless the provider documents that as safe.

Credential evolution is not conversational memory. It should be recorded only as a non-secret authentication generation identifier, never as token contents or a publicly comparable file hash.

Never:

- commit credentials;
- copy them into a container image layer;
- print or archive them in events, traces, manifests, or results;
- mount them inside `/workspace`;
- expose subscription credentials to untrusted repositories or prompts;
- allow unrestricted network access when it is not required.

Subscription execution is practical for local controlled experiments, but it is less suitable for untrusted or shared CI. Provider rate limits and quota windows can confound results; run at low concurrency, interleave conditions, and record throttling. For unattended organizational CI, service/API credentials are preferable when available.

## 17. Delivery plan

### Phase 0: specification and fixtures

- finalize command names, position convention, envelope, warning codes, and compact grammar;
- add a small semantic fixture project and oracles;
- capture raw golden payloads from supported Merlin versions;
- build the benchmark manifest and result schema.

Exit criterion: schemas and fixtures can distinguish correct, partial, stale, truncated, and failed results.

### Phase 1: shell MVP

- implement workspace discovery and safe argument handling;
- invoke Merlin through the active opam switch;
- support `capabilities`, `status`, `outline`, `type`, `definition`, `references`, `errors`, and `inspect`;
- add `auto|always|never` Dune refresh;
- implement compact AI and JSONL output;
- add `doctor` and structured tracing.

Exit criterion: the tool works on the fixture and `mltorch`, accurately reports the missing project index in the current OCaml 4.14 switch, and never claims false completeness.

### Phase 2: integration and evaluation

- add a typed MCP adapter;
- build hermetic Codex and Claude runners;
- instrument `rg`, file reads, builds, and semantic operations;
- run the initial 450-trial study;
- tune defaults from evidence rather than anecdote.

Exit criterion: statistically analyzed results show whether the tool or hybrid workflow improves success or reduces effort for at least the targeted task classes.

### Phase 3: persistent OCaml backend

- prototype a pinned `merlin-lib` service;
- reuse pipelines across aggregated and repeated queries;
- integrate Dune RPC/watch state where supported;
- compare correctness and latency to the shell backend on identical tests;
- retain backend selection and shell fallback.

Exit criterion: the persistent backend provides a material measured benefit that justifies its version-maintenance cost.

## 18. Decisions and open questions

### 18.1 Recommended decisions

- Treat `rg` and Merlin as complementary.
- Make `inspect` the primary high-level operation.
- Use compact tagged text for direct agent context and JSONL as canonical machine interchange.
- Let Dune determine freshness; do not use timestamp heuristics.
- Return partial results with explicit completeness when indexing fails.
- Start with a shell specification, then evaluate a persistent OCaml backend.
- Expose the same core through a typed MCP/native adapter.
- Evaluate end-to-end success, not token or call counts in isolation.
- Use fresh isolated trials and credential-only injection for subscription CLIs.

### 18.2 Open questions to resolve with prototypes

- Can a narrower Dune target than `@check` be derived reliably for cross-file locate across supported Dune versions?
- Which Merlin version combinations need normalization adapters?
- How accurately can declaration and implementation chains be recovered through wrapped/private aliases?
- Should `inspect` include references by default, or only a count plus the first few uses?
- What ranking best predicts usefulness: proximity, same module, definition/reference role, or source recency?
- Is a continuation token necessary for a stateless shell tool, or is a stable offset sufficient?
- Can Dune RPC expose index readiness portably enough to avoid filesystem discovery?
- At what query frequency does a persistent `merlin-lib` backend outperform process reuse via Merlin's server protocol?
- Which benchmark tasks are genuinely improved by semantic navigation rather than better `rg` prompting?

## 19. References

- [Merlin protocol and command-line API](https://github.com/ocaml/merlin/blob/main/doc/dev/PROTOCOL.md)
- [Merlin as a library and API stability warning](https://github.com/ocaml/merlin#merlin-as-a-library)
- [ocaml-lsp repository](https://github.com/ocaml/ocaml-lsp)
- [Dune `ocaml-index` alias](https://dune.readthedocs.io/en/stable/reference/aliases/ocaml-index.html)
- [Dune Merlin configuration query](https://github.com/ocaml/dune/blob/main/doc/usage.rst#querying-merlin-configuration)
- [Official OpenAI Codex non-interactive mode documentation](https://learn.chatgpt.com/docs/non-interactive-mode)
- [Official OpenAI Codex authentication documentation](https://learn.chatgpt.com/docs/auth)
- [Claude Code CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-usage)
- [Claude Code authentication and setup](https://docs.anthropic.com/en/docs/claude-code/getting-started)

