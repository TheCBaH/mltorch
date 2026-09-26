.PHONY: benchmark.region_compute benchmark.region_pixel build check \
	check.file-size check.int-signatures check.whitespace clean \
	expr_bench.js-benchmark expr_bench.runtest expr_order.runtest \
	expr_probe.deep-runtest expr_probe.runtest format inference inference-runa \
	inline-timing-report inline-timing-report-js js.build js.runtest \
	jsoo.build jsoo.inline-runtest jsoo.pt2.download jsoo.pt2.run \
	jsoo.pt2.runtest jsoo.pt2.vars jsoo.runtest loop.js.runtest \
	loop_js.bench loop_js.node.pt2.runtest loop_js.pt2.run melange.build \
	melange.build.scaffold melange.runtest native-infer-verify \
	native-infer-verify.% native-infer-verify-direct \
	native-infer-verify-direct.% native-transform-verify \
	native-transform-verify.% precommit profile.landmarks \
	profile.memtrace pt2.download pt2.download-all pt2.download-cram \
	pt2.json-model-support pt2.runtest pt2.vars runtest spike.runtest \
	spike.setup tailcall.js-benchmark tailcall.runtest test \
	verify.pristine visualizer.build visualizer.patch \
	visualizer.submodule webapp.bridge-runtest webapp.browser-runtest \
	webapp.build webapp.npm-install webapp.runtest webapp.serve
all: build

# Functional ATen model release, pinned alongside the producer submodule.
# PT2_MANIFEST is now a vendored, byte-for-byte copy of the producer's own
# release manifest.json (schemas/manifest.schema.json in the submodule) --
# not a hand-authored subset -- so archive URLs/digests/required members and
# retirement info all come from it instead of being re-typed here.
PT2_MANIFEST_VERSION := 2
PT2_RELEASE := v0.0.4
PT2_REPO := TheCBaH/devcontainer.pytorch-image-models
PT2_MODEL := mobilenetv2_050
PT2_DIR := data/pt2-functional
PT2_MANIFEST := data/pt2-functional-manifest.json
PT2_MODEL_DIR = $(PT2_DIR)/$(PT2_MODEL)

# Roles are explicit compatibility claims, not a mirror of the producer zoo.
PT2_MODELS_FIXTURES := mobilenetv2_050 test_convnext2
PT2_MODELS_POOL := mobilenetv2_050 csatv2
PT2_MODELS_CRAM := test_convnext2 mobilenetv2_050 regnetx_002 efficientnet_b0 fastvit_sa12 mobilenetv3_small_050
# efficientnet_b0/test_convnext2 joined once GELU/Sigmoid landed (see
# .ai/pt2_model_support.md) -- both now pass native-infer-verify and
# native-transform-verify identically to mobilenetv2_050/regnetx_002.
# fastvit_sa12 (1218 nodes) isn't in this list purely because a full run takes
# several minutes, not because of any known failure -- add it once that cost
# is judged worth paying in CI. csatv2 doesn't fully lower into Native at all
# yet (it's a deliberately graph-only fixture), so it can't join until that
# lands. mobilenetv3_small_050 joined for the native_graph_*/native_transform_*
# structural crams: it is the closest available architectural match
# to the retired mobilenet_v3_small role model (hardswish decomposition), and
# at ~8MB it is cheap next to the others.
PT2_MODELS_NATIVE_VERIFY := mobilenetv2_050 regnetx_002 efficientnet_b0 test_convnext2
# [csatv2] is selected for its real [7,7] adaptive-pool graph, not as a
# whole-model interpreter claim: it currently reaches unsupported aten.stack.
# Keep that distinction explicit so the CI runnable cohort cannot silently
# grow from a graph-only compatibility role.
PT2_MODELS_INFERENCE := $(PT2_MODELS_CRAM)
PT2_MODELS_ALL := $(PT2_MODELS_CRAM) csatv2
PT2_NATIVE_VERIFY_MODELS := $(PT2_MODELS_NATIVE_VERIFY)

# Fetch one manifest-listed release asset.  Download is the only target allowed
# to refresh assets; tests merely preflight what is already present.
pt2.download:
	scripts/pt2-download.sh $(PT2_MODEL) $(PT2_MANIFEST) $(PT2_MODEL_DIR) $(PT2_RELEASE)

# Download the models the cram tests need (cheap next to the full set: 6
# models, ~418 MB).
pt2.download-cram:
	for m in $(PT2_MODELS_CRAM); do $(MAKE) pt2.download PT2_MODEL=$$m; done

# Download every selected role model, one at a time.
pt2.download-all:
	for m in $(PT2_MODELS_ALL); do $(MAKE) pt2.download PT2_MODEL=$$m; done

# Emit the pt2 cache path/key as KEY=VALUE lines, so CI reads them from here
# (the single source of truth) instead of re-deriving them in YAML. One cache
# entry covers every model zip downloaded so far this release (glob path); the
# key folds in a hash of the model lists so adding/removing a model creates a
# fresh entry (a cache is immutable once saved under a given key) instead of
# silently never picking up the change. restore-key drops the hash, so a
# model-list change still warm-starts from the closest prior cache.
#
# Both lists are hashed, not just ALL: what CI actually populates the cache with
# is PT2_MODELS_CRAM, so a model moving into that list has to invalidate the key
# too. Hashing ALL alone would leave the pre-change cache — which lacks the new
# model — valid forever, and every run would re-download it.
PT2_MODELS_HASH := $(shell echo "$(PT2_MANIFEST_VERSION) $(PT2_REPO) $(PT2_MODELS_ALL) $(PT2_MODELS_CRAM)" \
	| md5sum | cut -c1-8)
pt2.vars:
	@echo "pt2_zip_glob=$(PT2_DIR)/*/*.zip"
	@echo "pt2_cache_key=pt2-functional-$(PT2_RELEASE)-v$(PT2_MANIFEST_VERSION)-$(PT2_MODELS_HASH)"
	@echo "pt2_cache_restore_key=pt2-functional-$(PT2_RELEASE)-v$(PT2_MANIFEST_VERSION)-"

# Run the gated end-to-end tests (load + interpreter) against the downloaded data.
#
# The two native_transform_verify crams had dune stanzas but appeared in no make
# target and no CI workflow, so nothing ran them and their goldens could drift
# unnoticed. They are the only end-to-end evidence that the symbolic verifier
# still proves what it proved, which the Native4D work is about to depend on.
#
# Deliberately does NOT depend on pt2.download-cram: that target's `unzip -o`
# always re-extracts even when the zip is already present, which would refresh
# every file's mtime (and momentarily remove it) on every runtest invocation.
# Instead this checks the data is there and fails fast with a fix-it hint --
# --auto-promote must never run against missing data, since dune would then
# bake the resulting failure straight into the committed goldens (see
# [[testing_strategy]] on the `(universe)` dep it also takes to guard against
# a stale cached failure doing the same after the data reappears).
pt2.runtest:
	@for m in $(PT2_MODELS_CRAM); do \
		test -f $(PT2_DIR)/$$m/$$m.pt2 || { \
			echo "pt2.runtest: missing $(PT2_DIR)/$$m/$$m.pt2 -- run 'make pt2.download-cram' first" >&2; \
			exit 1; \
		}; \
		for f in inputs.pt outputs.pt expected.json preprocessing.json; do \
			test -f $(PT2_DIR)/$$m/$$f || { echo "pt2.runtest: missing $(PT2_DIR)/$$m/$$f -- run 'make pt2.download-cram' first" >&2; exit 1; }; \
		done; \
	done
	@test -f $(PT2_DIR)/csatv2/csatv2.pt2 || { \
		echo "pt2.runtest: missing $(PT2_DIR)/csatv2/csatv2.pt2 -- run 'make pt2.download PT2_MODEL=csatv2' first" >&2; \
		exit 1; \
	}
	PT2_DATA=$(abspath $(PT2_DIR)) NO_COLOR=1 opam exec -- dune runtest \
		test/pt2_load_cram.t test/interp_functional_cram.t test/const_ssa_trace_cram.t \
		test/const_ssa_payload_free_cram.t test/const_ssa_evaluate_cram.t \
		test/pt2_model_support_cram.t \
		test/native_graph_regnetx_002_cram.t test/native_graph_mobilenetv2_050_cram.t \
		test/native_graph_mobilenetv3_small_050_cram.t \
		test/native_transform_regnetx_002_cram.t test/native_transform_mobilenetv2_050_cram.t \
		test/native_transform_mobilenetv3_small_050_cram.t \
		test/native_transform_verify_cram.t test/native4d_to4d_cram.t test/me_visualize_cram.t

# The payload-free counterpart of test/pt2_model_support_cram.t, over every
# model.json in the pinned producer submodule -- no download needed, so it
# covers ~100 models where the cram (real .pt2 archives, gated on PT2_DATA)
# covers 6. See bin/pt2_json_model_support.ml for the mechanism and why this
# isn't a cram test: it calls Me_export.session directly (the library
# native_graph's own `visualize` command calls), so the schema is the OCaml
# record it emits, not a shell/jq reconstruction of one.
#
# Verification is `verify.pristine`, same as every other generated/checked-in
# file in this repo: regenerate here, and CI's later `git status --porcelain`
# check catches drift. Run `make pt2.json-model-support` locally and commit
# the result when a model's frontier moves.
PT2_JSON_MODELS_DIR := modules/devcontainer.pytorch-image-models/models
PT2_JSON_MODEL_SUPPORT := test/data/pt2_json_model_support.jsonl
pt2.json-model-support:
	@mkdir -p $(dir $(PT2_JSON_MODEL_SUPPORT))
	opam exec -- dune exec bin/pt2_json_model_support.exe -- \
		$(PT2_JSON_MODELS_DIR) $(PT2_JSON_MODEL_SUPPORT)

# Shared argument list for every interp_run.exe invocation below, so
# inference-run and benchmark.inference can't drift apart.
PT2_INFER_ARGS = $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 $(PT2_MODEL_DIR)/inputs.pt \
	$(PT2_MODEL_DIR)/expected.json $(PT2_MODEL_DIR)/outputs.pt

# Run the interpreter once on $(PT2_MODEL) (downloading it first if needed)
# and print per-image inference timing to stderr. No correctness assertion -
# the cram tests already check that for the smallest model per architecture;
# this is a smoke test + benchmark for every other model. Convenience target
# for ad hoc single-model runs (e.g. `make inference-run PT2_MODEL=vit_b_16`);
# `dune exec` builds the binary as needed.
inference-run: pt2.download
	opam exec -- dune exec test/interp_run.exe -- $(PT2_INFER_ARGS)

# The raw interpreter binary, built once via plain `dune build` (never `dune
# exec`). inference.% below runs this directly instead, so `make -j` can fan
# the per-model runs out across cores: concurrent `dune exec`/`dune build`
# invocations contend on _build locks (see .ai/worktree_setup.md), but
# concurrent runs of an already-built plain binary don't.
PT2_RUN := _build/default/test/interp_run.exe
$(PT2_RUN):
	opam exec -- dune build test/interp_run.exe

# inference.<model>: run the smoke test for one specific model. PT2_MODEL is
# a pattern-specific variable, so $(PT2_INFER_ARGS) below resolves against
# the right model; the download itself still goes through a recursive
# $(MAKE) with an explicit override (pattern-specific variables don't
# propagate into a same-named shared prerequisite's own recipe the way they
# do into this target's), which is fine to run concurrently under `make -j`
# since it's plain curl/unzip, no dune involved.
inference.%: PT2_MODEL = $*
inference.%: $(PT2_RUN)
	$(MAKE) pt2.download PT2_MODEL=$*
	$(PT2_RUN) $(PT2_INFER_ARGS)

# Run inference only for the interpreter-validated cohort, not every selected
# graph role. Every inference.<model> is independent (own dir, own download,
# own process), so this is safe to run with `-j` (e.g. `make -j$(nproc)
# inference`) to cut wall time.
inference: $(addprefix inference., $(PT2_MODELS_INFERENCE))

# Keep ATen out of native_graph: this target first writes a reference with the
# existing Interp.run evaluator, then asks the pure native executable to read
# and compare it.  The reference is temporary, never a committed golden file.
ATEN_GRAPH_REF := _build/default/bin/aten_graph_ref.exe
NATIVE_GRAPH := _build/default/bin/native_graph.exe
$(ATEN_GRAPH_REF) $(NATIVE_GRAPH):
	opam exec -- dune build bin/aten_graph_ref.exe bin/native_graph.exe

# CANONICAL is the default form this checks (2026-09-23, matching the JS
# side's own default -- js/jsoo/loop_js_pt2/loop_js_pt2.ml): evaluates the
# FOLDED (Pipeline.canonical ~fold:true, via `transform --fold`) graph and
# compares its output to the real ATen reference. native-transform-verify
# already checks canonical against the untransformed graph, numerically
# and structurally, for the same PT2_NATIVE_VERIFY_MODELS -- this instead
# anchors canonical's own output to ground truth, not just to the
# untransformed graph. native-infer-verify-direct below keeps the raw,
# untransformed graph checked against ATen too, but only for the smaller
# PT2_MODELS_NATIVE_VERIFY_DIRECT subset: paying the ATen-reference cost
# on both forms for every model would be redundant with the
# canonical-vs-untransformed check native-transform-verify already does.
native-infer-verify.%: PT2_MODEL = $*
native-infer-verify.%: $(ATEN_GRAPH_REF) $(NATIVE_GRAPH)
	test -f $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 || $(MAKE) pt2.download PT2_MODEL=$*
	set -eux; ref=$$(mktemp); trap 'rm -f "$$ref"' EXIT; \
	$(ATEN_GRAPH_REF) $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 $(PT2_MODEL_DIR)/inputs.pt "$$ref"; \
	$(NATIVE_GRAPH) transform --fold --pt2 $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 \
	  --input $(PT2_MODEL_DIR)/inputs.pt --expect "$$ref"

native-infer-verify: $(addprefix native-infer-verify., $(PT2_NATIVE_VERIFY_MODELS))

# The smaller subset that still verifies the DIRECTLY converted (raw,
# untransformed) native graph against the ATen reference, not only its
# canonicalized form -- see native-infer-verify's own doc above.
# mobilenetv2_050 alone carries this job, matching the JS side's own
# smaller-subset choice (loop_js.pt2.run --direct on fastvit_sa12).
PT2_MODELS_NATIVE_VERIFY_DIRECT := mobilenetv2_050

native-infer-verify-direct.%: PT2_MODEL = $*
native-infer-verify-direct.%: $(ATEN_GRAPH_REF) $(NATIVE_GRAPH)
	test -f $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 || $(MAKE) pt2.download PT2_MODEL=$*
	set -eux; ref=$$(mktemp); trap 'rm -f "$$ref"' EXIT; \
	$(ATEN_GRAPH_REF) $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 $(PT2_MODEL_DIR)/inputs.pt "$$ref"; \
	$(NATIVE_GRAPH) eval --pt2 $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 --input $(PT2_MODEL_DIR)/inputs.pt --expect "$$ref" --verbose

native-infer-verify-direct: $(addprefix native-infer-verify-direct., $(PT2_MODELS_NATIVE_VERIFY_DIRECT))

# Execute a TRANSFORMED graph and check it against the untransformed one. This
# is a make rule and not a cram golden for two reasons: a full inference is far
# too slow for the test suite, and the residual it reports is floating point, so
# pinning it as text would make the suite platform-dependent. What the cram
# pins instead is the transformed graph's STRUCTURE, which is exact.
#
# Both pipelines are checked, because they make different claims. The structural
# one claims only Identical and must agree bit-for-bit; --fold adds batch-norm
# folding, whose single Equivalent claim says "equal in exact arithmetic, rounds
# differently", so it is held to --atol instead. See
# .ai/native_transform_design.md §3.
native-transform-verify.%: PT2_MODEL = $*
native-transform-verify.%: $(NATIVE_GRAPH)
	test -f $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 || $(MAKE) pt2.download PT2_MODEL=$*
	set -eux; \
	$(NATIVE_GRAPH) transform --verify \
		--pt2 $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 \
		--input $(PT2_MODEL_DIR)/inputs.pt; \
	$(NATIVE_GRAPH) transform --fold --verify \
		--pt2 $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 \
		--input $(PT2_MODEL_DIR)/inputs.pt; \
	$(NATIVE_GRAPH) transform --fold --verify-symbolic standard \
		--pt2 $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2

native-transform-verify: $(addprefix native-transform-verify., $(PT2_NATIVE_VERIFY_MODELS))

# --- benchmark.*: permanent latency benchmarks, each isolating one component ---

# Wall/user/sys time + max RSS for one run of the smallest model (PT2_MODEL
# defaults to resnet18), via GNU time. Gives a fast latency signal on every
# push without waiting on the full `make inference` sweep across all 22
# models (20+ minutes).
benchmark.inference: pt2.download
	/usr/bin/time -v opam exec -- dune exec test/interp_run.exe -- \
		$(PT2_INFER_ARGS) --cram >/dev/null

# The pure-OCaml native engine's conv2d, on resnet18's first (smallest) real
# conv node - currently ~27s, which is why conv2d is excluded from the full
# ATen-vs-native comparison sweep (test/pt2_node_bridge_cram.t) and from
# pt2_op_native_walk_cram.t's real-scale walk. See .ai/pt2_inference_perf.md
# for the investigation this tracks regressions/improvements against.
# aten_spec_verify's default mode runs both ATen and native and compares
# them; ATen's share of the time is negligible here (ms, not s).
benchmark.native_conv2d:
	/usr/bin/time -v opam exec -- dune exec bin/aten_spec_verify.exe -- \
		test/data/resnet18/000_convolution_default.json >/dev/null

# Pixel hot-path baseline for the Region Foundation work. The executable warms
# each case, then prints 20-sample median wall time, GC words per output cell,
# and the exact fixed input-load/output-cell counts for an identity and add
# Kernel. Keep it outside normal validation: timing belongs to a controlled
# before/after comparison, not CI noise.
benchmark.region_pixel:
	opam exec -- dune exec bin/region_pixel_bench.exe

benchmark.region_compute:
	opam exec -- dune exec bin/region_compute_bench.exe

# --- profile.*: on-demand profiling of the native conv2d benchmark ---
#
# By default both instrument the same fixture benchmark.native_conv2d
# measures, so a profile run's hotspots map directly onto that number.
# Neither is part of `make build`/`runtest`/`benchmark.*` themselves: see
# bin/aten_spec_verify.ml and lib/native/dune for why each is safe to leave
# permanently available without costing the uninstrumented path anything
# (memtrace/landmarks linked in but idle) or, for landmarks, why it needs a
# separate build profile at all (measured ~2.3x slower even idle once
# `--auto`-instrumented — see .ai/pt2_inference_perf.md's landmarks
# section).
PROFILE_DIR := .profile
PROFILE_FIXTURE := test/data/resnet18/000_convolution_default.json

# Allocation profile: MEMTRACE self-gates at runtime (Memtrace.
# trace_if_requested), so this runs the same binary as benchmark.
# native_conv2d, no separate build. View with memtrace-viewer/
# memtrace-hotspot (see the memtrace skill).
profile.memtrace:
	mkdir -p $(PROFILE_DIR)
	MEMTRACE=$(PROFILE_DIR)/native_conv2d.ctf opam exec -- dune exec bin/aten_spec_verify.exe -- \
		$(PROFILE_FIXTURE) >/dev/null
	@echo "memtrace: wrote $(PROFILE_DIR)/native_conv2d.ctf"

# CPU-time/call-count profile: needs lib/native rebuilt under --profile
# landmarks (landmarks-ppx --auto); LANDMARKS=1 then starts profiling at
# aten_spec_verify's entry. Landmark's own at_exit hook prints the callgraph
# (a text table, by call count and cycles) to stderr once the run finishes —
# nothing to redirect here beyond the >/dev/null on stdout below. Unlike
# memtrace (which only samples allocations, so cost is independent of
# call count), *active* landmarks tracking times every single instrumented
# call/exit — on PROFILE_FIXTURE's default (802,816 output pixels, ~1.83B
# inner calls) that's minutes, not seconds (measured: killed after 2.5+ min
# still climbing). CI overrides PROFILE_FIXTURE to a tiny synthetic conv
# (test/data/profile/conv2d_small.json, same code path, ~0.05s) instead;
# override PROFILE_FIXTURE=... locally for anything in between.
profile.landmarks:
	LANDMARKS=1 opam exec -- dune exec --profile landmarks bin/aten_spec_verify.exe -- \
		$(PROFILE_FIXTURE) >/dev/null

build:
	opam exec -- dune build $(BUILD_OPTIONS)

format:
	opam exec -- dune fmt

# File-size regression check: no tracked .ml/.mli file over 1000 lines (tree
# check) and no file added by HEAD over 750 lines (new-file check), unless
# listed in scripts/file-size-exceptions.txt.
check.file-size:
	bash scripts/check-file-size.sh

# Signature ratchet for domain-typed integers: every declaration in an in-scope
# .mli that mentions int must be allowlisted with a category, and the list only
# shrinks (tools/int_signatures). Also part of runtest.
check.int-signatures:
	opam exec -- dune build @tools/int_signatures/runtest

# Whitespace/conflict-marker regression check: git's own diff-hygiene check
# against HEAD (trailing whitespace, a mixed tab/space indent the change
# introduced, an unresolved conflict marker). [check] below already runs
# this inline; named here so [precommit] can depend on it directly too.
check.whitespace:
	git diff --check HEAD

verify.pristine:
	@git_status=$$(git status --porcelain); \
	if [ -n "$$git_status" ]; then \
		echo "Git directory is not pristine. Modified or created files:"; \
		echo "$$git_status"; \
		exit 1; \
	fi

test:
	opam exec -- dune test

# NO_COLOR: --auto-promote bakes whatever a test prints straight into the
# committed golden. cmdliner and jsont both style their error text with raw
# ANSI escapes whenever $TERM looks color-capable, invisible in a terminal or
# in dune's own diff rendering but real bytes -- and byte-for-byte wrong
# against a headless CI run. See .ai/testing_strategy.md's "Promoting from an
# interactive shell" note; NO_COLOR forces the plain (uncolored) renderer so a
# promotion made from an interactive shell can't disagree with one made on a
# runner.
runtest:
	NO_COLOR=1 opam exec -- dune runtest --auto-promote

# Per-partition inline-test timing report + regression gate, over every
# (inline_tests) library in the project -- see
# scripts/inline-test-timing-report-all.sh and
# _ai_/lstm_scale_test_timing_notes.md. Split into best/js like
# runtest/js.runtest: js needs node and is heavier to build.
#
# Separate thresholds: js_of_ocaml is slower per-op than native (measured on
# lstm_scale_test.ml's shrunk fixture -- 6.7s native, 36s under node), so a
# shared threshold would either be too loose natively or too tight on js.
INLINE_TIMING_THRESHOLD_SECONDS := 15
INLINE_TIMING_THRESHOLD_SECONDS_JS := 60

inline-timing-report:
	opam exec -- scripts/inline-test-timing-report-all.sh best $(INLINE_TIMING_THRESHOLD_SECONDS)

inline-timing-report-js:
	opam exec -- scripts/inline-test-timing-report-all.sh js $(INLINE_TIMING_THRESHOLD_SECONDS_JS)

# Standalone recursion experiment.  These are committed backend-specific
# expectations rather than a native-vs-JS diff: the point is precisely that
# the compilers support different recursive call shapes.
TAILCALL_BUILD := _build/default/experiments/tailcall

tailcall.runtest:
	opam exec -- dune build --profile melange \
	  experiments/tailcall/jsoo/tailcall_probe.bc.js \
	  experiments/tailcall/jsoo_cps/tailcall_probe.bc.js \
	  @tailcall-melange
	@node $(TAILCALL_BUILD)/jsoo/tailcall_probe.bc.js 200000 \
	  > $(TAILCALL_BUILD)/actual.jsoo
	@diff -u experiments/tailcall/expected.jsoo $(TAILCALL_BUILD)/actual.jsoo
	@node $(TAILCALL_BUILD)/jsoo_cps/tailcall_probe.bc.js 200000 \
	  > $(TAILCALL_BUILD)/actual.jsoo-cps
	@diff -u experiments/tailcall/expected.jsoo-cps \
	  $(TAILCALL_BUILD)/actual.jsoo-cps
	@node $(TAILCALL_BUILD)/melange/output/experiments/tailcall/melange/tailcall_probe.js \
	  200000 > $(TAILCALL_BUILD)/actual.melange
	@diff -u experiments/tailcall/expected.melange \
	  $(TAILCALL_BUILD)/actual.melange
	@echo "tailcall: jsoo, jsoo CPS, and melange match their stack-safety contracts"

tailcall.js-benchmark:
	opam exec -- dune build --profile melange \
	  experiments/tailcall/jsoo_bench/tailcall_js_bench.bc.js \
	  @tailcall-bench-melange
	@echo "js_of_ocaml"
	node $(TAILCALL_BUILD)/jsoo_bench/tailcall_js_bench.bc.js 64 200000
	@echo "melange"
	node $(TAILCALL_BUILD)/melange_bench/output/experiments/tailcall/melange_bench/tailcall_js_bench.js \
	  64 200000

# Stage 0 of the Expr tail-call conversion (.ai/'s implementation plan): the
# evaluation-order oracle. Native, bytecode and jsoo build one shared source,
# js/probe/order_probe.ml, and each route's output is diffed against its own
# committed golden -- there is no cross-backend diff here, unlike
# jsoo.runtest, because the whole point is to record where backends are
# ALLOWED to differ (unspecified evaluation order), not to require agreement.
ORDER_PROBE_BUILD := _build/default/js

expr_order.runtest:
	opam exec -- dune build --profile melange \
	  js/probe/order_probe.exe js/probe/order_probe.bc \
	  js/jsoo/order_probe/order_probe.bc.js \
	  @expr-order-melange
	@$(ORDER_PROBE_BUILD)/probe/order_probe.exe \
	  > $(ORDER_PROBE_BUILD)/probe/order_probe.actual.native
	@diff -u js/probe/order_probe.expected.native \
	  $(ORDER_PROBE_BUILD)/probe/order_probe.actual.native
	@$(ORDER_PROBE_BUILD)/probe/order_probe.bc \
	  > $(ORDER_PROBE_BUILD)/probe/order_probe.actual.bytecode
	@diff -u js/probe/order_probe.expected.bytecode \
	  $(ORDER_PROBE_BUILD)/probe/order_probe.actual.bytecode
	@node $(ORDER_PROBE_BUILD)/jsoo/order_probe/order_probe.bc.js \
	  > $(ORDER_PROBE_BUILD)/jsoo/order_probe/order_probe.actual.jsoo
	@diff -u js/probe/order_probe.expected.jsoo \
	  $(ORDER_PROBE_BUILD)/jsoo/order_probe/order_probe.actual.jsoo
	@node $(ORDER_PROBE_BUILD)/melange/order_probe/output/js/melange/order_probe/order_probe.js \
	  > $(ORDER_PROBE_BUILD)/melange/order_probe/order_probe.actual.melange
	@diff -u js/probe/order_probe.expected.melange \
	  $(ORDER_PROBE_BUILD)/melange/order_probe/order_probe.actual.melange
	@echo "expr_order: native, bytecode, jsoo and melange each match their own golden"

# The tail-call conversion's plain-value probe (Stage 3 onward, .ai/'s
# implementation plan): unlike expr_order.runtest above, this one checks
# cross-backend AGREEMENT (native is the golden, jsoo and Melange are each
# diffed against it), because none of probe_expr.ml's cases are
# order-sensitive -- that is exactly what order_probe.ml exists to isolate.
EXPR_PROBE_BUILD := _build/default/js

expr_probe.runtest:
	opam exec -- dune build --profile melange \
	  js/probe/probe_expr.exe \
	  js/jsoo/probe_expr/probe_expr.bc.js \
	  @expr-probe-melange
	@$(EXPR_PROBE_BUILD)/probe/probe_expr.exe \
	  > $(EXPR_PROBE_BUILD)/probe/probe_expr.actual.native
	@node $(EXPR_PROBE_BUILD)/jsoo/probe_expr/probe_expr.bc.js \
	  > $(EXPR_PROBE_BUILD)/jsoo/probe_expr/probe_expr.actual.jsoo
	@diff -u $(EXPR_PROBE_BUILD)/probe/probe_expr.actual.native \
	  $(EXPR_PROBE_BUILD)/jsoo/probe_expr/probe_expr.actual.jsoo
	@node $(EXPR_PROBE_BUILD)/melange/probe_expr/output/js/melange/probe_expr/probe_expr.js \
	  > $(EXPR_PROBE_BUILD)/melange/probe_expr/probe_expr.actual.melange
	@diff -u $(EXPR_PROBE_BUILD)/probe/probe_expr.actual.native \
	  $(EXPR_PROBE_BUILD)/melange/probe_expr/probe_expr.actual.melange
	@echo "expr_probe: jsoo and melange match native"

# Stage 7's own deep-isolated-Expr gate: probe_expr.ml's [--deep] mode, run
# only against jsoo and Melange -- never native, whose [Eval.value] is still
# ordinary recursion and would genuinely overflow on the depth-200,000
# Value/Bool cases this checks against closed forms. Each route's own exit
# code is the gate (probe_expr.ml exits nonzero on any mismatch), not a diff
# against a golden: unlike expr_probe.runtest above, there is no native run
# to diff against here.
expr_probe.deep-runtest:
	opam exec -- dune build --profile melange \
	  js/jsoo/probe_expr/probe_expr.bc.js \
	  @expr-probe-melange
	node $(EXPR_PROBE_BUILD)/jsoo/probe_expr/probe_expr.bc.js --deep
	node $(EXPR_PROBE_BUILD)/melange/probe_expr/output/js/melange/probe_expr/probe_expr.js --deep
	@echo "expr_probe.deep: jsoo and melange both survive and check out at depth 200000"

# The tail-call conversion's Stage 5 correctness harness (see
# test/expr_bench/corpus.ml and .ai/): every candidate in
# Corpus.candidate_evaluators against the reference evaluator, on every
# applicable case, plus one deep smoke case the reference evaluator is never
# asked to run. Each route's own exit code is the gate -- expr_bench_run.ml
# exits nonzero on any mismatch.
EXPR_BENCH_BUILD := _build/default/test/expr_bench

expr_bench.runtest:
	opam exec -- dune build --profile melange \
	  test/expr_bench/jsoo/expr_bench_run.bc.js \
	  @expr-bench-melange
	node $(EXPR_BENCH_BUILD)/jsoo/expr_bench_run.bc.js
	node $(EXPR_BENCH_BUILD)/melange/output/test/expr_bench/melange/expr_bench_run.js

# Same two routes, [--bench] mode: Sys.time over the shallow corpus plus a
# couple of above-frontier deep_chain depths, reference included on the
# corpus but candidates only on the deep depths (see expr_bench_run.ml).
# Manual, not part of runtest/js.runtest -- a timing report, not a gate.
expr_bench.js-benchmark:
	opam exec -- dune build --profile melange \
	  test/expr_bench/jsoo/expr_bench_run.bc.js \
	  @expr-bench-melange
	@echo "js_of_ocaml"
	node $(EXPR_BENCH_BUILD)/jsoo/expr_bench_run.bc.js --bench
	@echo "melange"
	node $(EXPR_BENCH_BUILD)/melange/output/test/expr_bench/melange/expr_bench_run.js --bench

# JavaScript backends. Deliberately outside `runtest`, same reasoning as
# pt2.runtest: linking js_of_ocaml on every local test run is not worth it, and
# these need node. See .ai/js_backends_design.md.
#
# The comparison is always between two runs of ONE program: native_probe.exe
# against node's native_probe.bc.js. There is no golden file to promote -- the
# native binary IS the golden, so the two cannot drift apart independently.

JS_BUILD := _build/default/js

# The tier-1 fixture, copied out of the models submodule by a rule in
# js/probe/dune. Both sides of the diff are handed the same path.
JS_MODEL_JSON := $(JS_BUILD)/probe/model.json

jsoo.build:
	opam exec -- dune build js/probe js/jsoo js/run

jsoo.runtest: jsoo.build
	@$(JS_BUILD)/probe/native_probe.exe $(JS_MODEL_JSON) > $(JS_BUILD)/native.txt
	@node $(JS_BUILD)/jsoo/native_probe.bc.js $(JS_MODEL_JSON) > $(JS_BUILD)/jsoo.txt
	@diff -u $(JS_BUILD)/native.txt $(JS_BUILD)/jsoo.txt \
	  && echo "jsoo: output matches native ($$(wc -l < $(JS_BUILD)/native.txt) lines)"

# Inline expect tests under node. Complementary to the probe above, not a
# duplicate of it: the probe diffs two runs of one program, so it answers "do
# the backends agree" but never "is the answer right", while these carry the
# same committed goldens the native suites do. The runtest-js alias comes from
# the root dune -- without it these would attach to `runtest` and every
# `make runtest` would start linking js_of_ocaml.

jsoo.inline-runtest:
	NO_COLOR=1 opam exec -- dune build @runtest-js

# Tier 2: open a real .pt2, lower it, run inference, diff native against node.
# Outside js.runtest for the same reason as pt2.runtest -- it needs downloaded
# weights. Guards BOTH inputs, because a partial extraction or a half-restored
# cache would otherwise sail past the check and fail later as a bare filesystem
# error, losing the fix-it hint that is the whole point of checking up front.
#
# mobilenet_v3_small rather than the resnet18 used elsewhere: measured natively
# it is 10s and 99MB against resnet18's 86s and 241MB
# (.ai/native_inference_verify.md), and node multiplies whatever the native cost
# is -- ~4.9x when this was written. Never point this at a PT2_MODELS_VIT entry
# larger than vit_b_*: vit_l_16/vit_l_32 are ~1.15GB each, and nothing has
# measured node/jsoo actually handling a Pt2_archive read anywhere near
# Pt2_archive.max_file_bytes -- see lib/pt2/pt2_archive.ml.

JS_PT2_MODEL := mobilenetv2_050
JS_PT2_DIR := $(PT2_DIR)/$(JS_PT2_MODEL)
JS_PT2_ARCHIVE := $(JS_PT2_DIR)/$(JS_PT2_MODEL).pt2
JS_PT2_INPUT := $(JS_PT2_DIR)/inputs.pt

jsoo.pt2.download:
	$(MAKE) pt2.download PT2_MODEL=$(JS_PT2_MODEL)

# Cache vars for the javascript workflow, deliberately a DIFFERENT key and a
# DIFFERENT glob from pt2.vars. That one covers every model CI has downloaded
# this release; this one covers a single model. Sharing the key would be a
# silent, permanent regression rather than a failure: caches are immutable once
# saved, so whichever workflow reached a cold key first would save its own
# contents there, and a one-model archive stored under the all-models key makes
# every later save a no-op and every build.yml run re-download the rest.
#
# No model-list hash here, unlike PT2_MODELS_HASH: the key names the one model
# it holds, so changing JS_PT2_MODEL already changes the key.
jsoo.pt2.vars:
	@echo "jsoo_pt2_zip_glob=$(PT2_DIR)/$(JS_PT2_MODEL)/*.zip"
	@echo "jsoo_pt2_cache_key=pt2-jsoo-functional-$(PT2_RELEASE)-v$(PT2_MANIFEST_VERSION)-$(JS_PT2_MODEL)"
	@echo "jsoo_pt2_cache_restore_key=pt2-jsoo-functional-$(PT2_RELEASE)-v$(PT2_MANIFEST_VERSION)-"

jsoo.pt2.runtest: jsoo.build
	@for f in $(JS_PT2_ARCHIVE) $(JS_PT2_INPUT); do \
		test -f $$f || { \
			echo "jsoo.pt2.runtest: missing $$f -- run 'make pt2.download PT2_MODEL=$(JS_PT2_MODEL)' first" >&2; \
			exit 1; \
		}; \
	done
	@$(JS_BUILD)/probe/pt2_probe.exe $(JS_PT2_ARCHIVE) $(JS_PT2_INPUT) \
	  > $(JS_BUILD)/pt2_native.txt
	@node $(JS_BUILD)/jsoo/pt2_probe.bc.js $(JS_PT2_ARCHIVE) $(JS_PT2_INPUT) \
	  > $(JS_BUILD)/pt2_jsoo.txt
	@diff -u $(JS_BUILD)/pt2_native.txt $(JS_BUILD)/pt2_jsoo.txt \
	  && echo "jsoo: $(JS_PT2_MODEL) inference matches native ($$(wc -l < $(JS_BUILD)/pt2_native.txt) lines)"

# Every sample of the same model through the PURE engine under node, checked
# against the rankings shipped in the release. A third thing, not a bigger
# jsoo.pt2.runtest: that target diffs the two backends against each other and
# would pass just as happily if both were wrong, while this one has an external
# reference and no native counterpart in the comparison.
#
# --strict is what makes it a gate: without it a top-5 disagreement only shows
# up in the printed report and the command still exits 0, which is the contract
# the interp_*_cram.t goldens rely on and is deliberately left alone.
#
# MANUAL. Not in runtest, js.runtest, or jsoo.pt2.runtest: ten inferences at the
# ~4.9x node multiplier is minutes, against ~45s for the one-image differential.
# Depending on the download IS safe here, unlike pt2.runtest -- there is no cram
# and no --auto-promote, so unzip -o refreshing mtimes costs nothing.

JS_PT2_RUN_ARGS = $(JS_PT2_ARCHIVE) $(JS_PT2_DIR)/inputs.pt \
	$(JS_PT2_DIR)/expected.json $(JS_PT2_DIR)/outputs.pt

jsoo.pt2.run: jsoo.pt2.download jsoo.build
	node $(JS_BUILD)/jsoo/pt2_run.bc.js $(JS_PT2_RUN_ARGS) --strict

# Every node of JS_PT2_MODEL (mobilenetv2_050 -- no Region-authored node, so
# only --nodes exercises anything new on it, design §4.5) through its own
# generated-JS kernel, checked bitwise against the interpreter (--shadow) and
# against the release's own ranking (--strict). Whole-graph-js-compile plan
# T5.2/T5.5. Reuses JS_PT2_MODEL's own already-downloaded/cached archive
# (jsoo.pt2.download, the SAME cache key jsoo.pt2.runtest uses) -- no new
# download plumbing. Tier 2: ~100s under node, the same order as
# jsoo.pt2.runtest's own step in the same CI job.
#
# Runs canonical (loop_js_pt2's default, see its own doc comment) -- the
# raw, directly-converted graph is exercised by loop_js.pt2.run's --direct
# instead, so this doesn't pay for both on every push.
loop_js.node.pt2.runtest: jsoo.pt2.download jsoo.build
	node $(JS_BUILD)/jsoo/loop_js_pt2/loop_js_pt2.bc.js $(JS_PT2_RUN_ARGS) \
	  --nodes --shadow --strict

# The whole-model verification through GENERATED JAVASCRIPT (not just the
# reference path jsoo.pt2.run/jsoo.pt2.runtest exercise): a real model's
# Region-authored nodes (RmsNorm/LayerNorm/Softmax/Sdpa/Lstm) run through
# Loop_js_exec-compiled JavaScript, in-process, checked against the
# release's own results.json.
#
# fastvit_sa12, not JS_PT2_MODEL: it is the smallest model in PT2_MODELS_CRAM
# with an SDPA node (a pure-CNN model like mobilenetv2_050 has ZERO
# Region-authored nodes and would exercise nothing new). Already downloaded
# by pt2.download-cram, so this needs no separate download plumbing of its
# own -- if it is missing, `make pt2.download-cram` fetches it same as every
# other PT2_MODELS_CRAM entry.
#
# ONE sample only ([Infer_report]'s own [~max_samples]), and MANUAL --
# further from CI than jsoo.pt2.run itself: one fastvit_sa12 sample measured
# at ~103s through the native evaluator, so even a single sample under node
# is minutes, and results.json holds ten.
#
# --direct: canonical (Pipeline.canonical's output) is loop_js_pt2's DEFAULT
# as of 2026-09-23 -- this target opts back into the raw, untransformed
# graph so the direct PT2-to-Native conversion path stays exercised
# end-to-end on at least one model, not only its canonicalized result.
# fastvit_sa12 alone carries that job; loop_js.node.pt2.runtest below (and
# every other PT2_MODELS_CRAM model, verified manually) runs canonical.
LOOP_JS_PT2_MODEL := fastvit_sa12
LOOP_JS_PT2_DIR := $(PT2_DIR)/$(LOOP_JS_PT2_MODEL)
LOOP_JS_PT2_RUN_ARGS = $(LOOP_JS_PT2_DIR)/$(LOOP_JS_PT2_MODEL).pt2 \
	$(LOOP_JS_PT2_DIR)/inputs.pt $(LOOP_JS_PT2_DIR)/expected.json \
	$(LOOP_JS_PT2_DIR)/outputs.pt

loop_js.pt2.run: jsoo.build
	@test -f $(LOOP_JS_PT2_DIR)/$(LOOP_JS_PT2_MODEL).pt2 || \
	  $(MAKE) pt2.download PT2_MODEL=$(LOOP_JS_PT2_MODEL)
	node $(JS_BUILD)/jsoo/loop_js_pt2/loop_js_pt2.bc.js \
	  $(LOOP_JS_PT2_RUN_ARGS) --strict --direct

# Melange lives behind `--profile melange` so that `dune build` and `make build`
# never compile it -- a melange.emit stanza is otherwise attached to @all and
# would drag the whole JS toolchain into every ordinary build.
#
# The melange probe covers only the pure half of the engine (walk_core + core);
# Bigarray and Jsont_bytesrw are out of reach, so it is diffed against
# subset_probe.exe rather than native_probe.exe. Both sides are still one
# program built two ways.
#
# melange.build.scaffold is the floor: shim + fmt + jsont_base, no probe. It is
# what to run when diagnosing whether a melange failure is in the vendored
# libraries or in what sits on top of them.

MEL_OUT := _build/default/js/melange/output/js/melange

melange.build:
	opam exec -- dune build --profile melange @melange

melange.build.scaffold:
	opam exec -- dune build --profile melange @melange-scaffold

melange.runtest: melange.build
	@opam exec -- dune build js/probe/subset_probe.exe
	@_build/default/js/probe/subset_probe.exe > _build/default/js/subset_native.txt
	@node $(MEL_OUT)/subset_probe.js > _build/default/js/subset_melange.txt
	@diff -u _build/default/js/subset_native.txt _build/default/js/subset_melange.txt \
	  && echo "melange: output matches native ($$(wc -l < _build/default/js/subset_native.txt) lines)"

# The build-side aggregate, mirroring js.runtest. Composed of the two existing
# build targets rather than a bare `dune build js`, which would defeat the
# profile gate that keeps melange.emit off @all.
js.build: jsoo.build melange.build

# The Loop IR's generated-code gate: the emitted JavaScript, run under node,
# must print what the Loop interpreter says the same program does. Needs node,
# so it is not part of `runtest`; a dune alias of its own (not `runtest-js`,
# which is the inline-test alias) keeps the two selectable separately.
loop.js.runtest:
	NO_COLOR=1 opam exec -- dune build @test/loop_ir/loop-js-gate

js.runtest: jsoo.runtest jsoo.inline-runtest melange.runtest loop.js.runtest

# Compares Loop_interp.run's wall-clock cost natively (ocamlopt) against
# itself and the generated-code executor under node (jsoo), on the same
# elementwise kernel (test/loop_ir/loop_bench_program.ml) -- the tradeoff
# the Loop IR JavaScript backend design doc's Outcome leaves unmeasured
# (only emitted code size was). Manual, not part of runtest/js.runtest -- a
# timing report, not a gate.
loop_js.bench:
	opam exec -- dune build test/loop_ir/loop_bench_native.exe \
	  js/loop_js_exec/bench/loop_bench_jsoo.bc.js
	_build/default/test/loop_ir/loop_bench_native.exe
	node _build/default/js/loop_js_exec/bench/loop_bench_jsoo.bc.js

clean:
	opam exec -- dune clean
	git submodule foreach --recursive git clean -xdf

# --- the Model Explorer visualizer bundle -----------------------------------
#
# Built from the submodule that also pins the OCaml schema, so ONE commit fixes
# both halves of the integration. There is no published release that can: npm's
# `ai-edge-model-explorer-visualizer` is stuck at 0.1.2 (2025-06-23) while
# `Graph.tasksData`, `EdgeOverlaysData.graphName`, `EdgeOverlay.visibleEdgeHops`
# and `SyncNavigationData.showDiffHighlights` all landed upstream after it --
# every one of them emitted by `lib/model_explorer_export` and, against 0.1.2,
# silently dropped. `tasksData` was the visible cost: the fusion overlay on
# `g/kernel/000` rendered nothing at all.
#
# `scripts/build_npm.sh` is upstream's own release script and emits exactly the
# `dist/` layout the npm package had, so consumers below changed only their
# source path. It builds `model_explorer` too, and must: `worker_service.ts`
# constructs its worker from a runtime string rather than `new URL(...)`, so the
# `custom_element` target emits no worker chunk of its own.
#
# See `.ai/model_explorer_design.md` and `patches/` for why the patch exists.

MODEL_EXPLORER   := vendored/ocaml-model-explorer/model-explorer
VISUALIZER_SRC   := $(MODEL_EXPLORER)/src/ui
VISUALIZER_DIST  := $(VISUALIZER_SRC)/custom_element_npm/dist
VISUALIZER_PATCH := $(abspath patches/model-explorer-custom-element-worker.patch)

# CI checks submodules out top-level only, by deliberate policy (build.yml), so
# the nested one is named here the same way pytorch's two are.
#
# Guarded on the checkout already existing rather than run unconditionally: the
# CI jobs initialise it on the RUNNER, before the cache step that needs its
# commit for a key, and this target then runs again inside the devcontainer
# against the same bind-mounted tree. Re-running `git submodule update` across
# that boundary is work at best and a `dubious ownership` failure at worst.
visualizer.submodule:
	@test -f $(VISUALIZER_SRC)/package.json || { \
		git submodule update --init --depth 1 vendored/ocaml-model-explorer && \
		git -C vendored/ocaml-model-explorer submodule update --init --depth 1 model-explorer; }

# Idempotent, and it REFUSES rather than skips. A tree that silently went
# unpatched fails later, in an Angular type error that says nothing about the
# submodule having moved.
visualizer.patch: visualizer.submodule
	@cd $(MODEL_EXPLORER) && \
	if git apply --reverse --check $(VISUALIZER_PATCH) 2>/dev/null; then \
		echo "visualizer: patch already applied"; \
	elif git apply --check $(VISUALIZER_PATCH) 2>/dev/null; then \
		git apply $(VISUALIZER_PATCH) && echo "visualizer: patch applied"; \
	else \
		echo "visualizer: $(VISUALIZER_PATCH)" >&2; \
		echo "  no longer applies at $$(git rev-parse --short HEAD)." >&2; \
		echo "  Re-derive it against the new custom_element stand-in, or drop it" >&2; \
		echo "  if upstream has fixed the target. See patches/ for the rationale." >&2; \
		exit 1; \
	fi

# File-targeted, with EVERY step inside the recipe, so an existing bundle costs
# nothing at all: no npm ci, no patch, no Angular build. That is what makes the
# CI cache worth having -- keyed on the submodule commit, a hit skips a
# 1191-package install and two Angular builds outright. A prerequisite on
# node_modules would defeat it, since a cached bundle with no node_modules would
# reinstall and then rebuild against the newer timestamp.
#
# There is no dep-tracking on the submodule source, the same caveat `data/dune`
# carries: after bumping the submodule, `rm -rf $(VISUALIZER_DIST)` to rebuild.
$(VISUALIZER_DIST)/main_browser.js:
	$(MAKE) visualizer.patch
	cd $(VISUALIZER_SRC) && npm ci --no-audit --no-fund
	cd $(VISUALIZER_SRC) && PATH="$$PWD/node_modules/.bin:$$PATH" ./scripts/build_npm.sh

visualizer.build: $(VISUALIZER_DIST)/main_browser.js

# --- Stage 0A contract spike -----------------------------------------------
#
# Drives the real Model Explorer element to settle the questions the browser
# coordinator design assumes. Needs node, npm and a downloaded Chromium, so it
# is outside every other target; its ANSWERS are recorded in
# .ai/model_explorer_design.md and do not need re-deriving to read.

spike.setup: visualizer.build
	cd web && npm ci --no-audit --no-fund
	cd web && PLAYWRIGHT_BROWSERS_PATH="$(abspath web/.playwright-browsers)" \
		npm run install:chromium
	mkdir -p web/src/vendor
	cp -r $(VISUALIZER_DIST)/* web/src/vendor/

# Five specs, deliberately different in what they check. The 0A spike drives a
# hand-authored fixture to settle questions about the RENDERER; the Stage 1 gate
# drives the session `visualize` actually produced, to settle a question about
# US -- whether a document [Me_session.validate] accepts is a document the
# renderer accepts. Our validator is our own reading of the schema. The Path B
# gate settles whether the cancellation LIFECYCLE is real, the comparison gate
# whether the two-pane surface is -- panes, a supplied correspondence, and
# per-pane data -- and the flow gate what the SELECTED-NODE event actually
# emits: `NodeInfo` carries no origin bit, so the flow adapter can only separate
# a user's click from its own setup `selectNode` by the identity and order of
# the events, and this enumerates that sequence rather than assuming it. The
# last three assert nearly everything they record: they are the premises their
# designs rest on, not observations about a third party we must accommodate.
spike.runtest:
	opam exec -- dune exec test/model_explorer/spike/gen_fixture.exe -- web/src
	opam exec -- dune exec bin/native_graph.exe -- visualize \
		--model modules/devcontainer.pytorch-image-models/models/mobilenetv2_050/models/model.json \
		--verify-symbolic quick --output web/src/session.json
	@detail_value=$$(jq -er 'first(.graphCollections[] | .graphs[] | select(.id == "g/kernel/000") | .nodes[] | .id | select(test("^v[0-9]+$$")) | ltrimstr("v"))' web/src/session.json); \
	opam exec -- dune exec bin/native_graph.exe -- detail \
		--model modules/devcontainer.pytorch-image-models/models/mobilenetv2_050/models/model.json \
		--graph g/kernel/000 --value "$$detail_value" --output web/src/detail-delta.json
	cd web && PLAYWRIGHT_BROWSERS_PATH="$(abspath web/.playwright-browsers)" \
		npm run spike

webapp.npm-install:
	cd web && npm ci --no-audit --no-fund

webapp.build: webapp.npm-install visualizer.build
	opam exec -- dune build js/webapp/webapp_worker.bc.js js/webapp/webapp_bridge.bc.js
	node web/scripts/build-webapp.mjs

webapp.serve: webapp.build
	cd webapp-dist && ../web/node_modules/.bin/http-server -p 8123 -c-1

webapp.runtest:
	node --check web/app/app.js
	node --check web/app/coordinator.js
	node --check web/app/panels.js
	node --check web/app/presentation.js
	node --check web/app/renderer.js
	node --check web/app/source_store.js
	node --test web/test/renderer-unit.test.mjs web/test/webapp-unit.test.mjs \
	  web/test/presentation-unit.test.mjs

# The jsoo bridge itself, driven the way the page drives it. Separate from
# webapp.runtest because it reads the dune output, which that target
# deliberately does not build -- and it is the only place the `Js.Unsafe`
# options decoder, its normalised echo, and a malformed raw JS value can be
# reached at all: test/model_explorer links model_explorer_export, not this
# executable.
webapp.bridge-runtest: webapp.build
	node --test web/test/bridge-unit.test.mjs

webapp.browser-runtest: webapp.build
	cd web && PLAYWRIGHT_BROWSERS_PATH="$(abspath web/.playwright-browsers)" npm run webapp

check: build format runtest melange.build melange.runtest
	git diff --check HEAD

# The local pre-commit gate: every check that needs no extra toolchain or
# downloaded model data (the JS backends and the pt2/inference suites are
# excluded -- see make js.runtest / make pt2.runtest). check.file-size runs
# first since, like in CI, it needs no build. Deliberately excludes
# verify.pristine: that target asserts NO uncommitted changes, which is
# never true of the changes precommit is meant to check.
precommit: build runtest format check.whitespace check.file-size \
	check.int-signatures
