.PHONY: spike.setup spike.runtest webapp.npm-install webapp.build webapp.serve webapp.runtest webapp.browser-runtest melange.build melange.build.scaffold melange.runtest build test format runtest clean pt2.download pt2.download-all pt2.download-cram pt2.runtest pt2.vars inference inference-runa native-infer-verify native-infer-verify.% native-transform-verify native-transform-verify.% jsoo.build jsoo.runtest jsoo.inline-runtest jsoo.pt2.runtest jsoo.pt2.run jsoo.pt2.download jsoo.pt2.vars js.build js.runtest
all: build

# Models release published at github.com/TheCBaH/pytorch.models.pt2
PT2_RELEASE := v0.0.4
PT2_REPO := TheCBaH/pytorch.models.pt2
PT2_MODEL := resnet18
PT2_DIR := data/pt2
# Each model gets its own subdir, so several models' weights/images/results
# can sit on disk at once (the interpreter cram tests are split by
# architecture, one model family per file).
# `=` (lazily expanded), not `:=`: `make -j inference` overrides PT2_MODEL
# per inference.<model> target via a pattern-specific variable (below), and
# these three need to re-expand against that override at recipe time rather
# than freeze the top-of-file default at parse time.
PT2_MODEL_DIR = $(PT2_DIR)/$(PT2_MODEL)
PT2_ZIP = $(PT2_MODEL_DIR)/$(PT2_MODEL).release.zip
# Public release asset; the direct URL needs no auth (curl follows the redirect).
PT2_URL = https://github.com/$(PT2_REPO)/releases/download/$(PT2_RELEASE)/$(PT2_MODEL).release.zip

# Models exercised by the graph interpreter, grouped by architecture.
PT2_MODELS_RESNET := resnet18 resnet34 resnet50 resnet101 resnet152
PT2_MODELS_EFFICIENTNET := efficientnet_b0 efficientnet_b1 efficientnet_b2 \
	efficientnet_b3 efficientnet_b4 efficientnet_b5 efficientnet_b6 \
	efficientnet_b7 efficientnet_v2_s efficientnet_v2_m efficientnet_v2_l
PT2_MODELS_MOBILENET := mobilenet_v2 mobilenet_v3_large mobilenet_v3_small
PT2_MODELS_VIT := vit_b_16 vit_b_32 vit_l_16 vit_l_32
PT2_MODELS_ALL := $(PT2_MODELS_RESNET) $(PT2_MODELS_EFFICIENTNET) \
	$(PT2_MODELS_MOBILENET) $(PT2_MODELS_VIT)

# The smallest/fastest model per architecture - exactly what
# test/interp_*_cram.t run, and all `make pt2.runtest` needs on disk.
# mobilenet_v2 is the one exception to "smallest per architecture": the native
# graph/transform crams read it as well as v3_small, because v2 is the only
# model in the zoo that uses hardtanh (relu6) and so the only whole-graph
# coverage that constructor has. It is structural-only there, so the cost is the
# download, not runtime.
PT2_MODELS_CRAM := resnet18 efficientnet_b0 mobilenet_v2 mobilenet_v3_small \
	vit_b_32

# Whole-graph native verification is intentionally a small, explicit list:
# native convolution is pure OCaml and much slower than the ATen reference.
# Both native-infer-verify and native-transform-verify expand this, so each
# entry costs roughly five full native inferences.
#
# mobilenet_v3_small REPLACES resnet18: measured across both verify targets it
# costs 40.9s of CPU against resnet18's 423.3s (0.10x), while covering the
# importer arms resnet18 cannot reach — clamp, clone, mul, div and the
# compile-time scalars the exporter writes into Tensor slots. resnet18's one
# unique op, max_pool2d_with_indices, keeps its coverage in the targeted tests
# (native/compute, graph, symbolic, native_bridge_test and the generated ATen
# walks) rather than in a whole-model runtime job an order of magnitude more
# expensive than the rest of this list. Run it by hand with
# `make native-infer-verify.resnet18`. Measurements and the four-model
# comparison are in .ai/native_inference_verify.md.
PT2_NATIVE_VERIFY_MODELS := mobilenet_v3_small

# Fetch the release zip (model .pt2 + sample input images) for the interpreter
# tests, under data/pt2/<model>/. Gitignored and not part of the default build
# (network-bound). The download is skipped when the zip is already present
# (e.g. restored from a CI cache); the unzip is cheap and always refreshed.
pt2.download:
	mkdir -p $(PT2_MODEL_DIR)
	test -f $(PT2_ZIP) || curl -fsSL -o $(PT2_ZIP) $(PT2_URL)
	cd $(PT2_MODEL_DIR) && unzip -o $(PT2_MODEL).release.zip

# Download the models the cram tests need (cheap next to the full set: 5
# models, ~410 MB).
pt2.download-cram:
	for m in $(PT2_MODELS_CRAM); do $(MAKE) pt2.download PT2_MODEL=$$m; done

# Download every supported model, one at a time (expensive: 22 models,
# ~5.3 GB). Used by `make inference` and for ad hoc spot-checks.
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
PT2_MODELS_HASH := $(shell echo "$(PT2_MODELS_ALL) $(PT2_MODELS_CRAM)" \
	| md5sum | cut -c1-8)
pt2.vars:
	@echo "pt2_zip_glob=$(PT2_DIR)/*/*.release.zip"
	@echo "pt2_cache_key=pt2-$(PT2_RELEASE)-$(PT2_MODELS_HASH)"
	@echo "pt2_cache_restore_key=pt2-$(PT2_RELEASE)-"

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
	done
	PT2_DATA=$(abspath $(PT2_DIR)) NO_COLOR=1 opam exec -- dune runtest \
		test/pt2_load_cram.t test/interp_resnet_cram.t \
		test/interp_efficientnet_cram.t test/interp_mobilenet_cram.t \
		test/interp_vit_cram.t test/native_graph_cram.t \
		test/native_transform_cram.t test/native_transform_fold_cram.t \
		test/native_transform_verify_cram.t \
		test/native_transform_verify_fold_cram.t \
		test/native4d_to4d_cram.t test/me_visualize_cram.t \
		--auto-promote

# Shared argument list for every interp_run.exe invocation below, so
# inference-run and benchmark.inference can't drift apart.
PT2_INFER_ARGS = $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 $(PT2_MODEL_DIR)/images \
	$(PT2_MODEL_DIR)/imagenet_lsvrc_2015_synsets.txt \
	$(PT2_MODEL_DIR)/imagenet_metadata.txt $(PT2_MODEL_DIR)/results.json

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

# Run inference for every supported model (all of PT2_MODELS_ALL), not just
# the ones exercised by the cram tests. Every inference.<model> is
# independent (own dir, own download, own process), so this is safe to run
# with `-j` (e.g. `make -j$(nproc) inference`) to cut wall time.
inference: $(addprefix inference., $(PT2_MODELS_ALL))

# Keep ATen out of native_graph: this target first writes a reference with the
# existing Interp.run evaluator, then asks the pure native executable to read
# and compare it.  The reference is temporary, never a committed golden file.
ATEN_GRAPH_REF := _build/default/bin/aten_graph_ref.exe
NATIVE_GRAPH := _build/default/bin/native_graph.exe
$(ATEN_GRAPH_REF) $(NATIVE_GRAPH):
	opam exec -- dune build bin/aten_graph_ref.exe bin/native_graph.exe

native-infer-verify.%: PT2_MODEL = $*
native-infer-verify.%: $(ATEN_GRAPH_REF) $(NATIVE_GRAPH)
	test -f $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 || $(MAKE) pt2.download PT2_MODEL=$*
	set -eux; ref=$$(mktemp); trap 'rm -f "$$ref"' EXIT; \
	$(ATEN_GRAPH_REF) $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 $(PT2_MODEL_DIR)/images/000000000149.pt "$$ref"; \
	$(NATIVE_GRAPH) eval --pt2 $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 --input $(PT2_MODEL_DIR)/images/000000000149.pt --expect "$$ref" --verbose

native-infer-verify: $(addprefix native-infer-verify., $(PT2_NATIVE_VERIFY_MODELS))

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
		--input $(PT2_MODEL_DIR)/images/000000000149.pt; \
	$(NATIVE_GRAPH) transform --fold --verify \
		--pt2 $(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 \
		--input $(PT2_MODEL_DIR)/images/000000000149.pt; \
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

JS_PT2_MODEL := mobilenet_v3_small
JS_PT2_DIR := $(PT2_DIR)/$(JS_PT2_MODEL)
JS_PT2_ARCHIVE := $(JS_PT2_DIR)/$(JS_PT2_MODEL).pt2
JS_PT2_INPUT := $(JS_PT2_DIR)/images/000000000149.pt

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
	@echo "jsoo_pt2_zip_glob=$(PT2_DIR)/$(JS_PT2_MODEL)/*.release.zip"
	@echo "jsoo_pt2_cache_key=pt2-jsoo-$(PT2_RELEASE)-$(JS_PT2_MODEL)"
	@echo "jsoo_pt2_cache_restore_key=pt2-jsoo-$(PT2_RELEASE)-"

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

JS_PT2_RUN_ARGS = $(JS_PT2_ARCHIVE) $(JS_PT2_DIR)/images \
	$(JS_PT2_DIR)/imagenet_lsvrc_2015_synsets.txt \
	$(JS_PT2_DIR)/imagenet_metadata.txt $(JS_PT2_DIR)/results.json

jsoo.pt2.run: jsoo.pt2.download jsoo.build
	node $(JS_BUILD)/jsoo/pt2_run.bc.js $(JS_PT2_RUN_ARGS) --strict

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

js.runtest: jsoo.runtest jsoo.inline-runtest melange.runtest

clean:
	opam exec -- dune clean
	git submodule foreach --recursive git clean -xdf

# --- Stage 0A contract spike -----------------------------------------------
#
# Drives the real pinned Model Explorer element to settle the questions the
# browser coordinator design assumes. Needs node, npm and a downloaded
# Chromium, so it is outside every other target; its ANSWERS are recorded in
# .ai/model_explorer_design.md and do not need re-deriving to read.

spike.setup:
	cd web && npm ci --no-audit --no-fund
	cd web && PLAYWRIGHT_BROWSERS_PATH="$(abspath web/.playwright-browsers)" \
		npm run install:chromium
	mkdir -p web/src/vendor
	cp -r web/node_modules/ai-edge-model-explorer-visualizer/dist/* web/src/vendor/

# Two specs, deliberately different in what they check. The 0A spike drives a
# hand-authored fixture to settle questions about the RENDERER; the Stage 1 gate
# drives the session `visualize` actually produced, to settle a question about
# US -- whether a document [Me_session.validate] accepts is a document the
# renderer accepts. Our validator is our own reading of the schema.
spike.runtest:
	opam exec -- dune exec test/model_explorer/spike/gen_fixture.exe -- web/src
	opam exec -- dune exec bin/native_graph.exe -- visualize \
		--model modules/pytorch.models.pt2/models/resnet18/models/model.json \
		--verify-symbolic quick --output web/src/session.json
	opam exec -- dune exec bin/native_graph.exe -- detail \
		--model modules/pytorch.models.pt2/models/resnet18/models/model.json \
		--graph g/kernel/000 --value 125 --output web/src/detail-delta.json
	cd web && PLAYWRIGHT_BROWSERS_PATH="$(abspath web/.playwright-browsers)" \
		npm run spike

webapp.npm-install:
	cd web && npm ci --no-audit --no-fund

webapp.build: webapp.npm-install
	opam exec -- dune build js/webapp/webapp_worker.bc.js js/webapp/webapp_bridge.bc.js
	node web/scripts/build-webapp.mjs

webapp.serve: webapp.build
	cd webapp-dist && ../web/node_modules/.bin/http-server -p 8123 -c-1

webapp.runtest:
	node --check web/app/app.js
	node --check web/app/coordinator.js
	node --check web/app/renderer.js
	node --check web/app/source_store.js
	node --test web/test/renderer-unit.test.mjs web/test/webapp-unit.test.mjs

webapp.browser-runtest: webapp.build
	cd web && PLAYWRIGHT_BROWSERS_PATH="$(abspath web/.playwright-browsers)" npm run webapp
