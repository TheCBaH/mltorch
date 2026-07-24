.PHONY: build test format runtest clean pt2.download pt2.download-all pt2.download-cram pt2.runtest pt2.vars inference inference-runa
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
PT2_MODELS_CRAM := resnet18 efficientnet_b0 mobilenet_v3_small vit_b_32

# Fetch the release zip (model .pt2 + sample input images) for the interpreter
# tests, under data/pt2/<model>/. Gitignored and not part of the default build
# (network-bound). The download is skipped when the zip is already present
# (e.g. restored from a CI cache); the unzip is cheap and always refreshed.
pt2.download:
	mkdir -p $(PT2_MODEL_DIR)
	test -f $(PT2_ZIP) || curl -fsSL -o $(PT2_ZIP) $(PT2_URL)
	cd $(PT2_MODEL_DIR) && unzip -o $(PT2_MODEL).release.zip

# Download the models the cram tests need (cheap: 4 models, ~150 MB).
pt2.download-cram:
	for m in $(PT2_MODELS_CRAM); do $(MAKE) pt2.download PT2_MODEL=$$m; done

# Download every supported model, one at a time (expensive: 22 models,
# ~5.3 GB). Used by `make inference` and for ad hoc spot-checks.
pt2.download-all:
	for m in $(PT2_MODELS_ALL); do $(MAKE) pt2.download PT2_MODEL=$$m; done

# Emit the pt2 cache path/key as KEY=VALUE lines, so CI reads them from here
# (the single source of truth) instead of re-deriving them in YAML. One cache
# entry covers every model zip downloaded so far this release (glob path); the
# key folds in a hash of PT2_MODELS_ALL so adding/removing a model creates a
# fresh entry (a cache is immutable once saved under a given key) instead of
# silently never picking up the change. restore-key drops the hash, so a
# model-list change still warm-starts from the closest prior cache.
PT2_MODELS_HASH := $(shell echo "$(PT2_MODELS_ALL)" | md5sum | cut -c1-8)
pt2.vars:
	@echo "pt2_zip_glob=$(PT2_DIR)/*/*.release.zip"
	@echo "pt2_cache_key=pt2-$(PT2_RELEASE)-$(PT2_MODELS_HASH)"
	@echo "pt2_cache_restore_key=pt2-$(PT2_RELEASE)-"

# Run the gated end-to-end tests (load + interpreter) against the downloaded data.
pt2.runtest:
	PT2_DATA=$(abspath $(PT2_DIR)) opam exec -- dune runtest \
		test/pt2_load_cram.t test/interp_resnet_cram.t \
		test/interp_efficientnet_cram.t test/interp_mobilenet_cram.t \
		test/interp_vit_cram.t test/native_graph_cram.t

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

runtest:
	opam exec -- dune runtest --auto-promote

clean:
	opam exec -- dune clean
	git submodule foreach --recursive git clean -xdf
