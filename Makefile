.PHONY: build test format runtest clean pt2.download pt2.download-all pt2.download-cram pt2.runtest pt2.vars inference inference-run

# Models release published at github.com/TheCBaH/pytorch.models.pt2
PT2_RELEASE := v0.0.4
PT2_REPO := TheCBaH/pytorch.models.pt2
PT2_MODEL := resnet18
PT2_DIR := data/pt2
# Each model gets its own subdir, so several models' weights/images/results
# can sit on disk at once (the interpreter cram tests are split by
# architecture, one model family per file).
PT2_MODEL_DIR := $(PT2_DIR)/$(PT2_MODEL)
PT2_ZIP := $(PT2_MODEL_DIR)/$(PT2_MODEL).release.zip
# Public release asset; the direct URL needs no auth (curl follows the redirect).
PT2_URL := https://github.com/$(PT2_REPO)/releases/download/$(PT2_RELEASE)/$(PT2_MODEL).release.zip

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
		test/interp_vit_cram.t

# Run the interpreter once on $(PT2_MODEL) (downloading it first if needed)
# and print per-image inference timing to stderr. No correctness assertion -
# the cram tests already check that for the smallest model per architecture;
# this is a smoke test + benchmark for every other model.
inference-run: pt2.download
	opam exec -- dune exec test/interp_run.exe -- \
		$(PT2_MODEL_DIR)/$(PT2_MODEL).pt2 $(PT2_MODEL_DIR)/images \
		$(PT2_MODEL_DIR)/imagenet_lsvrc_2015_synsets.txt \
		$(PT2_MODEL_DIR)/imagenet_metadata.txt $(PT2_MODEL_DIR)/results.json

# inference.<model>: run inference-run for one specific model.
inference.%:
	$(MAKE) inference-run PT2_MODEL=$*

# Run inference for every supported model (all of PT2_MODELS_ALL), not just
# the ones exercised by the cram tests.
inference: $(addprefix inference., $(PT2_MODELS_ALL))

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
