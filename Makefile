.PHONY: build test format runtest clean

build:
	opam exec -- dune build $(BUILD_OPTIONS)

format:
	opam exec -- dune fmt

test:
	opam exec -- dune test

runtest:
	opam exec -- dune runtest

clean:
	opam exec -- dune clean
	git submodule foreach --recursive git clean -xdf
