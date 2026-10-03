ARG VARIANT=13
FROM debian:${VARIANT}
RUN set -eux;\
    apt-get update;\
    env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends\
  ca-certificates\
  ccache\
  clang\
  cmake\
  curl\
  g++\
  git\
  libclang-rt-dev-wasm32\
  libffi-dev\
  lld\
  make\
  pkg-config\
  python3\
  python3-typing-extensions\
  python3-yaml\
  ripgrep\
  time\
  wasi-libc\
  wget\
  xxd\
  xz-utils\
;\
    apt-get autoremove -y; apt-get clean -y ; rm -rf /var/lib/apt/lists/*;\
    echo DONE
