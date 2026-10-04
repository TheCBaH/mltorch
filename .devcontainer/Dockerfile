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
  libasound2t64\
  libatk-bridge2.0-0t64\
  libatk1.0-0t64\
  libatspi2.0-0t64\
  libcairo2\
  libclang-rt-dev-wasm32\
  libcups2t64\
  libdbus-1-3\
  libdrm2\
  libffi-dev\
  libfontconfig1\
  libfreetype6\
  libgbm1\
  libglib2.0-0t64\
  libnspr4\
  libnss3\
  libpango-1.0-0\
  libx11-6\
  libxcb1\
  libxcomposite1\
  libxdamage1\
  libxext6\
  libxfixes3\
  libxkbcommon0\
  libxrandr2\
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
