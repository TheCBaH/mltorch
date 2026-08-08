#!/bin/sh
set -eu
set -x

echo "Activating feature 'npm packages'"

export DEBIAN_FRONTEND=noninteractive

if [ "$(find /var/lib/apt/lists/* | wc -l)" = "0" ]; then
    apt-get update -y
fi

apt-get -o Acquire::Retries=3 -y install --no-install-recommends \
    nodejs="${NODE_VERSION}" npm="${NPM_VERSION}"

if [ -n "${PACKAGES:-}" ]; then
    npm install -g $PACKAGES
fi

if [ "${PLAYWRIGHT:-false}" = "true" ]; then
    npm install -g playwright
    export PLAYWRIGHT_BROWSERS_PATH=/usr/local/ms-playwright
    npx playwright install --with-deps chromium
    chmod -R o+rx $PLAYWRIGHT_BROWSERS_PATH

    cat > /etc/profile.d/playwright.sh << 'PROFILE'
export PLAYWRIGHT_BROWSERS_PATH=/usr/local/ms-playwright
PROFILE
fi

apt-get clean
rm -rf /var/lib/apt/lists/*
