#!/bin/bash
# Update the Homebrew formula with a new version's URL and SHA256.
# Usage: ./update-formula.sh 0.2.1

set -euo pipefail

VERSION="${1:?Usage: $0 <version>}"
FORMULA="$(dirname "$0")/Formula/abbacus-cortex.rb"

URL="https://files.pythonhosted.org/packages/source/a/abbacus-cortex/abbacus_cortex-${VERSION}.tar.gz"

echo "Fetching ${URL}..."
SHA=$(curl -sL "$URL" | shasum -a 256 | cut -d' ' -f1)

if [ -z "$SHA" ] || [ "$SHA" = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ]; then
    echo "Error: Could not download sdist (empty or missing). Is v${VERSION} published on PyPI?"
    exit 1
fi

sed -i '' "s|url \".*\"|url \"${URL}\"|" "$FORMULA"
sed -i '' "s|sha256 \".*\"|sha256 \"${SHA}\"|" "$FORMULA"

echo "Updated formula to v${VERSION}"
echo "  url:    ${URL}"
echo "  sha256: ${SHA}"
