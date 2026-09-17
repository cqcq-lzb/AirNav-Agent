#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$bundle_dir"

archive=airway-navigation-h100.tar.gz
sha256sum -c "$archive.sha256"
gzip -dc "$archive" | docker load
test -f .env || { cp .env.example .env; printf '%s\n' "Edit $bundle_dir/.env, then run this script again."; exit 2; }
mkdir -p runtime_data runtime_cases runtime_status
docker compose -f docker-compose.gpu.yml up -d --no-build
docker compose -f docker-compose.gpu.yml ps
