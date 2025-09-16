set dotenv-load := true

target := "$DOCKER_REPO"
version := "$(git rev-parse --short HEAD)"


@default:
    #!/usr/bin/env bash
    set -exo pipefail
    echo target: {{target}}
    echo version: {{version}}

build:
    #!/usr/bin/env bash
    set -exo pipefail
    docker build -t "{{target}}:{{version}}" -t latest .

push:
    #!/usr/bin/env bash
    set -exo pipefail
    docker push "{{target}}:{{version}}"
