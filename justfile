set dotenv-load := true

target := "$DOCKER_REPO"
version := "$(git rev-parse --short HEAD)"


@default:
    #!/usr/bin/env bash
    set -exo pipefail
    echo target: {{target}}
    echo version: {{version}}

docker-build:
    #!/usr/bin/env bash
    set -exo pipefail
    docker build -t "{{target}}:{{version}}" -t latest .

docker-push:
    #!/usr/bin/env bash
    set -exo pipefail
    docker push "{{target}}:{{version}}"

clear-dist:
    rm -rf ./dist

package: clear-dist
    uv build

publish: package
    uv publish --index xyme-pypi --username "test"