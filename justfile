set dotenv-load := true

target := "$DOCKER_REPO"
version := "$(git rev-parse --short HEAD)"


@default:
    #!/usr/bin/env bash
    set -exo pipefail
    echo target: {{target}}
    echo version: {{version}}

# Build the docker image
docker-build:
    #!/usr/bin/env bash
    set -exo pipefail
    docker build -t "{{target}}:{{version}}" -t latest .

# Push the docker image to the repository
docker-push:
    #!/usr/bin/env bash
    set -exo pipefail
    docker push "{{target}}:{{version}}"

# Run the unit test suite
test:
    python -m pytest tests

# Format the code
lint:
    pyright .

# Format the code
format:
    ruff format

# Check whether the code is formatted
format-check:
    ruff format --check

clear-dist:
    rm -rf ./dist

package: clear-dist
    uv build

# Publish the package to the xyme index
publish: package
    uv publish --index xyme-pypi --username "test"