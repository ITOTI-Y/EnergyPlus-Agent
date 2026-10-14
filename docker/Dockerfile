FROM ubuntu:24.04

# idfpy 26.1 writes EnergyPlus 26.1 models, so the image installs that release;
# the official tarballs are built for Ubuntu 24.04 on x86_64 and arm64.
ARG ENERGYPLUS_VERSION=26.1.0
ARG ENERGYPLUS_BUILD=6f2e40d102
ARG TARGETARCH

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl libgomp1 libx11-6 && \
    rm -rf /var/lib/apt/lists/*

RUN case "$TARGETARCH" in \
        amd64) arch=x86_64 ;; \
        arm64) arch=arm64 ;; \
        *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac && \
    name="EnergyPlus-${ENERGYPLUS_VERSION}-${ENERGYPLUS_BUILD}-Linux-Ubuntu24.04-${arch}" && \
    curl -fsSL "https://github.com/NatLabRockies/EnergyPlus/releases/download/v${ENERGYPLUS_VERSION}/${name}.tar.gz" \
        | tar -xz -C /opt && \
    ln -s "/opt/${name}/energyplus" /usr/local/bin/energyplus

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_CACHE_DIR=/app/.cache/uv
ENV UV_PYTHON_INSTALL_DIR=/app/.local/share/uv/python

COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
COPY main.py ./

RUN mkdir -p /app/.cache && \
    uv sync --frozen && \
    chown -R 1000:1000 /app

EXPOSE 8000

CMD ["uv", "run", "main.py", "mcp-server", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
