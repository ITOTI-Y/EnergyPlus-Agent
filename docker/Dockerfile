FROM nrel/energyplus:25.1.0

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_CACHE_DIR=/app/.cache/uv
ENV UV_PYTHON_INSTALL_DIR=/app/.local/share/uv/python

RUN mkdir -p ./data/dependencies && \
    cp /EnergyPlus-*/Energy+.idd ./data/dependencies/
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY main.py ./

RUN mkdir -p /app/.cache && \
    uv sync --frozen && \
    chown -R 1000:1000 /app

EXPOSE 8000

CMD ["uv", "run", "main.py", "mcp-server", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]