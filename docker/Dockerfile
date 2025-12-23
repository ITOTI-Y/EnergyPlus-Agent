FROM nrel/energyplus:25.1.0 AS energyplus

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=energyplus /EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64 /opt/energyplus
ENV PATH="/opt/energyplus:$PATH"

WORKDIR /app

EXPOSE 8000