"""The reference library on the Hugging Face Hub, a private dataset repo."""

from pathlib import Path
from typing import Final

from huggingface_hub import HfApi, hf_hub_download

from src.reference import library

LIBRARY_FILE: Final = "reference_library.sqlite"

CARD: Final = """---
license: other
pretty_name: EnergyPlus-Agent reference library
tags: [energyplus, building-energy, embeddings]
---

# EnergyPlus-Agent reference library

Materials, constructions and schedules of the DOE prototype building models,
deduplicated, described and embedded for semantic search by EnergyPlus-Agent.

- Sources: commercial prototypes (ASHRAE 90.1-2019 and 2022, 16 building
  types, 19 locations) and residential prototypes (IECC 2021 and 2024,
  single- and multifamily, four foundations, 16 climate zones), from
  https://www.energycodes.gov/prototype-building-models.
- Objects are taken from the models reduced to these types and set to
  EnergyPlus 26.1; the 22.1 to 26.1 transition rules change none of them.
- `{file}`: SQLite with tables `reference_object` (one row per distinct
  object, JSON data, description, building types, standards, climate zones,
  uses), `embedding` (float32 vectors, little-endian) and `metadata`.

Metadata: {metadata}

Cite the sources as: DOE and PNNL. 2023. Commercial Prototype Building
Models. Richland, WA: Pacific Northwest National Laboratory; and the PNNL
residential prototype building models.
"""


def publish(path: Path, repo: str) -> str:
    """Upload the library and its card to a private dataset repo.

    Returns:
        The commit URL.
    """
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    card = CARD.format(file=LIBRARY_FILE, metadata=library.metadata(path))
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    commit = api.upload_file(
        path_or_fileobj=path,
        path_in_repo=LIBRARY_FILE,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Update {LIBRARY_FILE}",
    )
    return commit.commit_url


def pull(repo: str, path: Path) -> Path:
    """Download the library to ``path``."""
    downloaded = hf_hub_download(repo, LIBRARY_FILE, repo_type="dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Path(downloaded).read_bytes())
    return path
