# Unified Data Loader Design Document

> **Status**: Ready for Implementation  
> **Author**: OpenCode  
> **Date**: 2025-12-07

## Overview

This document describes the design for a unified data loading system for NERVE that consolidates all data download and Neo4j loading scripts into a single, extensible CLI tool.

### Goals

1. **Single command** to load all data into Neo4j
2. **Unified configuration** via `.env` file
3. **Extensible architecture** for future data sources
4. **Parallel execution** where possible for performance
5. **Robust error handling** with retry and fail-fast behavior

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Monolithic script with plugin-based sources | Simpler deployment, single entry point |
| Default mode | **REPLACE** (DELETE + CREATE) | Faster for full reloads |
| Alternative mode | `--merge` flag for idempotent MERGE | Safe for incremental updates |
| Error handling | Retry with exponential backoff, then fail fast | Don't mask failures |
| Parallelization | Yes, within independent stages | Performance optimization |
| Progress output | Rich progress bars | Better UX |
| Entry point | `python -m nerve.loader` | Consistent with existing CLI |
| Old scripts | Remove (available in git history) | Reduce maintenance burden |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      src/nerve/loader/                          │
├─────────────────────────────────────────────────────────────────┤
│  __init__.py      │ Package exports                             │
│  __main__.py      │ CLI entry point                             │
│  config.py        │ Config dataclass, .env loading              │
│  protocol.py      │ DataSource protocol, LoadStats              │
│  registry.py      │ Source discovery and dependency resolution  │
│  executor.py      │ Stage execution with parallelization        │
│  retry.py         │ Exponential backoff retry handler           │
│  sources/         │ Individual data source implementations      │
│    __init__.py    │                                             │
│    monarch.py     │ MonarchKGSource                             │
│    hpo.py         │ HPOSource                                   │
│    reactome.py    │ ReactomeSource                              │
│    hgnc.py        │ HGNCSource                                  │
│    cosmic.py      │ COSMICSource                                │
│    disgenet.py    │ DisGeNETSource                              │
│    publications.py│ Metadata, Retractions, Citations sources    │
│    hpo_siblings.py│ HPOSiblingMapSource                         │
└─────────────────────────────────────────────────────────────────┘
```

### DataSource Protocol

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class LoadStats:
    """Statistics from a load operation."""
    source: str
    nodes_created: int = 0
    edges_created: int = 0
    nodes_updated: int = 0
    edges_updated: int = 0
    duration_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None

class DataSource(Protocol):
    """Protocol for all data sources."""
    
    # Class attributes
    name: str                      # Unique identifier (e.g., "monarch", "disgenet")
    display_name: str              # Human-readable name
    stage: int                     # Execution stage (1-4)
    requires_credentials: list[str]  # Required env vars (empty = no creds needed)
    dependencies: list[str]        # Sources that must complete first
    
    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """Check if required credentials are available.
        
        Returns:
            (True, None) if credentials available or not required
            (False, "reason") if credentials missing
        """
        ...
    
    def download(self, config: Config, force: bool = False) -> None:
        """Download data files. May be no-op for API-only sources.
        
        Args:
            config: Loader configuration
            force: Re-download even if files exist
            
        Raises:
            DownloadError: If download fails after retries
        """
        ...
    
    def load(
        self, 
        driver: Neo4jDriver, 
        config: Config, 
        mode: Literal["replace", "merge"]
    ) -> LoadStats:
        """Load data into Neo4j.
        
        Args:
            driver: Neo4j driver instance
            config: Loader configuration
            mode: "replace" (fast, destructive) or "merge" (idempotent)
            
        Returns:
            LoadStats with counts and timing
            
        Raises:
            LoadError: If loading fails after retries
        """
        ...
```

---

## Execution Stages

Sources are organized into stages with dependency ordering:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Core KG Data                        │
│  Sequential execution (monarch creates schema first)            │
├─────────────────────────────────────────────────────────────────┤
│  1. monarch  - Monarch KG nodes & edges (creates schema)        │
│  2. hpo      - HPO gene/disease phenotype annotations           │
│  3. reactome - Reactome pathway-gene relationships              │
│  4. hgnc     - HGNC ID mapping (memory only, enables others)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 2: External Enrichment                    │
│  Parallel execution (independent sources)                       │
├─────────────────────────────────────────────────────────────────┤
│  • cosmic   - Cancer Gene Census (requires COSMIC creds)        │
│  • disgenet - Gene-disease associations (optional API key)      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 3: Publication Enrichment                  │
│  Parallel execution (all enrich Publication nodes)              │
├─────────────────────────────────────────────────────────────────┤
│  • pub_metadata - Title, authors, journal, year                 │
│  • retractions  - Retraction status                             │
│  • citations    - CITES relationships                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 4: Post-Processing                        │
│  Sequential execution                                           │
├─────────────────────────────────────────────────────────────────┤
│  • hpo_siblings - HPO sibling map JSON for GNN training         │
└─────────────────────────────────────────────────────────────────┘
```

### Source Registry

| Source | Stage | Credentials | Dependencies | Parallel |
|--------|-------|-------------|--------------|----------|
| `monarch` | 1 | None | - | No (first) |
| `hpo` | 1 | None | monarch | No |
| `reactome` | 1 | None | monarch | No |
| `hgnc` | 1 | None | - | No |
| `cosmic` | 2 | COSMIC_EMAIL, COSMIC_PASSWORD | monarch | Yes |
| `disgenet` | 2 | DISGENET_API_KEY (optional) | monarch | Yes |
| `pub_metadata` | 3 | NCBI_API_KEY (optional) | monarch | Yes |
| `retractions` | 3 | NCBI_API_KEY (optional) | monarch | Yes |
| `citations` | 3 | NCBI_API_KEY (optional) | monarch | Yes |
| `hpo_siblings` | 4 | None | hpo | No |

---

## Configuration

### `.env.example`

```bash
# =============================================================================
# Neo4j Configuration (Required)
# =============================================================================
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=nerve123

# =============================================================================
# API Keys (Optional - enables additional features/sources)
# =============================================================================

# NCBI API key for higher rate limits (10 req/s vs 3 req/s)
# Get one at: https://www.ncbi.nlm.nih.gov/account/settings/
NCBI_API_KEY=

# DisGeNET API key for gene-disease associations
# Register at: https://www.disgenet.org/api/
DISGENET_API_KEY=

# COSMIC credentials for Cancer Gene Census
# Register at: https://cancer.sanger.ac.uk/cosmic/register
COSMIC_EMAIL=
COSMIC_PASSWORD=

# =============================================================================
# Data Loading Options
# =============================================================================

# Base directory for downloaded data files (default: ./data)
DATA_DIR=./data

# COSMIC version to download (default: v103)
COSMIC_VERSION=v103

# Species filter for Reactome pathways (default: Homo sapiens)
REACTOME_SPECIES=Homo sapiens

# Batch size for Neo4j transactions (default: 5000)
NEO4J_BATCH_SIZE=5000

# =============================================================================
# Retry Configuration
# =============================================================================

# Maximum retry attempts for failed operations (default: 3)
MAX_RETRIES=3

# Initial retry delay in seconds (default: 2.0)
RETRY_INITIAL_DELAY=2.0

# Maximum retry delay cap in seconds (default: 60.0)
RETRY_MAX_DELAY=60.0
```

### Config Dataclass

```python
@dataclass
class Config:
    """Loader configuration loaded from environment."""
    
    # Neo4j connection
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    
    # API keys (optional)
    ncbi_api_key: str | None = None
    disgenet_api_key: str | None = None
    cosmic_email: str | None = None
    cosmic_password: str | None = None
    
    # Paths
    data_dir: Path = Path("./data")
    
    # Options
    cosmic_version: str = "v103"
    reactome_species: str = "Homo sapiens"
    batch_size: int = 5000
    
    # Retry settings
    max_retries: int = 3
    retry_initial_delay: float = 2.0
    retry_max_delay: float = 60.0
    
    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Config":
        """Load configuration from environment variables and optional .env file."""
        ...
    
    def has_cosmic_credentials(self) -> bool:
        return bool(self.cosmic_email and self.cosmic_password)
    
    def has_ncbi_api_key(self) -> bool:
        return bool(self.ncbi_api_key)
    
    def has_disgenet_api_key(self) -> bool:
        return bool(self.disgenet_api_key)
```

---

## CLI Interface

### Entry Point

```bash
# Standard invocation
uv run python -m nerve.loader

# Or via package script (if added to pyproject.toml)
uv run nerve-loader
```

### Commands and Options

```bash
# Full load (default: REPLACE mode)
uv run python -m nerve.loader

# Use MERGE mode (idempotent, slower)
uv run python -m nerve.loader --merge

# Select specific sources
uv run python -m nerve.loader --sources monarch,hpo,disgenet

# Skip specific sources
uv run python -m nerve.loader --skip cosmic,citations

# Run only specific stages
uv run python -m nerve.loader --stages 1,2

# Download only (no loading)
uv run python -m nerve.loader --download-only

# Skip download (use existing files)
uv run python -m nerve.loader --skip-download

# Force re-download all files
uv run python -m nerve.loader --force-download

# Sample mode for testing (load subset)
uv run python -m nerve.loader --sample 10000

# Dry run (show what would be done)
uv run python -m nerve.loader --dry-run

# Custom .env file
uv run python -m nerve.loader --env .env.production

# List available sources with status
uv run python -m nerve.loader --list-sources

# Override retry settings
uv run python -m nerve.loader --max-retries 5 --retry-delay 2.0

# Verbose output
uv run python -m nerve.loader --verbose
```

### Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--merge` | flag | false | Use MERGE mode instead of REPLACE |
| `--sources` | str | all | Comma-separated list of sources to load |
| `--skip` | str | none | Comma-separated list of sources to skip |
| `--stages` | str | all | Comma-separated list of stages to run (1-4) |
| `--download-only` | flag | false | Only download, don't load to Neo4j |
| `--skip-download` | flag | false | Skip download, use existing files |
| `--force-download` | flag | false | Re-download even if files exist |
| `--sample` | int | none | Load only first N items per source |
| `--dry-run` | flag | false | Show plan without executing |
| `--env` | path | .env | Path to .env file |
| `--max-retries` | int | 3 | Maximum retry attempts |
| `--retry-delay` | float | 2.0 | Initial retry delay (seconds) |
| `--verbose` | flag | false | Enable verbose logging |

---

## Error Handling

### Retry Strategy

```python
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 2.0  # seconds
MAX_DELAY_CAP = 60.0  # cap exponential backoff

class RetryHandler:
    """Handles retries with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        max_delay: float = MAX_DELAY_CAP,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
    
    def execute(self, func: Callable[[], T], operation_name: str) -> T:
        """Execute function with retry logic.
        
        Raises:
            FatalLoadError: After all retries exhausted
        """
        delay = self.initial_delay
        last_error: Exception | None = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except RETRYABLE_ERRORS as e:
                last_error = e
                if attempt == self.max_retries:
                    break
                console.print(
                    f"  [yellow]⚠ {operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay:.1f}s...[/yellow]"
                )
                time.sleep(delay)
                delay = min(delay * 2, self.max_delay)
        
        raise FatalLoadError(
            f"{operation_name} failed after {self.max_retries + 1} attempts: {last_error}"
        )

# Errors that should trigger retry
RETRYABLE_ERRORS = (
    URLError,
    HTTPError,
    ConnectionError,
    TimeoutError,
    RateLimitError,
    IncompleteRead,
)
```

### Fail-Fast Behavior

When a source fails after all retries:

1. **Stop execution** immediately
2. **Print summary** of what completed successfully
3. **Print clear error message** with the failure reason
4. **Exit with non-zero status**
5. **Suggest recovery**: User can re-run with `--skip` for completed sources

```
╭──────────────────────────────────────────────────────────────────╮
│  ✗ LOAD FAILED                                                   │
╰──────────────────────────────────────────────────────────────────╯

Source 'disgenet' failed after 3 retries:
  HTTPError 429: Too Many Requests

Completed sources:
  ✓ monarch (1,234,567 nodes, 5,678,901 edges)
  ✓ hpo (12,345 associations)
  ✓ reactome (23,456 relationships)

To resume, run:
  uv run python -m nerve.loader --skip monarch,hpo,reactome
```

---

## REPLACE vs MERGE Mode

### REPLACE Mode (Default)

Fastest option for full reloads. Deletes existing data from source before loading.

```python
def load_replace(self, driver: Neo4jDriver, config: Config) -> LoadStats:
    """Fast loading with clean slate per source."""
    with driver.session() as session:
        # Clear existing data from this source
        session.run(
            "MATCH (n) WHERE n._source = $source DETACH DELETE n",
            source=self.name
        )
        
        # Batch insert with CREATE (faster than MERGE)
        for batch in self.iter_batches(config):
            session.run(
                "UNWIND $batch AS row CREATE (n:Node) SET n = row",
                batch=batch
            )
```

### MERGE Mode (`--merge`)

Idempotent option for incremental updates. Preserves existing data.

```python
def load_merge(self, driver: Neo4jDriver, config: Config) -> LoadStats:
    """Idempotent loading, preserves existing data."""
    with driver.session() as session:
        # Use MERGE for upsert behavior
        for batch in self.iter_batches(config):
            session.run(
                "UNWIND $batch AS row "
                "MERGE (n:Node {id: row.id}) "
                "SET n += row",
                batch=batch
            )
```

---

## Progress Output

Using Rich library for progress visualization:

```
╭──────────────────────────────────────────────────────────────────╮
│                     NERVE Data Loader v1.0                       │
╰──────────────────────────────────────────────────────────────────╯

Configuration:
  Neo4j:     bolt://localhost:7687
  Data dir:  ./data
  Mode:      REPLACE

Credentials:
  NCBI API Key:    ✓ configured
  DisGeNET API:    ✓ configured
  COSMIC:          ✗ not configured (source will be skipped)

══════════════════════════════════════════════════════════════════
 STAGE 1: Core KG Data
══════════════════════════════════════════════════════════════════

[1/4] Monarch KG
  Downloading nodes ━━━━━━━━━━━━━━━━━━━━ 100% 45.2/45.2 MB
  Downloading edges ━━━━━━━━━━━━━━━━━━━━ 100% 123.8/123.8 MB
  Loading nodes     ━━━━━━━━━━━━━━━━━━━━ 100% 1,234,567 nodes
  Loading edges     ━━━━━━━━━━━━━━━━━━━━ 100% 5,678,901 edges
  ✓ Complete (2m 34s)

[2/4] HPO Annotations
  Downloading       ━━━━━━━━━━━━━━━━━━━━ 100% 2 files
  Loading           ━━━━━━━━━━━━━━━━━━━━ 100% 21,246 associations
  ✓ Complete (15s)

[3/4] Reactome Pathways
  Loading           ━━━━━━━━━━━━━━━━━━━━ 100% 23,456 relationships
  ✓ Complete (8s)

[4/4] HGNC Mapping
  Loading           ━━━━━━━━━━━━━━━━━━━━ 100% 43,000 mappings
  ✓ Complete (2s)

══════════════════════════════════════════════════════════════════
 STAGE 2: External Enrichment
══════════════════════════════════════════════════════════════════

[1/2] COSMIC CGC
  ⊘ Skipped (credentials not configured)

[2/2] DisGeNET
  Querying API      ━━━━━━━━━━━━━━━━━━━━ 100% 21 genes
  Loading           ━━━━━━━━━━━━━━━━━━━━ 100% 1,234 associations
  ✓ Complete (45s)

══════════════════════════════════════════════════════════════════
 STAGE 3: Publication Enrichment
══════════════════════════════════════════════════════════════════

Running 3 sources in parallel...

  [pub_metadata]    ━━━━━━━━━━━━━━━━━━━━ 100% 5,678 publications
  [retractions]     ━━━━━━━━━━━━━━━━━━━━ 100% 5,678 checked, 23 retracted
  [citations]       ━━━━━━━━━━━━━━━━━━━━ 100% 12,345 relationships

  ✓ Stage complete (4m 23s)

══════════════════════════════════════════════════════════════════
 STAGE 4: Post-Processing
══════════════════════════════════════════════════════════════════

[1/1] HPO Sibling Map
  Building map      ━━━━━━━━━━━━━━━━━━━━ 100% 8,901 phenotypes
  ✓ Saved to data/hpo_sibling_map.json (12s)

══════════════════════════════════════════════════════════════════
                         ✓ COMPLETE
══════════════════════════════════════════════════════════════════

Summary:
┌────────────────┬─────────────┬─────────────┬──────────┐
│ Source         │ Nodes       │ Edges       │ Time     │
├────────────────┼─────────────┼─────────────┼──────────┤
│ monarch        │ 1,234,567   │ 5,678,901   │ 2m 34s   │
│ hpo            │ -           │ 21,246      │ 15s      │
│ reactome       │ -           │ 23,456      │ 8s       │
│ hgnc           │ 43,000      │ -           │ 2s       │
│ cosmic         │ (skipped)   │ -           │ -        │
│ disgenet       │ 890         │ 1,234       │ 45s      │
│ pub_metadata   │ -           │ -           │ 3m 12s   │
│ retractions    │ -           │ -           │ 1m 45s   │
│ citations      │ -           │ 12,345      │ 2m 15s   │
│ hpo_siblings   │ -           │ -           │ 12s      │
├────────────────┼─────────────┼─────────────┼──────────┤
│ TOTAL          │ 1,278,457   │ 5,737,182   │ 11m 28s  │
└────────────────┴─────────────┴─────────────┴──────────┘

Skipped sources:
  • cosmic - Set COSMIC_EMAIL and COSMIC_PASSWORD in .env
```

---

## Extensibility

### Adding New Data Sources

To add a new data source (e.g., IntAct for protein interactions):

1. **Create source file**: `src/nerve/loader/sources/intact.py`

```python
from nerve.loader.protocol import DataSource, LoadStats
from nerve.loader.config import Config

class IntActSource(DataSource):
    """IntAct protein-protein interaction data."""
    
    name = "intact"
    display_name = "IntAct PPI"
    stage = 2  # External enrichment
    requires_credentials: list[str] = []  # No credentials needed
    dependencies = ["monarch"]  # Load after Monarch
    
    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        return True, None  # No credentials required
    
    def download(self, config: Config, force: bool = False) -> None:
        dest = config.data_dir / "intact"
        dest.mkdir(parents=True, exist_ok=True)
        
        if not force and (dest / "intact.txt").exists():
            return  # Already downloaded
        
        # Download IntAct data
        download_file(
            "https://ftp.ebi.ac.uk/pub/databases/intact/...",
            dest / "intact.txt"
        )
    
    def load(
        self, 
        driver: Neo4jDriver, 
        config: Config, 
        mode: Literal["replace", "merge"]
    ) -> LoadStats:
        # Load IntAct data into Neo4j
        ...
        return LoadStats(
            source=self.name,
            edges_created=count,
            duration_seconds=elapsed,
        )
```

2. **Register in `sources/__init__.py`**:

```python
from .intact import IntActSource

ALL_SOURCES: list[type[DataSource]] = [
    MonarchKGSource,
    HPOSource,
    ReactomeSource,
    HGNCSource,
    COSMICSource,
    DisGeNETSource,
    PublicationMetadataSource,
    RetractionStatusSource,
    CitationsSource,
    HPOSiblingMapSource,
    IntActSource,  # New source
]
```

3. **Update `.env.example`** if credentials needed

### Future Sources (from roadmap)

- **IntAct** - Protein-protein interactions
- **GO** - Gene Ontology annotations
- **Live edge fetcher** - `live_edges_for_gene(gene_id)` for Reactome/GO/IntAct

---

## Implementation Tasks

### Phase 1: Foundation (Priority: High)

| # | Task | File | Effort |
|---|------|------|--------|
| 1 | Create package structure | `src/nerve/loader/` | S |
| 2 | Create `.env.example` | `.env.example` | S |
| 3 | Implement `Config` dataclass with .env loading | `config.py` | S |
| 4 | Define `DataSource` protocol and `LoadStats` | `protocol.py` | S |
| 5 | Implement `RetryHandler` with exponential backoff | `retry.py` | S |
| 6 | Implement `SourceRegistry` with dependency resolution | `registry.py` | M |

### Phase 2: Core Sources - Stage 1 (Priority: High)

| # | Task | Effort |
|---|------|--------|
| 7 | Implement `MonarchKGSource` (extract from `load_monarch_to_neo4j.py`) | M |
| 8 | Implement `HPOSource` (extract from `load_monarch_to_neo4j.py`) | S |
| 9 | Implement `ReactomeSource` (extract from `load_monarch_to_neo4j.py`) | S |
| 10 | Implement `HGNCSource` (extract from `load_monarch_to_neo4j.py`) | S |

### Phase 3: Enrichment Sources - Stages 2-3 (Priority: Medium)

| # | Task | Effort |
|---|------|--------|
| 11 | Implement `COSMICSource` (wrap `download_cosmic_cgc.py`) | S |
| 12 | Implement `DisGeNETSource` (wrap `enrich_disgenet.py`) | S |
| 13 | Implement `PublicationMetadataSource` (wrap `enrich_publication_metadata.py`) | S |
| 14 | Implement `RetractionStatusSource` (wrap `enrich_retraction_status.py`) | S |
| 15 | Implement `CitationsSource` (wrap `enrich_citations.py`) | S |

### Phase 4: Orchestration & CLI (Priority: High)

| # | Task | Effort |
|---|------|--------|
| 16 | Implement `HPOSiblingMapSource` (wrap `build_hpo_sibling_map.py`) | S |
| 17 | Implement `StageExecutor` with parallel execution | M |
| 18 | Implement CLI with argparse in `__main__.py` | M |
| 19 | Add Rich progress bars and summary output | M |
| 20 | Implement `--dry-run` mode | S |

### Phase 5: Cleanup (Priority: Low)

| # | Task | Effort |
|---|------|--------|
| 21 | Remove old scripts from `scripts/` directory | S |
| 22 | Update `docs/todo.md` and `docs/roadmap.md` | S |
| 23 | Add `rich` to dependencies in `pyproject.toml` | S |
| 24 | Add basic tests for the loader | M |
| 25 | Update README with new loading instructions | S |

**Effort Key**: S = Small (~30 min), M = Medium (~1-2 hours)

---

## Scripts to Remove

After implementation, these scripts should be deleted (available in git history):

- `scripts/load_monarch_to_neo4j.py`
- `scripts/download_cosmic_cgc.py`
- `scripts/enrich_disgenet.py`
- `scripts/enrich_publication_metadata.py`
- `scripts/enrich_retraction_status.py`
- `scripts/enrich_citations.py`
- `scripts/build_hpo_sibling_map.py`

---

## Dependencies to Add

```toml
# pyproject.toml
[project]
dependencies = [
    # ... existing deps ...
    "rich>=13.0.0",  # Progress bars and rich console output
    "python-dotenv>=1.0.0",  # .env file loading
]
```

---

## Testing Strategy

1. **Unit tests**: Each source's `download()` and `load()` methods
2. **Integration test**: Full load with `--sample 100` against test Neo4j
3. **Dry-run test**: Verify `--dry-run` produces correct plan without side effects
4. **Retry test**: Mock network failures to verify retry behavior
5. **Parallel test**: Verify Stage 2/3 sources run concurrently

---

## Appendix: Existing Script Analysis

### Code Reuse Map

| Old Script | New Location | Reuse Strategy |
|------------|--------------|----------------|
| `load_monarch_to_neo4j.py` | `sources/monarch.py`, `sources/hpo.py`, `sources/reactome.py`, `sources/hgnc.py` | Extract functions, adapt to DataSource protocol |
| `download_cosmic_cgc.py` | `sources/cosmic.py` | Wrap with minimal changes |
| `enrich_disgenet.py` | `sources/disgenet.py` | Wrap with minimal changes |
| `enrich_publication_metadata.py` | `sources/publications.py` | Wrap, share NCBI utilities |
| `enrich_retraction_status.py` | `sources/publications.py` | Wrap, share NCBI utilities |
| `enrich_citations.py` | `sources/publications.py` | Wrap, share NCBI utilities |
| `build_hpo_sibling_map.py` | `sources/hpo_siblings.py` | Wrap with minimal changes |

### Shared Utilities to Extract

- **NCBI E-utilities client**: Used by pub_metadata, retractions, citations
- **Download helper with progress**: Used by monarch, cosmic
- **Neo4j batch writer**: Used by all loading sources
