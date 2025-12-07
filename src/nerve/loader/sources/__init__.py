"""Data source implementations for the unified loader.

Each source implements the DataSource protocol and handles downloading
and loading data from a specific biomedical data source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nerve.loader.protocol import DataSource

# Import all source classes
from nerve.loader.sources.monarch import MonarchKGSource
from nerve.loader.sources.hpo import HPOSource
from nerve.loader.sources.reactome import ReactomeSource
from nerve.loader.sources.hgnc import HGNCSource
from nerve.loader.sources.cosmic import COSMICSource
from nerve.loader.sources.disgenet import DisGeNETSource
from nerve.loader.sources.publications import (
    PublicationMetadataSource,
    RetractionStatusSource,
    CitationsSource,
)
from nerve.loader.sources.hpo_siblings import HPOSiblingMapSource

# All available sources in registration order
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
]

__all__ = [
    "ALL_SOURCES",
    "MonarchKGSource",
    "HPOSource",
    "ReactomeSource",
    "HGNCSource",
    "COSMICSource",
    "DisGeNETSource",
    "PublicationMetadataSource",
    "RetractionStatusSource",
    "CitationsSource",
    "HPOSiblingMapSource",
]
