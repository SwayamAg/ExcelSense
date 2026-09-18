"""Natural-language query understanding and PS routing."""
from .entities import Entity, EntityExtractor, ExtractionResult
from .matcher import Candidate, PSMatcher
from .normalizer import NormalizedQuery, find_unsupported, normalize
from .router import (
    AMBIGUOUS,
    CONFIDENCE_FLOOR,
    REFUSED_INVALID_ENTITY,
    REFUSED_LOW_CONFIDENCE,
    REFUSED_UNSUPPORTED,
    ROUTED,
    QueryRouter,
    RoutingDecision,
    get_router,
)

__all__ = [
    "Entity", "EntityExtractor", "ExtractionResult",
    "Candidate", "PSMatcher",
    "NormalizedQuery", "find_unsupported", "normalize",
    "QueryRouter", "RoutingDecision", "get_router",
    "ROUTED", "AMBIGUOUS", "REFUSED_UNSUPPORTED", "REFUSED_LOW_CONFIDENCE",
    "REFUSED_INVALID_ENTITY", "CONFIDENCE_FLOOR",
]
