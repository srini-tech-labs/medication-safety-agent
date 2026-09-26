"""RxNorm/RxTerms medication search and detail lookup.

Ported from the retired medication-lookup-api notebook (see
docs/databricks-endpoint-lifecycle.md) — this is a pure public-API proxy
with no Databricks dependency, so it runs directly in FastAPI.
"""

import time

import httpx

from app.config import settings

_RXNAV_BASE = "https://rxnav.nlm.nih.gov"
_SEARCH_CACHE_TTL_SECONDS = 60 * 60
_search_cache: dict[tuple[str, int], dict] = {}


class RxNormError(Exception):
    """Raised when an RxNorm/RxTerms lookup fails."""


def _rxnav_get(path: str, params: dict | None = None, timeout: float = 15) -> dict:
    try:
        response = httpx.get(f"{_RXNAV_BASE}{path}", params=params or {}, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise RxNormError(f"RxNorm request to {path} failed: {exc}") from exc


def search_medications(query: str, limit: int | None = None) -> dict:
    query = (query or "").strip()
    if len(query) < settings.rxnorm_min_search_characters:
        return {
            "query": query,
            "minimum_characters": settings.rxnorm_min_search_characters,
            "matches": [],
            "message": f"Enter at least {settings.rxnorm_min_search_characters} characters.",
        }

    limit = max(1, min(int(limit or settings.rxnorm_default_search_limit), 25))
    cache_key = (query.lower(), limit)
    cached = _search_cache.get(cache_key)
    if cached and cached["expires_at"] > time.time():
        result = dict(cached["value"])
        result["cache"] = "memory"
        return result

    matches_by_rxcui: dict[str, dict] = {}

    prescribable = _rxnav_get(
        "/REST/Prescribe/approximateTerm.json",
        params={"term": query, "maxEntries": min(limit * 3, 100), "option": 1},
    )
    for candidate in prescribable.get("approximateGroup", {}).get("candidate", []) or []:
        rxcui, name = candidate.get("rxcui"), candidate.get("name")
        if not rxcui or not name:
            continue
        matches_by_rxcui[rxcui] = {
            "rxcui": rxcui,
            "name": name,
            "rank": int(candidate.get("rank", 9999)),
            "score": float(candidate.get("score", 0)),
            "search_scope": "PRESCRIBABLE_RXNORM",
        }

    if len(matches_by_rxcui) < limit:
        general = _rxnav_get(
            "/REST/approximateTerm.json",
            params={"term": query, "maxEntries": min(limit * 3, 100), "option": 1},
        )
        for candidate in general.get("approximateGroup", {}).get("candidate", []) or []:
            rxcui, name = candidate.get("rxcui"), candidate.get("name")
            if not rxcui or not name or rxcui in matches_by_rxcui:
                continue
            matches_by_rxcui[rxcui] = {
                "rxcui": rxcui,
                "name": name,
                "rank": int(candidate.get("rank", 9999)),
                "score": float(candidate.get("score", 0)),
                "search_scope": "RXNORM_FALLBACK",
            }

    priority = {"PRESCRIBABLE_RXNORM": 0, "RXNORM_FALLBACK": 1}
    matches = sorted(
        matches_by_rxcui.values(),
        key=lambda item: (priority.get(item["search_scope"], 9), item["rank"], -item["score"], item["name"].lower()),
    )[:limit]

    result = {
        "query": query,
        "matches": matches,
        "count": len(matches),
        "cache": "miss",
        "source": "NLM RxNorm / Prescribable RxNorm",
    }
    _search_cache[cache_key] = {"value": result, "expires_at": time.time() + _SEARCH_CACHE_TTL_SECONDS}
    return result


def fetch_medication_details(rxcui: str) -> dict:
    """Live RxNorm/RxTerms enrichment for one rxcui (no durable cache here — see medication_management.py).

    Ported field-for-field from the retired medication-lookup-api notebook's
    get_medication_details, including its 3-step ingredient resolution
    (historystatus derivedConcepts -> IN/PIN self -> related.json fallback).
    """
    rxcui = str(rxcui or "").strip()
    if not rxcui:
        raise RxNormError("rxcui is required.")

    properties = _rxnav_get(f"/REST/rxcui/{rxcui}/properties.json").get("properties") or {}
    rxnorm_name = properties.get("name") or ""
    rxnorm_tty = properties.get("tty") or ""

    rxterms = _rxnav_get(f"/REST/RxTerms/rxcui/{rxcui}/allinfo.json").get("rxtermsProperties") or {}

    history = _rxnav_get(f"/REST/rxcui/{rxcui}/historystatus.json")
    derived = history.get("rxcuiStatusHistory", {}).get("derivedConcepts", {}) or {}
    ingredient_concepts = derived.get("ingredientConcept") or []

    ingredients = []
    for ingredient in ingredient_concepts:
        ingredient_rxcui = ingredient.get("ingredientRxcui")
        ingredient_name = ingredient.get("ingredientName")
        if ingredient_rxcui and ingredient_name:
            ingredients.append({"rxcui": str(ingredient_rxcui), "name": ingredient_name})

    if not ingredients and rxnorm_tty in {"IN", "PIN"}:
        ingredients = [{"rxcui": rxcui, "name": rxnorm_name}]

    if not ingredients:
        related = _rxnav_get(f"/REST/rxcui/{rxcui}/related.json", params={"tty": "IN"})
        for group in related.get("relatedGroup", {}).get("conceptGroup", []) or []:
            for concept in group.get("conceptProperties") or []:
                if concept.get("rxcui") and concept.get("name"):
                    ingredients.append({"rxcui": str(concept["rxcui"]), "name": concept["name"]})

    ingredients = list({item["rxcui"]: item for item in ingredients}.values())
    primary = ingredients[0] if ingredients else {}
    full_name = rxterms.get("fullName") or rxnorm_name

    return {
        "selected_rxcui": rxcui,
        "selected_name": full_name,
        "rxnorm_name": rxnorm_name,
        "full_name": full_name,
        "full_generic_name": rxterms.get("fullGenericName"),
        "strength": rxterms.get("strength"),
        "route": rxterms.get("route"),
        "rxterms_dose_form": rxterms.get("rxtermsDoseForm"),
        "rxnorm_dose_form": rxterms.get("rxnormDoseForm"),
        "term_type": rxterms.get("termType") or rxnorm_tty,
        "generic_rxcui": rxterms.get("genericRxcui"),
        "primary_ingredient_rxcui": primary.get("rxcui"),
        "primary_ingredient_name": primary.get("name"),
        "ingredients": ingredients,
        # RxNorm/RxTerms alone do not reliably tell us whether the selected
        # product is OTC or Rx-only.
        "medication_type": "UNKNOWN",
        "source": "NLM RxNorm + RxTerms",
    }


def rxnorm_version() -> dict:
    return _rxnav_get("/REST/version.json")
