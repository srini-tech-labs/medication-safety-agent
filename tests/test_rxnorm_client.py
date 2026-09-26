from unittest.mock import patch

from app.services.rxnorm_client import search_medications


def test_search_medications_rejects_short_query_without_network_call():
    with patch("app.services.rxnorm_client._rxnav_get") as mock_get:
        result = search_medications("li")

    mock_get.assert_not_called()
    assert result["matches"] == []
    assert "at least 3 characters" in result["message"]


@patch("app.services.rxnorm_client._rxnav_get")
def test_search_medications_prioritizes_prescribable_over_fallback(mock_get):
    mock_get.side_effect = [
        {"approximateGroup": {"candidate": [{"rxcui": "1", "name": "lisinopril", "rank": "1", "score": "9"}]}},
        {"approximateGroup": {"candidate": [{"rxcui": "2", "name": "lisinopril fallback", "rank": "1", "score": "9"}]}},
    ]

    result = search_medications("lisi", limit=10)

    assert result["matches"][0]["search_scope"] == "PRESCRIBABLE_RXNORM"
    assert result["matches"][0]["rxcui"] == "1"


@patch("app.services.rxnorm_client._rxnav_get")
def test_search_medications_caches_repeated_queries(mock_get):
    mock_get.return_value = {"approximateGroup": {"candidate": []}}

    search_medications("metformin", limit=5)
    result2 = search_medications("metformin", limit=5)

    assert result2["cache"] == "memory"
