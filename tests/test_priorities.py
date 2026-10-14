"""Transparent deterministic evidence-derived scoring, with no need estimates."""
from copy import deepcopy
from types import SimpleNamespace

from src.graph.chokepoints import analyze_chokepoints
from src.graph.priorities import rank_priorities
from test_road_graph import bundle, flood, pipeline


def test_status_order_weights_breakdown_and_value_only_explanations():
    _, _, network, isolation = pipeline()
    template = isolation.settlements[0]
    rows = []
    for identifier, status, access in (("a", "CUT OFF", "none"), ("b", "UNCERTAIN", "full"),
                                      ("c", "CONNECTED", "degraded"), ("d", "CONNECTED", "full")):
        row = deepcopy(template)
        row.update(osm_id=identifier, status=status, access_detail=access,
                   cut_from=["hospital", "town"] if status == "CUT OFF" else [],
                   detour_factors={"hospital": 2. if access == "degraded" else 1.})
        if status == "UNCERTAIN":
            row["confidence_tier"] = 1
            row["evidence"][0]["tier"] = 1
        rows.append(row)
    result = rank_priorities(SimpleNamespace(settlements=rows), SimpleNamespace(damaged=[], latent=[]))
    assert [row["id"] for row in result["settlements"]] == ["a", "b", "c"]
    for row in result["settlements"]:
        assert row["score"] == sum(value["contribution"] for value in row["components"].values())
        assert all(str(row["components"][key]["value"]) in why for key, why in zip(row["components"], row["why"]))
        assert not any(word in " ".join(row["why"]).lower() for word in ("casualty", "population"))


def test_link_ranking_and_stable_ties():
    _, _, network, isolation = pipeline()
    chokepoints = analyze_chokepoints(network, isolation)
    result = rank_priorities(isolation, chokepoints)
    assert result["links"][0]["components"]["settlements"]["value"] == 2
    assert result["links"][0]["confidence"] == "strong"
    assert [row["id"] for row in result["settlements"]] == sorted(row["id"] for row in result["settlements"])


def test_baseline_only_and_unsnapped_uncertainty_are_data_gaps():
    _, _, _, isolation = pipeline(osm=bundle(hospitals=[], towns=False), fm=flood(empty=True))
    result = rank_priorities(isolation, SimpleNamespace(damaged=[], latent=[]))
    assert result["settlements"] == []
    assert result["message"] == "No flood-attributable priorities"
    assert [row["id"] for row in result["data_gaps"]] == sorted(row["osm_id"] for row in isolation.settlements)
    assert all("no baseline road access" in row["reason"] for row in result["data_gaps"])
    _, _, _, unsnapped = pipeline(osm=bundle([], hospitals=[], towns=False), fm=flood(empty=True))
    result = rank_priorities(unsnapped, SimpleNamespace(damaged=[], latent=[]))
    assert not result["settlements"] and len(result["data_gaps"]) == 2
    assert all(not row["snapped"] and row["reason"] for row in result["data_gaps"])


def test_baseline_uncertainty_with_noncausal_hit_is_not_ranked():
    _, _, _, isolation = pipeline()
    row = deepcopy(isolation.settlements[0])
    row.update(status="UNCERTAIN", cut_from=[], possible_cut_from=[], access_detail="unmapped",
               access_by_class={"hospital": "unmapped", "town": "full", "clinic": "unmapped"},
               baseline_connected={"hospital": False, "town": True}, reason="no baseline road access in OSM")
    result = rank_priorities(SimpleNamespace(settlements=[row]), SimpleNamespace(damaged=[], latent=[]))
    assert not result["settlements"] and len(result["data_gaps"]) == 1


def test_tier_one_hit_with_established_baseline_is_ranked():
    _, _, _, isolation = pipeline(fm=flood(tier=1))
    result = rank_priorities(isolation, SimpleNamespace(damaged=[], latent=[]))
    assert len(result["settlements"]) == 2 and not result["data_gaps"]
    assert all(row["confidence"] == "uncertain" for row in result["settlements"])
    assert result["message"] is None


def test_degraded_access_requires_responsible_hit_evidence():
    _, _, _, isolation = pipeline()
    row = deepcopy(isolation.settlements[0])
    row.update(status="CONNECTED", cut_from=[], possible_cut_from=[], access_detail="degraded",
               detour_factors={"hospital": 2.})
    result = rank_priorities(SimpleNamespace(settlements=[row]), SimpleNamespace(damaged=[], latent=[]))
    assert len(result["settlements"]) == 1
    row["evidence"] = []
    assert not rank_priorities(SimpleNamespace(settlements=[row]), SimpleNamespace(damaged=[], latent=[]))["settlements"]


def test_zero_leverage_damaged_links_are_separate_unranked_and_deterministic():
    _, _, network, isolation = pipeline()
    chokepoints = analyze_chokepoints(network, isolation)
    zero = deepcopy(chokepoints.damaged[0])
    zero.update(link_id="link_z", count=0, sentence=None)
    other = dict(zero, link_id="link_a")
    chokepoints.damaged.extend([zero, other])
    result = rank_priorities(isolation, chokepoints)
    assert len(result["links"]) == 1
    assert [row["link_id"] for row in result["damaged_no_leverage"]] == ["link_a", "link_z"]
    assert all("rank" not in row and "score" not in row and "geometry" not in row for row in result["damaged_no_leverage"])
