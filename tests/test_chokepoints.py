"""Single-link reconnection leverage and undamaged bridge risks."""
from src.graph.chokepoints import analyze_chokepoints
from test_road_graph import pipeline, flood


def test_damaged_restoration_reconnects_two_subjects():
    _, _, network, isolation = pipeline()
    result = analyze_chokepoints(network, isolation)
    assert len(result.damaged) == 1
    row = result.damaged[0]
    assert row["count"] == row["hospital_access_reconnected"] == row["town_access_reconnected"] == 2
    assert row["mapped_buildings"] == 1 and row["damage_tier"] == 3
    assert "2 settlements" in row["sentence"]


def test_latent_bridge_and_work_cap():
    _, _, network, isolation = pipeline(fm=flood(empty=True))
    result = analyze_chokepoints(network, isolation)
    assert len(result.latent) == 1 and result.latent[0]["count"] == 2
    assert result.latent[0]["hospital_access_lost"] == 2
    limited = analyze_chokepoints(network, isolation, latent_limit=0)
    assert not limited.latent and limited.flags[0]["code"] == "latent_bridge_work_cap"


def test_zero_leverage_link_has_no_sentence():
    _, _, network, isolation = pipeline()
    network.subjects.clear()
    isolation.settlements.clear()
    result = analyze_chokepoints(network, isolation)
    assert result.damaged[0]["count"] == 0 and result.damaged[0]["sentence"] is None
