"""Tests for extended player features (position and starter)."""

import pytest

from src import db
from src.config import DB_PATH
from src.features.player_features import compute_player_features


@pytest.fixture
def connection():
    """Create a database connection for testing."""
    conn = db.get_connection(DB_PATH)
    yield conn
    conn.close()


def test_player_features_with_position_fields(connection):
    """Test that player features include position and starter fields."""
    df = compute_player_features(connection, 2024)

    assert not df.empty, "Player features should not be empty"

    # Check new position fields exist
    assert "jumper_number" in df.columns
    assert "position_code" in df.columns
    assert "position_label" in df.columns
    assert "position_group" in df.columns
    assert "is_starter" in df.columns

    # Validate jumper_number range (should be 1-25 typically)
    assert df["jumper_number"].between(1, 30).all()

    # Validate position_code values
    valid_codes = ["FB", "WG", "CE", "FE", "HB", "PR", "HK", "SR", "LK", "INT", "RES"]
    assert df["position_code"].isin(valid_codes).all()

    # Validate position_group values
    valid_groups = ["Back", "Halfback", "Hooker", "Forward", "Interchange", "Reserve"]
    assert df["position_group"].isin(valid_groups).all()

    # Validate is_starter is binary
    assert df["is_starter"].isin([0, 1]).all()

    # Check that jersey 1-13 are starters
    starters = df[df["jumper_number"] <= 13]
    assert (starters["is_starter"] == 1).all()

    # Check that jersey 14+ are not starters
    bench = df[df["jumper_number"] >= 14]
    assert (bench["is_starter"] == 0).all()


def test_position_mapping_consistency(connection):
    """Test that position mappings are consistent."""
    df = compute_player_features(connection, 2024)

    # Fullback (jersey 1) should be "Back"
    fb = df[df["jumper_number"] == 1]
    if not fb.empty:
        assert (fb["position_code"] == "FB").all()
        assert (fb["position_label"] == "Fullback").all()
        assert (fb["position_group"] == "Back").all()

    # Halfback (jersey 7) should be "Halfback"
    hb = df[df["jumper_number"] == 7]
    if not hb.empty:
        assert (hb["position_code"] == "HB").all()
        assert (hb["position_label"] == "Halfback").all()
        assert (hb["position_group"] == "Halfback").all()

    # Hooker (jersey 9) should be "Hooker"
    hk = df[df["jumper_number"] == 9]
    if not hk.empty:
        assert (hk["position_code"] == "HK").all()
        assert (hk["position_label"] == "Hooker").all()
        assert (hk["position_group"] == "Hooker").all()

    # Prop (jersey 8) should be "Forward"
    pr = df[df["jumper_number"] == 8]
    if not pr.empty:
        assert (pr["position_code"] == "PR").all()
        assert (pr["position_label"] == "Prop").all()
        assert (pr["position_group"] == "Forward").all()

    # Wing (jersey 2) should be "Back"
    wg = df[df["jumper_number"] == 2]
    if not wg.empty:
        assert (wg["position_code"] == "WG").all()
        assert (wg["position_label"] == "Wing").all()
        assert (wg["position_group"] == "Back").all()

    # Interchange (jersey 14) should be "Interchange"
    ic = df[df["jumper_number"] == 14]
    if not ic.empty:
        assert (ic["position_code"] == "INT").all()
        assert (ic["position_label"] == "Interchange").all()
        assert (ic["position_group"] == "Interchange").all()


def test_position_group_distribution(connection):
    """Test that position group distribution is reasonable."""
    df = compute_player_features(connection, 2024)

    # Count observations by position group
    group_counts = df["position_group"].value_counts()

    # Should have multiple position groups represented
    assert len(group_counts) >= 4, "Should have at least 4 position groups"

    # Backs should be common (jerseys 1-5)
    assert "Back" in group_counts.index
    assert group_counts["Back"] > 100  # Should have many back observations

    # Forwards should be common (jerseys 8,10,11,12,13)
    assert "Forward" in group_counts.index
    assert group_counts["Forward"] > 100

    # Halfbacks should be present (jerseys 6,7)
    assert "Halfback" in group_counts.index

    # Interchange should be present (jerseys 14-17)
    assert "Interchange" in group_counts.index
