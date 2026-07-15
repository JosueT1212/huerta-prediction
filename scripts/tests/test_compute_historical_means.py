import pandas as pd


def test_historical_means_by_position_averages_first_n_rows_per_season():
    from scripts.compute_historical_means import historical_means_by_position

    df = pd.DataFrame({
        "temporada": ["T13", "T13", "T13", "T14", "T14", "T14"],
        "kg_reales": [10.0, 20.0, 30.0, 30.0, 40.0, 50.0],
    })
    means = historical_means_by_position(df, n_positions=3)
    assert means == {0: 20.0, 1: 30.0, 2: 40.0}


def test_historical_means_by_position_ignores_rows_past_n():
    from scripts.compute_historical_means import historical_means_by_position

    df = pd.DataFrame({
        "temporada": ["T13"] * 5,
        "kg_reales": [1.0, 2.0, 3.0, 999.0, 999.0],
    })
    means = historical_means_by_position(df, n_positions=3)
    assert means == {0: 1.0, 1: 2.0, 2: 3.0}


def test_n_positions_is_four():
    from scripts.compute_historical_means import N_POSITIONS
    assert N_POSITIONS == 4


def test_historical_means_by_position_covers_four_weeks_by_default():
    from scripts.compute_historical_means import historical_means_by_position

    df = pd.DataFrame({
        "temporada": ["T13"] * 5,
        "kg_reales": [1.0, 2.0, 3.0, 4.0, 999.0],
    })
    means = historical_means_by_position(df)  # uses module default N_POSITIONS
    assert means == {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
