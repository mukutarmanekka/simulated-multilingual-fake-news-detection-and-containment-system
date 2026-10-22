"""Phase 0: config constants and seed derivation are deterministic."""

from verifacto import config


def test_block_sizes_sum_to_500_users():
    assert sum(config.BLOCK_SIZES) == 500


def test_cascade_seed_is_deterministic():
    first = config.cascade_seed(3, 0)
    second = config.cascade_seed(3, 0)
    assert first == second


def test_cascade_seed_varies_across_posts_and_runs():
    seeds = {
        config.cascade_seed(post_id, run)
        for post_id in range(5)
        for run in range(5)
    }
    assert len(seeds) == 25


def test_greedy_seed_is_deterministic_and_disjoint():
    assert config.greedy_seed(3, 0) == config.greedy_seed(3, 0)
    # selection draws must never coincide with evaluation draws
    for post_id in range(5):
        for run in range(5):
            assert config.greedy_seed(post_id, run) != config.cascade_seed(post_id, run)


def test_paths_point_inside_repo():
    assert config.REPO_ROOT.is_dir()
    assert config.PIPELINE_DIR.is_dir()
    assert config.GRAPH_PATH.parent == config.DATA_DIR


def test_detection_policy_values_are_safe_defaults():
    assert config.DETECTOR_ERROR_POLICY in {"record", "raise"}
    assert config.UNCERTAIN_POLICY in {"no_action", "review"}
    assert config.DETECTOR_TIMEOUT_SECONDS > 0
    assert config.DETECTOR_MAX_ATTEMPTS >= 1
    assert config.DEFAULT_LIVE_MAX_POSTS >= 1
    assert config.DETECTOR_POLICY_VERSION
