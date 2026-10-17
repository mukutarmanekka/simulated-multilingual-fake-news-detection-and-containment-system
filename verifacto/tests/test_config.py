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


def test_paths_point_inside_repo():
    assert config.REPO_ROOT.is_dir()
    assert config.PIPELINE_DIR.is_dir()
    assert config.GRAPH_PATH.parent == config.DATA_DIR
