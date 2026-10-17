"""Phase 6: rank flagged posts by expected harm.

When the intervention budget is limited, act on the most harmful posts
first:

    priority = detector_confidence x expected_reach

where expected_reach is the mean cascade size from the post's author
(Monte Carlo over the standard seeded runs, no intervention).
"""

from ..config import cascade_seed
from ..simulation.cascade import run_ic_cascade


def expected_reach(G, post, eval_runs):
    total = 0
    for run in range(eval_runs):
        seed = cascade_seed(post.id, run)
        series = run_ic_cascade(G, post.author, seed=seed)
        total += sum(series)
    return total / eval_runs


def rank_flagged_posts(G, posts, detections, eval_runs):
    """Ranked flagged posts: (post_id, priority, confidence, expected_reach).

    Only posts the detector flagged fake are ranked — those are the ones
    an intervention would act on.
    """
    ranked = []
    for post in posts:
        detection = detections[post.id]
        if detection.is_fake is not True:
            continue
        reach = expected_reach(G, post, eval_runs)
        priority = detection.confidence * reach
        ranked.append((post.id, priority, detection.confidence, reach))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked
