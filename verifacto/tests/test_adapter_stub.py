"""Phase 4: the stub detector is deterministic and label-free at runtime."""

import inspect

from verifacto import config
from verifacto.data.posts import Post
from verifacto.detector.stub import StubDetector


def test_deterministic(posts):
    first = StubDetector(posts)
    second = StubDetector(posts)
    for post in posts:
        a = first.detect(post.text, post.language, post.id)
        b = second.detect(post.text, post.language, post.id)
        assert a == b


def test_no_label_parameter():
    signature = inspect.signature(StubDetector.detect)
    assert list(signature.parameters) == ["self", "text", "language", "post_id"]


def test_label_independent_at_runtime(posts):
    detector = StubDetector(posts)
    before = {
        post.id: detector.detect(post.text, post.language, post.id) for post in posts
    }
    # flip the corpus labels AFTER construction: outputs must not change
    flipped = [
        Post(
            id=post.id,
            text=post.text,
            true_label="fake" if post.true_label == "real" else "real",
            author=post.author,
            language=post.language,
        )
        for post in posts
    ]
    after = {
        post.id: detector.detect(post.text, post.language, post.id) for post in flipped
    }
    assert before == after


def test_error_rates_approx(posts):
    detector = StubDetector(posts)
    n_fake = 0
    n_real = 0
    flagged_fake = 0
    flagged_real = 0
    for post in posts:
        result = detector.detect(post.text, post.language, post.id)
        if post.true_label == "fake":
            n_fake += 1
            if result.is_fake is True:
                flagged_fake += 1
        else:
            n_real += 1
            if result.is_fake is True:
                flagged_real += 1
    sensitivity = flagged_fake / n_fake
    false_positive_rate = flagged_real / n_real
    # seeded, so these are exact; the margin documents the intent
    assert abs(sensitivity - config.STUB_SENSITIVITY) < 0.15
    assert abs(false_positive_rate - config.STUB_FALSE_POSITIVE_RATE) < 0.15
