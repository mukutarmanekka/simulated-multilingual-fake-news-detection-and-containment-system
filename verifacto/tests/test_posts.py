"""Phase 2: the post corpus is balanced, valid, and deterministic."""

from verifacto import config
from verifacto.data.posts import Post, generate_posts, load_posts, save_posts


def test_count_and_fields(graph):
    posts = generate_posts(graph)
    assert len(posts) == config.N_POSTS
    languages_seen = set()
    for post in posts:
        assert isinstance(post, Post)
        assert post.text.strip()
        assert post.true_label in {"fake", "real"}
        assert isinstance(post.author, int)
        assert post.language in config.LANGUAGES
        languages_seen.add(post.language)
    # every language block has posts
    assert languages_seen == set(config.LANGUAGES)


def test_author_in_language_block(graph):
    posts = generate_posts(graph)
    for post in posts:
        assert post.author in graph.nodes
        assert graph.nodes[post.author]["language"] == post.language


def test_ratio_explicit(graph):
    posts = generate_posts(graph)
    n_real = sum(1 for p in posts if p.true_label == "real")
    n_fake = sum(1 for p in posts if p.true_label == "fake")
    assert n_real == 40
    assert n_fake == 40


def test_no_duplicate_texts(graph):
    posts = generate_posts(graph)
    texts = [p.text for p in posts]
    assert len(set(texts)) == len(texts)


def test_unique_ids_and_deterministic(graph):
    first = generate_posts(graph)
    second = generate_posts(graph)
    ids = [p.id for p in first]
    assert len(set(ids)) == len(ids)
    for a, b in zip(first, second):
        assert a.id == b.id
        assert a.text == b.text
        assert a.true_label == b.true_label
        assert a.author == b.author
        assert a.language == b.language


def test_save_load_roundtrip(graph, tmp_path):
    posts = generate_posts(graph)
    path = tmp_path / "posts.json"
    save_posts(posts, path)
    loaded = load_posts(path)
    assert len(loaded) == len(posts)
    for a, b in zip(posts, loaded):
        assert a == b
