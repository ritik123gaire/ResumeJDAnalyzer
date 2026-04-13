from src.matching.tfidf_matcher import TFIDFMatcher


def test_tfidf_score_range():
    matcher = TFIDFMatcher()
    result = matcher.score("python sql tableau", "python sql")
    assert 0.0 <= result.score <= 1.0
