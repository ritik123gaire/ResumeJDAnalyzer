from src.preprocessing import TextPreprocessor


def test_normalize_text_basic():
    text = "Email me at a@b.com and visit https://example.com"
    normalized = TextPreprocessor.normalize_text(text)
    assert "a@b.com" not in normalized
    assert "https://" not in normalized
