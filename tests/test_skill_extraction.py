from src.skill_extraction import SkillExtractor


def test_skill_alias_and_detection():
    extractor = SkillExtractor()
    result = extractor.extract("Experienced in JS, Python, SQL, and teamwork")
    assert "javascript" in result.technical
    assert "python" in result.technical
    assert "sql" in result.technical
    assert "teamwork" in result.soft
