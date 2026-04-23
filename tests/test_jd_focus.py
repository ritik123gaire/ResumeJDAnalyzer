from src.jd_focus import focus_job_description


def test_focus_trims_footer():
    jd = """
intro corporate text here
job description

need python and sql for analytics work. build pipelines, dashboards, and etl jobs with stakeholders.
document assumptions, validate data quality, and ship reproducible notebooks for cross-functional teams.

base pay range: 100
primary location: earth
""".strip().lower()
    focused, meta = focus_job_description(jd, min_chars=20)
    assert "python" in focused
    assert "base pay" not in focused
    assert meta.get("trim_applied") is True


def test_focus_trims_inclusion_and_pay_blocks():
    jd = """
key responsibilities include:
build active learning pipelines for image labeling.
analyze model performance and iterate on data quality.

our commitment to inclusion & belonging:
airbnb is committed to a diverse talent pool.

pay range
$4,500 biweekly
""".strip().lower()
    focused, meta = focus_job_description(jd, min_chars=20)
    assert "active learning" in focused
    assert "commitment to inclusion" not in focused
    assert "pay range" not in focused
    assert meta.get("trim_applied") is True
