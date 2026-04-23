from src.gap_analysis import GapAnalyzer


def test_infers_required_terms_not_in_skill_db():
    analyzer = GapAnalyzer()
    resume_skills = {"technical": ["python"], "soft": [], "certifications": []}
    jd_skills = {"technical": ["python"], "soft": [], "certifications": []}
    jd_text = "Required experience with semiconductor process control and metrology systems."
    resume_text = "Python automation for data pipelines."

    gaps = analyzer.analyze(resume_skills, jd_skills, jd_text, resume_text, {"experience": resume_text})
    names = {g.skill for g in gaps}

    assert "semiconductor process control" in names
    assert "metrology systems" in names


def test_does_not_duplicate_existing_skill_gap():
    analyzer = GapAnalyzer()
    resume_skills = {"technical": [], "soft": [], "certifications": []}
    jd_skills = {"technical": ["python"], "soft": [], "certifications": []}
    jd_text = "Python is required. Required python for automation."
    resume_text = "No coding tools listed."

    gaps = analyzer.analyze(resume_skills, jd_skills, jd_text, resume_text, {"experience": resume_text})
    python_gaps = [g for g in gaps if g.skill == "python"]

    assert len(python_gaps) == 1


def test_filters_noisy_requirement_phrases():
    analyzer = GapAnalyzer()
    resume_skills = {"technical": ["python"], "soft": [], "certifications": []}
    jd_skills = {"technical": ["communication"], "soft": ["communication"], "certifications": []}
    jd_text = (
        "Minimum qualifications: currently enrolled in a bachelor's or master's degree program. "
        "Must be in your senior year of undergraduate studies or pursuing a graduate degree. "
        "Preferred familiarity with automotive communication protocols (can, ethernet)."
    )
    resume_text = "Python projects with APIs."

    gaps = analyzer.analyze(resume_skills, jd_skills, jd_text, resume_text, {"experience": resume_text})
    names = {g.skill for g in gaps}

    assert "communication" in names
    assert "can" in names
    assert "ethernet" in names
    assert "automotive communication protocols" not in names
    assert "qualifications" not in names
    assert "pursuing a graduate degree" not in names
    assert "ethernet)" not in names


def test_filters_work_authorization_tokens():
    analyzer = GapAnalyzer()
    resume_skills = {"technical": [], "soft": [], "certifications": []}
    jd_skills = {"technical": [], "soft": [], "certifications": []}
    jd_text = (
        "Work authorization for employment in the United States is required (CPT/OPT with 2-year STEM extension is accepted). "
        "If your position is employed by Airbnb, Inc., your recruiter will inform you."
    )
    resume_text = "Research assistant with Python and PyTorch."

    gaps = analyzer.analyze(resume_skills, jd_skills, jd_text, resume_text, {"experience": resume_text})
    names = {g.skill for g in gaps}

    assert "cpt" not in names
    assert "opt" not in names
    assert "inc" not in names
