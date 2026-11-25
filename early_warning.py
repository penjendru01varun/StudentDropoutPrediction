"""
EduGuard Early Warning & Intervention System (Phase 7).

Provides configurable prototype risk stratification, risk-driver-linked
decision-support recommendations, batch CSV scoring, and PDF dossier export.

NOTE: Risk thresholds (Critical >= 90%, High 70-89%, Medium 40-69%, Low < 40%)
are configurable prototype thresholds for educator decision support, not
clinically or universally validated cutoffs.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fpdf import FPDF


DEFAULT_RISK_THRESHOLDS = {
    "critical": 0.90,
    "high": 0.70,
    "medium": 0.40,
}


def classify_risk_level(
    prob: float,
    thresholds: Dict[str, float] | None = None,
) -> str:
    """Stratify dropout probability into 4 configurable prototype tiers."""
    t = thresholds or DEFAULT_RISK_THRESHOLDS
    if prob >= t["critical"]:
        return "Critical"
    if prob >= t["high"]:
        return "High"
    if prob >= t["medium"]:
        return "Medium"
    return "Low"


def generate_intervention_plan(
    student_row: Dict[str, Any],
    risk_proba: float,
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Generate decision-support recommendations grounded in observed student
    risk factors (academic, financial, engagement).
    """
    level = classify_risk_level(risk_proba, thresholds)
    plan: Dict[str, Any] = {
        "risk_level": level,
        "tier_label": f"{level} Risk ({risk_proba:.1%})",
        "academic": [],
        "financial": [],
        "engagement": [],
        "advisory_note": (
            "Decision-support output only. Educators should review contextual "
            "records before taking institutional action."
        ),
    }

    if level == "Low":
        plan["academic"].append(
            "Routine monitoring: academic and financial indicators are within normal bounds."
        )
        return plan

    def _val(key: str, default: float = 0.0) -> float:
        alt = key.replace("_", " ")
        v = student_row.get(key, student_row.get(alt, default))
        try:
            return float(v)
        except Exception:
            return default

    ccr1 = _val("Credit_Completion_Rate_1st_Sem", 1.0)
    ccr2 = _val("Credit_Completion_Rate_2nd_Sem", 1.0)
    grade_drop = _val("Grade_Drop", 0.0)
    stress = _val("Academic_Stress_Index", 0.0)
    engagement = _val("Engagement_Score", 1.0)
    debtor = _val("Debtor", 0.0)
    tuition = _val("Tuition_fees_up_to_date", 1.0)
    scholarship = _val("Scholarship_holder", 0.0)
    displaced = _val("Displaced", 0.0)

    # Academic risk factors
    if ccr1 < 0.6 or ccr2 < 0.6:
        plan["academic"].append(
            "Targeted Tutoring & Academic Coaching: Low credit completion rate detected; "
            "recommend foundational subject tutoring and study-skills advising."
        )
    if grade_drop > 1.5:
        plan["academic"].append(
            "Mid-Program Academic Counseling: Significant semester-to-semester grade drop "
            f"({grade_drop:.2f} pts); schedule diagnostic syllabus review."
        )
    if stress > 3.0:
        plan["academic"].append(
            "Course-Load Advising: High unapproved credit backlog "
            f"(Stress Index: {stress:.1f}); consider workload pacing."
        )

    # Financial risk factors
    if debtor == 1.0 or tuition == 0.0:
        plan["financial"].append(
            "Financial Assistance & Payment Plan Guidance: Unsettled institutional debt or "
            "overdue tuition flagged; connect student with Financial Aid office."
        )
    if scholarship == 0.0 and risk_proba >= 0.70:
        plan["financial"].append(
            "Scholarship & Bursary Screening: No active scholarship recorded; provide "
            "information on need-based grants or campus work-study options."
        )

    # Engagement & transition risk factors
    if engagement < 0.75:
        plan["engagement"].append(
            "Faculty Mentoring & Engagement Outreach: Below-target evaluation participation "
            f"(Engagement Score: {engagement:.2f}); schedule advisor check-in."
        )
    if displaced == 1.0:
        plan["engagement"].append(
            "Campus Integration & Peer Mentoring: Student is relocated/commuting; "
            "offer peer-mentorship and student engagement activities."
        )

    if not plan["academic"] and not plan["financial"] and not plan["engagement"]:
        plan["academic"].append(
            "Proactive Advisor Check-In: Elevated model risk score; review local SHAP/LIME "
            "drivers with the student during an advising session."
        )

    return plan


def _clean_pdf_text(text: Any) -> str:
    s = str(text)
    for ch in ("**", "###", "##", "#"):
        s = s.replace(ch, "")
    return s.encode("latin-1", "replace").decode("latin-1")


def build_student_pdf_dossier(
    student_id: str,
    student_row: Dict[str, Any],
    risk_proba: float,
    risk_level: str,
    plan: Dict[str, Any],
    nl_explanations: List[str],
) -> bytes:
    """Generate a printable PDF decision-support dossier for a single student."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(15, 12, 41)
    pdf.rect(0, 0, 210, 34, "F")
    pdf.set_y(9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 7, _clean_pdf_text("EDUGUARD | STUDENT EARLY WARNING DOSSIER"), 0, 1, "C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, _clean_pdf_text("Educator Decision-Support Report (Prototype Thresholds)"), 0, 1, "C")

    pdf.set_y(42)
    pdf.set_text_color(48, 43, 99)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, _clean_pdf_text(f"1. Student Summary — ID #{student_id}"), 0, 1, "L")

    pdf.set_text_color(51, 65, 85)
    pdf.set_font("Helvetica", "", 9)
    summary_lines = [
        f"Predicted Dropout Probability: {risk_proba:.2%} | Tier: {risk_level}",
        f"1st Sem Credit Completion: {float(student_row.get('Credit_Completion_Rate_1st_Sem', 0)):.2f} | "
        f"2nd Sem Credit Completion: {float(student_row.get('Credit_Completion_Rate_2nd_Sem', 0)):.2f}",
        f"Grade Drop (S1 - S2): {float(student_row.get('Grade_Drop', 0)):.2f} | "
        f"Academic Stress Index: {float(student_row.get('Academic_Stress_Index', 0)):.1f}",
        f"Engagement Score: {float(student_row.get('Engagement_Score', 0)):.2f} | "
        f"Financial Risk Indicator: {float(student_row.get('Financial_Risk', student_row.get('Financial_Risk_Score', 0))):.1f}",
    ]
    for ln in summary_lines:
        pdf.cell(0, 5.5, _clean_pdf_text(ln), 0, 1, "L")

    pdf.ln(3)
    pdf.set_text_color(48, 43, 99)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, _clean_pdf_text("2. Explainable AI Risk Drivers (SHAP)"), 0, 1, "L")
    pdf.set_text_color(51, 65, 85)
    pdf.set_font("Helvetica", "", 9)
    for b in nl_explanations:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 5, _clean_pdf_text(f"- {b}"))

    pdf.ln(3)
    pdf.set_x(pdf.l_margin)
    pdf.set_text_color(48, 43, 99)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, _clean_pdf_text("3. Recommended Decision-Support Interventions"), 0, 1, "L")
    pdf.set_text_color(51, 65, 85)
    for sec_title, key in [
        ("Academic Support", "academic"),
        ("Financial Guidance", "financial"),
        ("Engagement & Mentoring", "engagement"),
    ]:
        items = plan.get(key, [])
        if items:
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.cell(0, 5.5, _clean_pdf_text(f"[{sec_title}]"), 0, 1, "L")
            pdf.set_font("Helvetica", "", 9)
            for item in items:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(pdf.epw, 5, _clean_pdf_text(f"- {item}"))

    pdf.ln(4)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(pdf.epw, 4.5, _clean_pdf_text(plan.get("advisory_note", "")))

    return bytes(pdf.output())


def build_cohort_summary_pdf(
    model_name: str,
    metrics_dict: Dict[str, float],
    risk_counts: Dict[str, int],
    total_students: int,
) -> bytes:
    """Generate a printable cohort & model performance PDF report."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(15, 12, 41)
    pdf.rect(0, 0, 210, 34, "F")
    pdf.set_y(9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 7, _clean_pdf_text("EDUGUARD | COHORT & MODEL PERFORMANCE REPORT"), 0, 1, "C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, _clean_pdf_text(f"Active Model: {model_name} | Test Cohort Size: {total_students}"), 0, 1, "C")

    pdf.set_y(42)
    pdf.set_text_color(48, 43, 99)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, _clean_pdf_text("1. Measured Test-Split Model Metrics"), 0, 1, "L")
    pdf.set_text_color(51, 65, 85)
    pdf.set_font("Helvetica", "", 9.5)
    for k, v in metrics_dict.items():
        pdf.cell(0, 5.5, _clean_pdf_text(f"- {k}: {v:.4f}"), 0, 1, "L")

    pdf.ln(3)
    pdf.set_text_color(48, 43, 99)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, _clean_pdf_text("2. Risk Distribution (Prototype Thresholds)"), 0, 1, "L")
    pdf.set_text_color(51, 65, 85)
    pdf.set_font("Helvetica", "", 9.5)
    for tier in ["Critical", "High", "Medium", "Low"]:
        cnt = risk_counts.get(tier, 0)
        pct = (cnt / total_students * 100.0) if total_students > 0 else 0.0
        pdf.cell(0, 5.5, _clean_pdf_text(f"- {tier}: {cnt} students ({pct:.1f}%)"), 0, 1, "L")

    return bytes(pdf.output())
