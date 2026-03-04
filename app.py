"""
app.py — CertifyX Streamlit Application
========================================
Premium dark-themed UI.

Verdict states:
  GENUINE    → clean green UI, all green boxes
  SUSPICIOUS → orange warning — template mismatch or unknown issuer
  FAKE       → red forensic UI
"""

import io
import tempfile
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="CertifyX — Certificate Authenticity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] { font-family:'Inter',sans-serif !important; }
.main .block-container { padding-top:1rem; padding-bottom:2rem; background:#0d1117; }
body { background:#0d1117; color:#e6edf3; }

.certifyx-header {
    background:linear-gradient(135deg,#1e3a5f 0%,#0d2137 60%,#091929 100%);
    border:1px solid rgba(56,139,253,0.25); border-radius:16px;
    padding:2rem 2.5rem; margin-bottom:1.5rem; text-align:center;
}
.certifyx-header h1 { font-size:2.4rem;font-weight:700;color:#58a6ff;margin:0 0 0.3rem;letter-spacing:-0.5px; }
.certifyx-header p  { color:#8b949e;font-size:1rem;margin:0; }

.section-label {
    font-size:0.72rem;font-weight:600;letter-spacing:0.12em;
    text-transform:uppercase;color:#58a6ff;margin-bottom:0.5rem;
}
.card { background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:0.8rem; }

.verdict-fake {
    background:rgba(220,38,38,0.15);border:2px solid rgba(239,68,68,0.6);
    border-radius:14px;text-align:center;padding:1.4rem 1rem;
    animation:pulse-red 2s infinite;
}
.verdict-genuine {
    background:rgba(34,197,94,0.12);border:2px solid rgba(34,197,94,0.5);
    border-radius:14px;text-align:center;padding:1.4rem 1rem;
    animation:pulse-green 2.5s infinite;
}
.verdict-suspicious {
    background:rgba(249,115,22,0.12);border:2px solid rgba(249,115,22,0.6);
    border-radius:14px;text-align:center;padding:1.4rem 1rem;
    animation:pulse-orange 2s infinite;
}
.verdict-amber {
    background:rgba(251,191,36,0.10);border:2px solid rgba(251,191,36,0.5);
    border-radius:14px;text-align:center;padding:1.4rem 1rem;
}
@keyframes pulse-red    { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0);}    50%{box-shadow:0 0 18px 4px rgba(239,68,68,0.25);} }
@keyframes pulse-green  { 0%,100%{box-shadow:0 0 0 0 rgba(34,197,94,0);}    50%{box-shadow:0 0 18px 4px rgba(34,197,94,0.18);} }
@keyframes pulse-orange { 0%,100%{box-shadow:0 0 0 0 rgba(249,115,22,0);}   50%{box-shadow:0 0 18px 4px rgba(249,115,22,0.28);} }

.verdict-text-fake       { font-size:2.2rem;font-weight:700;color:#ef4444;letter-spacing:0.06em; }
.verdict-text-genuine    { font-size:2.2rem;font-weight:700;color:#22c55e;letter-spacing:0.06em; }
.verdict-text-suspicious { font-size:2.2rem;font-weight:700;color:#f97316;letter-spacing:0.06em; }
.verdict-text-amber      { font-size:2.2rem;font-weight:700;color:#fbbf24;letter-spacing:0.06em; }
.verdict-sub { font-size:0.88rem;color:#8b949e;margin-top:0.3rem; }

.risk-high   { background:rgba(220,38,38,0.10); border-color:rgba(239,68,68,0.40); }
.risk-medium { background:rgba(234,179,8,0.10); border-color:rgba(234,179,8,0.40); }
.risk-low    { background:rgba(34,197,94,0.08); border-color:rgba(34,197,94,0.35); }
.risk-label-high   { color:#ef4444;font-weight:700;font-size:1.15rem; }
.risk-label-medium { color:#eab308;font-weight:700;font-size:1.15rem; }
.risk-label-low    { color:#22c55e;font-weight:700;font-size:1.15rem; }

.issues-card {
    background:rgba(220,38,38,0.08);border:1px solid rgba(239,68,68,0.40);
    border-radius:12px;padding:1.2rem 1.4rem;margin-top:0.8rem;
}
.issue-item { display:flex;align-items:flex-start;gap:0.6rem;padding:0.35rem 0;
    font-size:0.93rem;color:#fca5a5;border-bottom:1px solid rgba(239,68,68,0.12); }
.issue-item:last-child { border-bottom:none; }
.issue-dot { color:#ef4444;font-size:1.1rem;flex-shrink:0;margin-top:1px; }

.suspicious-card {
    background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.45);
    border-radius:12px;padding:1.2rem 1.4rem;margin-top:0.8rem;
}
.suspicious-item { display:flex;align-items:flex-start;gap:0.6rem;padding:0.35rem 0;
    font-size:0.93rem;color:#fdba74;border-bottom:1px solid rgba(249,115,22,0.12); }
.suspicious-item:last-child { border-bottom:none; }
.suspicious-dot { color:#f97316;font-size:1.1rem;flex-shrink:0;margin-top:1px; }

.genuine-clean-card {
    background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.35);
    border-radius:12px;padding:1.2rem 1.4rem;margin-top:0.8rem;
}
.info-card {
    background:rgba(56,139,253,0.07);border:1px solid rgba(56,139,253,0.30);
    border-radius:10px;padding:0.9rem 1.1rem;margin-top:0.6rem;
}

.feat-table { width:100%;border-collapse:collapse;font-size:0.88rem; }
.feat-table th {
    background:#1c2128;color:#8b949e;font-weight:600;
    text-transform:uppercase;letter-spacing:0.08em;font-size:0.72rem;
    padding:0.5rem 0.8rem;text-align:left;border-bottom:1px solid #30363d;
}
.feat-table td { padding:0.45rem 0.8rem;border-bottom:1px solid #21262d;color:#e6edf3; }
.feat-ok   { color:#22c55e;font-weight:600; }
.feat-warn { color:#eab308;font-weight:600; }
.feat-bad  { color:#ef4444;font-weight:600; }

.legend-item { display:flex;align-items:center;gap:0.5rem;font-size:0.83rem;color:#8b949e; }
.legend-box  { width:14px;height:14px;border-radius:3px;flex-shrink:0; }

.footer { text-align:center;color:#484f58;font-size:0.78rem;
    margin-top:2rem;padding-top:1rem;border-top:1px solid #21262d; }
</style>
""", unsafe_allow_html=True)


def bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def render_verdict(prediction: str, confidence: int, is_low_confidence: bool):
    if prediction == "FAKE":
        st.markdown(f"""
        <div class="verdict-fake">
            <div class="section-label" style="color:#ef4444;">Verdict</div>
            <div class="verdict-text-fake">&#10008; FAKE</div>
            <div class="verdict-sub">Model confidence: {confidence}%</div>
        </div>
        """, unsafe_allow_html=True)
    elif prediction == "SUSPICIOUS":
        st.markdown(f"""
        <div class="verdict-suspicious">
            <div class="section-label" style="color:#f97316;">Verdict</div>
            <div class="verdict-text-suspicious">&#9888; SUSPICIOUS</div>
            <div class="verdict-sub">ML predicted GENUINE but verification checks failed</div>
        </div>
        """, unsafe_allow_html=True)
    elif prediction == "GENUINE" and is_low_confidence:
        st.markdown(f"""
        <div class="verdict-amber">
            <div class="section-label" style="color:#fbbf24;">Verdict</div>
            <div class="verdict-text-amber">&#9888; GENUINE?</div>
            <div class="verdict-sub">Low confidence: {confidence}% — manual review recommended</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-genuine">
            <div class="section-label" style="color:#22c55e;">Verdict</div>
            <div class="verdict-text-genuine">&#10004; GENUINE</div>
            <div class="verdict-sub">Model confidence: {confidence}%</div>
        </div>
        """, unsafe_allow_html=True)


def render_template_and_institution_info(result: dict):
    template_score    = result.get("template_score", 1.0)
    best_template     = result.get("best_template")
    templates_checked = result.get("templates_checked", False)
    institution_name  = result.get("institution_name")
    institution_known = result.get("institution_known")
    rows = []
    if templates_checked and best_template:
        pct   = int(template_score * 100)
        color = "#22c55e" if template_score >= 0.60 else "#f97316"
        icon  = "&#10003;" if template_score >= 0.60 else "&#9888;"
        rows.append(f'<tr><td style="color:#8b949e;">Template Match</td>'
                    f'<td style="color:{color};font-weight:600;">{icon} {pct}% ({best_template})</td></tr>')
    elif not templates_checked:
        rows.append('<tr><td style="color:#8b949e;">Template Match</td>'
                    '<td style="color:#484f58;">&#8212; No templates in templates/ folder (skipped)</td></tr>')
    if institution_name:
        if institution_known:
            rows.append(f'<tr><td style="color:#8b949e;">Issuer</td>'
                        f'<td style="color:#22c55e;font-weight:600;">&#10003; {institution_name}</td></tr>')
        else:
            rows.append(f'<tr><td style="color:#8b949e;">Issuer</td>'
                        f'<td style="color:#f97316;font-weight:600;">&#9888; {institution_name} (unrecognized)</td></tr>')
    elif institution_known is None:
        rows.append('<tr><td style="color:#8b949e;">Issuer</td>'
                    '<td style="color:#484f58;">&#8212; Institution header not detected</td></tr>')
    if not rows: return
    rows_html = "".join(rows)
    st.markdown(f"""
    <div class="section-label" style="margin-top:0.8rem;">Verification Checks</div>
    <div class="info-card">
        <table style="width:100%;font-size:0.88rem;border-collapse:collapse;">{rows_html}</table>
    </div>
    """, unsafe_allow_html=True)


def render_suspicious_reasons(reasons: list):
    if not reasons: return
    items = "".join(
        f'<div class="suspicious-item"><span class="suspicious-dot">&#9888;</span><span>{r}</span></div>'
        for r in reasons)
    st.markdown(f"""
    <div class="section-label" style="color:#f97316;">Why SUSPICIOUS?</div>
    <div class="suspicious-card">
        <div style="font-size:0.78rem;color:#f97316;margin-bottom:0.5rem;">
            The ML model predicted GENUINE, but the following verification checks failed:
        </div>
        {items}
    </div>
    """, unsafe_allow_html=True)


def render_risk(risk_level: str, forgery_pct: int):
    css  = f"risk-{risk_level.lower()}"
    lcss = f"risk-label-{risk_level.lower()}"
    icon = {"HIGH":"&#128308;","MEDIUM":"&#128992;","LOW":"&#128994;"}.get(risk_level,"")
    st.markdown(f"""
    <div class="card {css}" style="border-width:1px;border-style:solid;">
        <div class="section-label">Risk Assessment</div>
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
            <div><span style="font-size:0.82rem;color:#8b949e;">Risk Level&nbsp;&nbsp;</span>
                 <span class="{lcss}">{icon} {risk_level}</span></div>
            <div><span style="font-size:0.82rem;color:#8b949e;">Forgery Probability&nbsp;&nbsp;</span>
                 <span class="{lcss}">{forgery_pct}%</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_issues(issues: list, prediction: str, suppress: bool,
                  is_low_conf: bool, critical_count: int):
    if prediction == "GENUINE":
        if suppress:
            if is_low_conf:
                st.markdown("""
                <div class="card" style="border-color:rgba(251,191,36,0.4);background:rgba(251,191,36,0.06);margin-top:0.8rem;">
                    <div class="section-label" style="color:#fbbf24;">Advisory</div>
                    <p style="color:#fde68a;font-size:0.9rem;margin:0;">
                        &#9888; Model confidence below 60% — manually verify with issuing institution.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="genuine-clean-card">
                    <div class="section-label" style="color:#22c55e;">Document Status</div>
                    <p style="color:#22c55e;font-size:1rem;font-weight:500;margin:0;">&#10003; All key checks passed</p>
                    <p style="color:#86efac;font-size:0.85rem;margin:0.4rem 0 0;">
                        ML predicts genuine. Template and issuer verified. No anomalies found.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            if issues:
                items = "".join(
                    f'<div class="suspicious-item"><span class="suspicious-dot">&#9888;</span><span>{i}</span></div>'
                    for i in issues)
                st.markdown(f"""
                <div class="section-label" style="color:#fbbf24;">Critical Observations</div>
                <div class="suspicious-card">
                    <div style="font-size:0.78rem;color:#fbbf24;margin-bottom:0.5rem;">
                        {critical_count} critical features failed — manual review recommended.
                    </div>
                    {items}
                </div>
                """, unsafe_allow_html=True)
        return
    if not issues:
        st.markdown("""
        <div class="card" style="border-color:#22c55e40;background:rgba(34,197,94,0.07);">
            <div class="section-label">Detected Issues</div>
            <span style="color:#22c55e;font-size:0.95rem;">&#10003; No threshold issues detected</span>
        </div>
        """, unsafe_allow_html=True)
        return
    items = "".join(
        f'<div class="issue-item"><span class="issue-dot">&#9679;</span><span>{i}</span></div>'
        for i in issues)
    st.markdown(f"""
    <div class="section-label">Detected Issues</div>
    <div class="issues-card">{items}</div>
    """, unsafe_allow_html=True)


def render_feature_table(key_scores: dict, prediction: str):
    color_limits = {
        "OCR Confidence":(40,"below"), "Logo Similarity":(0.35,"below"),
        "Seal Shape Score":(0.30,"below"), "Signature Match":(0.30,"below"),
        "Font Variance":(200,"above"), "Char Spacing Variance":(100,"above"),
        "Image Sharpness":(50,"below"), "USN/ID Valid":(1.0,"below"),
        "Date Valid":(1.0,"below"), "Score Validity":(0.3,"below"),
        "Edge Density":(0.02,"below"), "Word Count":(5,"below"),
    }
    def _cls(name, val):
        if name not in color_limits: return "feat-ok"
        thr, direction = color_limits[name]
        try: v = float(val)
        except: return "feat-ok"
        bad  = (v < thr*0.6) if direction=="below" else (v > thr*1.6)
        warn = (v < thr)     if direction=="below" else (v > thr)
        if bad:  return "feat-bad" if prediction=="FAKE" else "feat-warn"
        if warn: return "feat-warn"
        return "feat-ok"
    icons = {"feat-ok":"&#10003; OK","feat-warn":"&#9888; Warning","feat-bad":"&#10008; Bad"}
    rows = "".join(
        f"<tr><td>{n}</td><td class='{_cls(n,v)}'>{v}</td>"
        f"<td><span class='{_cls(n,v)}'>{icons.get(_cls(n,v),'')}</span></td></tr>"
        for n,v in key_scores.items())
    st.markdown(f"""
    <div class="section-label">Key Feature Scores</div>
    <table class="feat-table">
        <thead><tr><th>Feature</th><th>Value</th><th>Status</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


def render_annotation_legend(prediction: str, critical_count: int):
    if prediction == "GENUINE" and critical_count < 3:
        st.markdown("""<div style="display:flex;gap:1rem;margin-top:0.6rem;">
            <div class="legend-item"><div class="legend-box" style="background:#22c55e;"></div>
            <span>All regions verified genuine</span></div></div>""", unsafe_allow_html=True)
    elif prediction == "SUSPICIOUS":
        st.markdown("""<div style="display:flex;gap:1rem;margin-top:0.6rem;flex-wrap:wrap;">
            <div class="legend-item"><div class="legend-box" style="background:#f97316;"></div>
            <span>Suspicious / unverified</span></div>
            <div class="legend-item"><div class="legend-box" style="background:#22c55e;"></div>
            <span>Passed check</span></div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="display:flex;gap:1rem;margin-top:0.6rem;flex-wrap:wrap;">
            <div class="legend-item"><div class="legend-box" style="background:#dc2626;"></div>
            <span>Anomalous region</span></div>
            <div class="legend-item"><div class="legend-box" style="background:#22c55e;"></div>
            <span>Passed check</span></div></div>""", unsafe_allow_html=True)


def main():
    st.markdown("""
    <div class="certifyx-header">
        <h1>&#128269; CertifyX</h1>
        <p>AI-powered Fake Academic Certificate Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Upload Certificate</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Upload Certificate Image",
        type=["png","jpg","jpeg","bmp","tiff","tif"],
        help="Upload a certificate image (JPG, PNG, BMP, TIFF).",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem;color:#8b949e;">
            <span style="font-size:3rem;">&#128196;</span><br><br>
            Upload a certificate image above to begin forensic analysis.
        </div>
        """, unsafe_allow_html=True)
        return

    file_bytes = uploaded_file.read()
    orig_bgr   = bytes_to_bgr(file_bytes)
    orig_pil   = bgr_to_pil(orig_bgr)

    col_img, _ = st.columns([3, 1])
    with col_img:
        st.markdown('<div class="section-label">Uploaded Certificate</div>', unsafe_allow_html=True)
        st.image(orig_pil, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if not st.button("&#128269;  Analyse Certificate", key="analyse_btn"):
        return

    with st.spinner("Running forensic analysis — please wait ..."):
        suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            from predict import predict_certificate
            result = predict_certificate(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if result.get("error"):
        st.error(f"Analysis failed: {result['error']}")
        st.info("Make sure certificate_detection_model.pkl exists and all dependencies are installed.")
        return

    prediction     = result["prediction"]
    confidence     = result["confidence"]
    issues         = result["detected_issues"]
    critical_count = result["critical_fail_count"]
    suppress       = result["suppress_issues"]
    is_low_conf    = result["is_low_confidence"]
    risk_level     = result["risk_level"]
    forgery_pct    = result["forgery_probability"]
    is_unknown     = result["is_unknown"]
    unknown_rsn    = result["unknown_reason"]
    suspicious_rsn = result["suspicious_reasons"]
    ann_img        = result["annotated_image"]
    key_scores     = result["key_scores"]

    st.markdown("---")
    left, right = st.columns([5, 5], gap="large")

    with left:
        if is_unknown:
            st.markdown(f"""
            <div class="card" style="border-color:rgba(234,179,8,0.4);background:rgba(234,179,8,0.08);">
                <strong style="color:#fbbf24;">&#9888; Unknown Certificate Format</strong><br>
                <span style="font-size:0.88rem;color:#fde68a;">{unknown_rsn}</span>
            </div>
            """, unsafe_allow_html=True)

        if is_low_conf and prediction == "GENUINE":
            st.markdown(f"""
            <div class="card" style="border-color:rgba(251,191,36,0.4);background:rgba(251,191,36,0.07);margin-bottom:0.6rem;">
                <span style="color:#fbbf24;font-weight:600;">&#9888; Low-Confidence Result ({confidence}%)</span>
                <p style="color:#fde68a;font-size:0.88rem;margin:0.3rem 0 0;">
                    Please manually verify this certificate with the issuing institution.
                </p>
            </div>
            """, unsafe_allow_html=True)

        render_verdict(prediction, confidence, is_low_conf)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Confidence</div>', unsafe_allow_html=True)
        st.progress(confidence / 100, text=f"{prediction} — {confidence}% confident")
        st.markdown("<br>", unsafe_allow_html=True)

        if prediction in ("FAKE", "SUSPICIOUS") or is_unknown or is_low_conf:
            render_risk(risk_level, forgery_pct)

        render_template_and_institution_info(result)

        if prediction == "SUSPICIOUS" and suspicious_rsn:
            render_suspicious_reasons(suspicious_rsn)

        render_issues(issues, prediction, suppress, is_low_conf, critical_count)

        if key_scores:
            st.markdown("<br>", unsafe_allow_html=True)
            render_feature_table(key_scores, prediction)

    with right:
        st.markdown('<div class="section-label">Forensic Annotation</div>', unsafe_allow_html=True)
        if ann_img is not None:
            st.image(bgr_to_pil(ann_img), use_container_width=True)
            render_annotation_legend(prediction, critical_count)
        else:
            st.info("Annotated image unavailable.")

    st.markdown("""
    <div class="footer">
        CertifyX &mdash; AI-powered certificate forensics &nbsp;|&nbsp;
        Results are indicative. Always verify with the issuing institution.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
