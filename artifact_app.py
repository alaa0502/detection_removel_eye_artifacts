# artifact_app.py
import streamlit as st
import mne
import os
import numpy as np

# 👇 your function (unchanged)
from final_model import clean_edf_file

# Try Plotly for the interactive preview
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# --------------------
# Page setup & title
# --------------------
st.set_page_config(page_title="Automatic detection and removal of eye artifacts", layout="wide")
st.title("Automatic detection and removal of eye artifacts")

# ---- Theme: light page, white uploader, dark sidebar & title ----
st.markdown("""
<style>
/* page background (very light grey) */
[data-testid="stAppViewContainer"] { background-color: #F5F6F8; }

/* title color (same dark as sidebar) */
h1, h2, h3 { color: #111827; }

/* ───────────────────────────────
   Sidebar styling + compaction
   ─────────────────────────────── */
section[data-testid="stSidebar"] > div {
  background-color: #111827;
  color: #E5E7EB;
  padding-top: 0 !important;                 /* pull content up */
}
section[data-testid="stSidebar"] { padding-top: 0 !important; }
section[data-testid="stSidebar"] h3:first-of-type {
  margin-top: 0 !important;                  /* no gap over "Demo" */
}
section[data-testid="stSidebar"] p {          /* caption under Demo */
  margin: 0.25rem 0 0.5rem 0 !important;     /* tighter caption */
}

/* Demo buttons: a little smaller */
section[data-testid="stSidebar"] [data-testid="stDownloadButton"] {
  margin: 3px 0 !important;                  /* tighter gaps */
}
section[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button,
section[data-testid="stSidebar"] [data-testid="stDownloadButton"] > a {
  padding: 4px 8px !important;               /* shorter, narrower */
  min-height: 28px !important;               /* lower height */
  font-size: 0.85rem !important;             /* smaller text */
  line-height: 1.15 !important;
  border-radius: 8px !important;             /* slightly less rounded */
}
section[data-testid="stSidebar"] a { color: #BFDBFE !important; }

/* ⬇️ Sidebar titles ("Demo", "Resources") — ultra-tight to content */
section[data-testid="stSidebar"] h3 {
  color: #F3F4F6;
  font-size: 1.5rem;
  line-height: 1.1;                          /* slightly tighter line box */
  margin: 0 !important;                      /* remove all margins */
}

/* Remove extra vertical space between ALL sidebar markdown blocks */
section[data-testid="stSidebar"] [data-testid="stMarkdown"] {
  margin: 0 !important;
}

/* uploader card: white and slightly raised */
section[data-testid="stFileUploader"],
div[data-testid="stFileUploader"] {
  background: #FFFFFF !important;
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
div[data-testid="stFileUploaderDropzone"] {
  background: #FFFFFF !important;
  border-radius: 10px;
  border: 1px dashed #E5E7EB;
}

/* pill style for the MODE toggle only */
.mode-pills div[role="radiogroup"] > label{
  border:1px solid #d1d5db; padding:8px 14px; border-radius:9999px;
  margin-right:8px; background:#f8fafc; color:#111827; cursor:pointer;
}
.mode-pills div[role="radiogroup"] > label[data-checked="true"]{
  background:#0EA5E9; border-color:#0EA5E9; color:white;
}

/* === Make the MODAL proceed download button RED === */
[data-testid="stDialog"] [data-testid="stDownloadButton"] > button,
[data-testid="stDialog"] [data-testid="stDownloadButton"] > a {
  background-color: #DC2626 !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 10px 16px !important;
  font-weight: 600 !important;
}
[data-testid="stDialog"] [data-testid="stDownloadButton"] > button:hover,
[data-testid="stDialog"] [data-testid="stDownloadButton"] > a:hover {
  opacity: .95;
}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    with st.sidebar:
        st.markdown("### Demo")
        st.caption("Demo files are available for you to try. Source: [PhysioNet EEGMMI Database](https://physionet.org/content/eegmmidb/1.0.0/).")

        demo_files = [
            ("⬇️ Demo — S018R02 (raw)", "demo/S018R02.edf"),
            ("⬇️ Demo — S026R02 (raw)", "demo/S026R02.edf"),
            ("⬇️ Demo — S028R02 (raw)", "demo/S028R02.edf"),
        ]
        for label, path in demo_files:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                st.download_button(
                    label,
                    data=data,
                    file_name=os.path.basename(path),
                    mime="application/octet-stream",
                    key=f"demo_{os.path.basename(path)}"
                )
            except Exception:
                st.caption(f"{label} — file not found at `{path}`")

        st.markdown("### Resources")

        st.markdown("""
        <style>
          section[data-testid="stSidebar"] { position: relative; z-index: 5; overflow: visible !important; }
          section[data-testid="stSidebar"] > div { overflow: visible !important; }

          /* Tighter list metrics right under "Resources" */
          section[data-testid="stSidebar"] ul.res-list {
            list-style: none;
            padding-left: 0;
            margin: -0.20rem 0 0 0 !important;  /* slight negative to hug the title */
          }
          section[data-testid="stSidebar"] ul.res-list li { margin: 0 !important; }  /* no gaps between items */

          section[data-testid="stSidebar"] .tooltip.label {
            position: relative; display: inline-block; cursor: default;
            color: #BFDBFE; font-weight: 600; text-decoration: none;
          }
          section[data-testid="stSidebar"] .tooltip.label:hover { text-decoration: underline; }

          section[data-testid="stSidebar"] .tooltip .note {
            position: absolute; left: 0; top: calc(100% + 8px); width: min(480px, 90vw);
            background: #0A1F44; color: #FFFFFF; padding: 12px 14px; border-radius: 10px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35); font-size: 1.18rem; line-height: 1.5;
            opacity: 0; visibility: hidden; transition: opacity .15s ease, visibility .15s ease;
            z-index: 99999; pointer-events: auto; white-space: normal;
          }
          section[data-testid="stSidebar"] .tooltip .note a {
            color: #BFDBFE; text-decoration: underline; font-weight: 600;
          }
          section[data-testid="stSidebar"] .tooltip .note:after {
            content: ""; position: absolute; left: 16px; top: -8px;
            border-width: 8px; border-style: solid; border-color: transparent transparent #0A1F44 transparent;
          }
          section[data-testid="stSidebar"] .tooltip:hover .note { opacity: 1; visibility: visible; }

          @media (max-width: 1200px) {
            section[data-testid="stSidebar"] .tooltip .note { width: 95%; }
          }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <ul class="res-list">
          <li>
            <span class="tooltip label">— (Komisaruk &amp; Nikulchev, 2021)
              <span class="note">
                Shows that machine learning, especially neural networks, can replace manual work in finding EEG artifacts.
                <br><br>
                <a href="http://dx.doi.org/10.14569/IJACSA.2021.0121204" target="_blank" rel="noopener">Open paper ↗</a>
              </span>
            </span>
          </li>
          <li>
            <span class="tooltip label">— (Maddirala &amp; Veluvolu, 2022)
              <span class="note">
                Proves that eye blinks can be removed even from one front channel (like FP1/FP2).
                <br><br>
                <a href="https://doi.org/10.3390/s22030931" target="_blank" rel="noopener">Open paper ↗</a>
              </span>
            </span>
          </li>
          <li>
            <span class="tooltip label">— (IJCCC, 2019)
              <span class="note">
                Shows a real-time method (wavelets) for detecting blinks, confirming it is possible to target them automatically.
                <br><br>
                <a href="https://univagora.ro/jour/index.php/ijccc/article/view/3516" target="_blank" rel="noopener">Open paper ↗</a>
              </span>
            </span>
          </li>
        </ul>
        """, unsafe_allow_html=True)

# ---- Session state (download + preview + disclaimer) ----
ss = st.session_state
if "clean_bytes" not in ss:
    ss.clean_bytes = None
    ss.clean_filename = None
    ss.clean_path = None
    ss.last_uploaded_name = None
if "disclaimer_pending" not in ss:
    ss.disclaimer_pending = False

# 🛠 persist user choices so Fine-tuned survives rerun
if "mode" not in ss: ss.mode = "Default artifact removal"
if "channel_logic" not in ss: ss.channel_logic = "either"
if "threshold_value" not in ss: ss.threshold_value = 0.35

# ========================
# File upload (shown first)
# ========================
uploaded = st.file_uploader("Upload an EDF file", type=["edf"])

# If a new file is selected, reset state
if uploaded and (ss.last_uploaded_name != uploaded.name):
    ss.clean_bytes = None
    ss.clean_filename = None
    ss.clean_path = None
    ss.disclaimer_pending = False
    ss.last_uploaded_name = uploaded.name

# Dedicated download area right under the uploader
download_area = st.container()

# Validate on Start
invalid_ext = bool(uploaded) and (not uploaded.name.lower().endswith(".edf"))

# ======================================
# Mode + Fine-tuning setup (below upload)
# ======================================
channel_logic = "either"   # default
threshold_value = 0.35     # default decision threshold

st.markdown('<div class="mode-pills">', unsafe_allow_html=True)
mode = st.radio("", ["Default artifact removal", "Fine-tuned artifact removal"],
                horizontal=True, key="mode_radio")
st.markdown('</div>', unsafe_allow_html=True)
ss.mode = mode  # persist

if mode == "Fine-tuned artifact removal":
    # --- Channel choice + help ---
    if hasattr(st, "popover"):
        ch_l, ch_r = st.columns([1, 0.1])
        with ch_l:
            ch_choice = st.radio("Channel choice", ["Both Fp1 and Fp2", "Single", "Either"],
                                 index=2, horizontal=True, key="ch_choice")
        with ch_r:
            with st.popover("Help"):
                st.markdown(
                    "**Channel logic (Fp1/Fp2)**\n"
                    "- **Both (AND):** ↑ precision/specificity, ↓ recall.\n"
                    "- **Either (OR):** ↑ sensitivity/recall.\n"
                    "- **Single:** use one lead if the other is missing."
                )
    else:
        ch_choice = st.radio(
            "Channel choice", ["Both Fp1 and Fp2", "Single", "Either"],
            index=2, horizontal=True, key="ch_choice",
            help="Both=AND (↑ precision/specificity, ↓ recall). Either=OR (↑ sensitivity/recall). Single=one lead."
        )

    if ch_choice == "Both Fp1 and Fp2":
        channel_logic = "both"
    elif ch_choice == "Single":
        channel_logic = "single"
    else:
        channel_logic = "either"

    # --- Threshold choice + help ---
    thr_options = ["High 0.50", "Standard 0.35", "Low 0.25"]
    if hasattr(st, "popover"):
        th_l, th_r = st.columns([1, 0.1])
        with th_l:
            thr_choice = st.radio("Threshold choice", thr_options, index=1, horizontal=True, key="thr_choice")
        with th_r:
            with st.popover("Help"):
                st.markdown(
                    "**Threshold (τ) — predict blink if score ≥ τ.**\n"
                    "- **High 0.50:** strict (↑ precision), fewer segments cleaned.\n"
                    "- **Standard 0.35:** balanced (default).\n"
                    "- **Low 0.25:** soft (↑ recall), may over-clean."
                )
    else:
        thr_choice = st.radio(
            "Threshold choice", thr_options, index=1, horizontal=True, key="thr_choice",
            help="High=strict, Standard=balanced, Low=soft."
        )

    if "0.50" in thr_choice:
        threshold_value = 0.50
    elif "0.35" in thr_choice:
        threshold_value = 0.35
    else:
        threshold_value = 0.25

    # persist fine-tune selections
    ss.channel_logic = channel_logic
    ss.threshold_value = threshold_value
else:
    # Default mode (exactly as you want): either + 0.35
    ss.channel_logic = "either"
    ss.threshold_value = 0.35

# ========================
# Start button
# ========================
if st.button("▶️ Start", key="start_btn"):
    if not uploaded:
        st.error("Upload an EDF file first.")
    elif invalid_ext:
        st.error("Only .edf files are supported.")
    else:
        try:
            # Save uploaded file to temp path
            in_path = "uploaded_tmp.edf"
            with open(in_path, "wb") as f:
                f.write(uploaded.read())

            out_path = os.path.splitext(uploaded.name)[0] + "_clean.edf"

            saved_path, n_segments, total_masked_sec = clean_edf_file(
                in_path=in_path,
                out_path=out_path,
                logic=ss.channel_logic,
                des_thresh=ss.threshold_value
            )

            st.success(f"Cleaned {uploaded.name}")

            # Store cleaned file for download + preview, then trigger modal
            with open(saved_path, "rb") as f:
                ss.clean_bytes = f.read()
            ss.clean_filename = out_path
            ss.clean_path = saved_path
            ss.disclaimer_pending = True

        except Exception as e:
            st.error(f"Processing failed: {e}")

# -------------------------
# Modal disclaimer (overlay)
# -------------------------
def _disclaimer_body():
    st.markdown("### ⚠️ Important note")
    st.write(
        "This tool targets only blink artifact cleaning on standard EEG. "
        "The underlying model was trained mainly on frontal channels Fp1 and Fp2; "
        "Suitable for students and non-expert use."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        if ss.clean_bytes:
            clicked = st.download_button(
                "I accept, Proceed ⬇",
                data=ss.clean_bytes,
                file_name=ss.clean_filename or "cleaned.edf",
                mime="application/octet-stream",
                key="dl_modal"
            )
            if clicked:
                ss.disclaimer_pending = False
                if hasattr(st, "rerun"):
                    st.rerun()
                elif hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
        else:
            st.info("Preparing file…")
    with c2:
        if st.button("Cancel", key="cancel_dl"):
            ss.clean_bytes = None
            ss.clean_filename = None
            ss.clean_path = None
            ss.disclaimer_pending = False
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

def show_disclaimer_modal_if_needed():
    if not ss.clean_bytes or not ss.disclaimer_pending:
        return
    if hasattr(st, "dialog"):
        @st.dialog("Important note", width="large")
        def _dlg():
            _disclaimer_body()
        _dlg()
    elif hasattr(st, "experimental_dialog"):
        @st.experimental_dialog("Important note")
        def _dlg():
            _disclaimer_body()
        _dlg()
    else:
        with st.container(border=True):
            _disclaimer_body()

# Invoke modal when needed
show_disclaimer_modal_if_needed()

# --- Render the download button ONLY after OK (backup under the uploader) ---
if ss.clean_bytes and not ss.disclaimer_pending:
    with download_area:
        with st.container(border=True):
            st.markdown("**Cleaned file ready**")
            st.download_button(
                "⬇️ Download cleaned EDF",
                data=ss.clean_bytes,
                file_name=ss.clean_filename or "cleaned.edf",
                mime="application/octet-stream",
                key="dl_top"
            )

# ========================
# Interactive preview (whole file) with Plotly rangeslider
# ========================
if ss.clean_path:
    try:
        raw_clean = mne.io.read_raw_edf(ss.clean_path, preload=True, verbose=False)
        sf = float(raw_clean.info["sfreq"])
        times = raw_clean.times
        data, _ = raw_clean[:, :]

        if PLOTLY_AVAILABLE:
            n_samples = data.shape[1]
            max_points = 20000
            step = max(1, n_samples // max_points)
            times_plot = times[::step] if step > 1 else times
            data_plot = data[:, ::step] if step > 1 else data

            fig = go.Figure()
            for i, ch in enumerate(raw_clean.ch_names[:8]):
                fig.add_trace(go.Scatter(
                    x=times_plot,
                    y=(data_plot[i] * 1e6) + i * 200,  # µV + vertical offset
                    mode="lines",
                    name=ch,
                    hoverinfo="skip"
                ))
            fig.update_layout(
                height=420,
                margin=dict(l=40, r=20, t=40, b=40),
                showlegend=False,
                xaxis=dict(title="Time (s)", rangeslider=dict(visible=True), zeroline=False),
                yaxis=dict(title="µV (offset per ch)", zeroline=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("For in-plot scrolling, install Plotly: `pip install plotly`. Showing a 10 s Matplotlib preview instead.")
            max_secs = min(10.0, times[-1])
            start_sample = 0
            stop_sample = int(max_secs * sf)
            data10 = data[:, start_sample:stop_sample]
            times10 = times[start_sample:stop_sample]

            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 5))
            for i, ch in enumerate(raw_clean.ch_names[:8]):
                plt.plot(times10, data10[i] * 1e6 + i * 200, label=ch)
            plt.xlabel("Time (s)")
            plt.ylabel("µV (offset per ch)")
            plt.title("Cleaned EEG — first 10s")
            plt.tight_layout()
            st.pyplot(plt.gcf(), clear_figure=True)

    except Exception as e:
        st.warning(f"Preview unavailable: {e}")
