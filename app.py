import streamlit as st
import pandas as pd
from src.dataset import REFERENCE_PROFILES
from src.discovery_search import CompliantLinkedInDiscovery
from src.ml_ranker import SchoolInfluencerMLRanker
from src.database import save_ranked_results

st.set_page_config(page_title="LinkedIn School Influencer Discovery Engine", layout="wide")

st.title("🏫 AI-Powered LinkedIn School Influencer Discovery Tool")
st.caption("Workflow: Reference Profile ➔ AI Context Analysis ➔ Similar Profile Discovery ➔ Hybrid ML Ranking (TF-IDF + LSA SVD + Multi-School Graph Authority) ➔ Explainable Results")

st.sidebar.header("🎯 1. Select or Input Reference Profile")
input_mode = st.sidebar.radio(
    "Reference Input Mode:",
    ["Choose Preset Reference Influencer", "Enter Any Custom LinkedIn Profile"]
)

if input_mode == "Choose Preset Reference Influencer":
    preset_names = [f"{p['name']} ({p['persona_category'].split(',')[0]})" for p in REFERENCE_PROFILES]
    selected_idx = st.sidebar.selectbox("Preset K-12 Reference Leader:", range(len(preset_names)), format_func=lambda i: preset_names[i])
    ref_profile = dict(REFERENCE_PROFILES[selected_idx])
else:
    ref_profile = {
        "profile_id": "CUSTOM-REF",
        "name": st.sidebar.text_input("Reference Name", "Dr. Rajeshwari Sharma"),
        "linkedin_url": st.sidebar.text_input("LinkedIn Profile URL", "https://www.linkedin.com/in/rajeshwari-sharma-k12-advisor/"),
        "current_role": st.sidebar.text_input("Current Role", "Senior K-12 School Consultant & Former CBSE Principal"),
        "organisation": st.sidebar.text_input("Organisation", "EduVision School Advisory & Turnkey Setup"),
        "location": st.sidebar.text_input("Location", "New Delhi, India"),
        "persona_category": st.sidebar.selectbox(
            "Primary Ecosystem Persona",
            [
                "School & Education Consultant / Advisor",
                "Former Principal, School Director, Owner & Trustee",
                "Education Entrepreneur & School Network Professional",
                "Experienced Multi-School Ecosystem Professional",
                "School Decision-Maker Relationship Builder"
            ]
        ),
        "experience_years": st.sidebar.slider("Years in School Ecosystem", 5, 40, 22),
        "schools_impacted_estimate": st.sidebar.number_input("Estimated Schools in Network", 5, 10000, 120),
        "boards_expertise": st.sidebar.text_input("Boards / Curriculum Expertise", "CBSE, ICSE, IB, NEP 2020, School Audits"),
        "ecosystem_indicators": st.sidebar.text_area(
            "Relevant Experience / School Ecosystem Indicators",
            "Advises 100+ private K-12 school trustees and principals on CBSE/IB affiliation, academic turnaround, teacher training, and new school launches."
        ),
        "bio_summary": "Experienced K-12 school education consultant, former principal, and advisor to school owners and trustees across India."
    }

include_xray = st.sidebar.checkbox("Attempt Live Public Search X-Ray (`site:linkedin.com/in/`)", value=False)

# Run End-to-End Pipeline
ranker = SchoolInfluencerMLRanker()
discovery = CompliantLinkedInDiscovery()

ref_analysis = ranker.analyze_reference_profile(ref_profile)
candidates = discovery.discover_candidates(ref_profile, include_live_xray=include_xray)
ranked_results = ranker.rank_candidates(ref_profile, candidates)
save_ranked_results(ref_profile["name"], ranked_results)

df = pd.DataFrame(ranked_results)

# Section 1: Reference Profile & AI Analysis Banner
st.subheader("🧠 Step 1 & 2: Reference Profile & AI Context Analysis")
c_ref1, c_ref2 = st.columns([3, 2])
with c_ref1:
    st.markdown(f"**Reference Profile:** [{ref_profile['name']}]({ref_profile['linkedin_url']})")
    st.markdown(f"**Role & Organisation:** `{ref_profile['current_role']}` at **{ref_profile['organisation']}** ({ref_profile['location']})")
    st.markdown(f"**Ecosystem Indicators:** {ref_profile['ecosystem_indicators']}")
with c_ref2:
    st.info(f"**AI Persona & Feature Extraction:**\n\n{ref_analysis['ai_context_summary']}")
    st.markdown(f"**Detected Curriculum Signals:** `{', '.join(ref_analysis['detected_boards'])}`")

st.divider()

# Top Metrics Bar
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Discovered Relevant Profiles", len(df))
m2.metric("High-Match Influencers (≥85%)", len(df[df["relevance_score"] >= 85.0]))
m3.metric("Total Schools Reached by Pool", f"{df['schools_impacted_estimate'].sum():,}+")
m4.metric("Top Match Score", f"{df['relevance_score'].max():.1f}%")
m5.metric("Avg Relevance Score", f"{df['relevance_score'].mean():.1f}%")

tab1, tab2, tab3 = st.tabs([
    "🏆 Ranked School Influencers & Match Explanations (26+ Profiles)",
    "🔬 Deep Profile Comparison & ML Score Breakdown",
    "📐 AI/ML Ranking Methodology & Production Compliance"
])

with tab1:
    f1, f2 = st.columns([4, 4])
    persona_filter = f1.multiselect(
        "Filter by Target School-Ecosystem Persona:",
        options=sorted(df["persona_category"].unique()),
        default=sorted(df["persona_category"].unique())
    )
    search_kw = f2.text_input("Search by Name, Role, Organisation, Location, or Board (e.g. CBSE, IB, Trustee, Mumbai):", "")

    filtered_df = df[df["persona_category"].isin(persona_filter)]
    if search_kw:
        mask = (
            filtered_df["name"].str.contains(search_kw, case=False, na=False) |
            filtered_df["current_role"].str.contains(search_kw, case=False, na=False) |
            filtered_df["organisation"].str.contains(search_kw, case=False, na=False) |
            filtered_df["location"].str.contains(search_kw, case=False, na=False) |
            filtered_df["ecosystem_indicators"].str.contains(search_kw, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    st.dataframe(
        filtered_df[[
            "rank", "name", "current_role", "organisation", "location",
            "persona_category", "schools_impacted_estimate", "relevance_score",
            "match_explanation", "linkedin_url"
        ]].rename(columns={
            "rank": "Rank", "name": "Name", "current_role": "Current Role",
            "organisation": "Organisation", "location": "Location",
            "persona_category": "Ecosystem Persona", "schools_impacted_estimate": "Schools in Network",
            "relevance_score": "Relevance Score (%)", "match_explanation": "Why This Person Matches",
            "linkedin_url": "LinkedIn URL"
        }),
        use_container_width=True
    )

    st.subheader("📇 Detailed Result Cards (With School-Ecosystem Indicators & Explanations)")
    for _, row in filtered_df.head(22).iterrows():
        with st.expander(f"#{row['rank']} — {row['name']} | {row['current_role']} ({row['organisation']}) — 🎯 {row['relevance_score']}% Match"):
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown(f"**🔗 LinkedIn Profile URL:** [{row['linkedin_url']}]({row['linkedin_url']})")
                st.markdown(f"**📍 Location:** {row['location']} | **👤 Persona:** `{row['persona_category']}`")
                st.markdown(f"**🏫 School-Ecosystem Indicators:** {row['ecosystem_indicators']}")
                st.markdown(f"**📚 Boards / Curriculum Footprint:** `{row['boards_expertise']}`")
            with col_b:
                st.success(f"**Why This Person Matches:**\n\n{row['match_explanation']}")
                st.caption(
                    f"Sub-Scores ➔ Semantic Context (45%): **{row['semantic_similarity_score']}%** | "
                    f"Multi-School Authority (35%): **{row['ecosystem_authority_score']}%** | "
                    f"Persona Alignment (20%): **{row['persona_alignment_score']}%**"
                )

with tab2:
    st.subheader("🔬 Inspect Individual Candidate vs. Reference Profile")
    chosen_cand_name = st.selectbox("Choose a Discovered Influencer:", df["name"].tolist())
    cand_row = df[df["name"] == chosen_cand_name].iloc[0]

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Overall Relevance Score", f"{cand_row['relevance_score']}%")
    sc2.metric("Semantic Context (45% wt)", f"{cand_row['semantic_similarity_score']}%")
    sc3.metric("Ecosystem Authority (35% wt)", f"{cand_row['ecosystem_authority_score']}%")
    sc4.metric("Persona Alignment (20% wt)", f"{cand_row['persona_alignment_score']}%")

    comp_df = pd.DataFrame([
        {"Dimension": "Name", "Reference Profile": ref_profile["name"], "Discovered Candidate": cand_row["name"]},
        {"Dimension": "Role & Organisation", "Reference Profile": f"{ref_profile['current_role']} ({ref_profile['organisation']})", "Discovered Candidate": f"{cand_row['current_role']} ({cand_row['organisation']})"},
        {"Dimension": "Ecosystem Persona", "Reference Profile": ref_profile["persona_category"], "Discovered Candidate": cand_row["persona_category"]},
        {"Dimension": "Estimated Schools Reached", "Reference Profile": f"{ref_profile['schools_impacted_estimate']:,}+ schools", "Discovered Candidate": f"{cand_row['schools_impacted_estimate']:,}+ schools"},
        {"Dimension": "Curriculum / Boards", "Reference Profile": ref_profile["boards_expertise"], "Discovered Candidate": cand_row["boards_expertise"]},
        {"Dimension": "Ecosystem Indicators", "Reference Profile": ref_profile["ecosystem_indicators"], "Discovered Candidate": cand_row["ecosystem_indicators"]}
    ])
    st.table(comp_df)

with tab3:
    st.markdown("""
    ### 📐 AI/ML Similarity & Multi-Factor Ranking Methodology
    Instead of relying on brittle exact keyword matching, our ranking engine (`src/ml_ranker.py`) evaluates professional context across three weighted machine learning & domain layers:
    1. **Latent Semantic Analysis & TF-IDF Cosine Similarity (45% Weight):**
       Converts each profile's role, institutional history, board expertise, and ecosystem indicators into unigram/bigram TF-IDF vectors, projects them into a dense latent semantic space via **TruncatedSVD (LSA)**, and computes **Cosine Similarity** against the reference profile. This captures conceptual equivalence (e.g., *"NPSC Chairperson"* $\leftrightarrow$ *"IPSC Boarding School Network Head"*).
    2. **Multi-School Ecosystem Authority Score (35% Weight):**
       Applies a log-scaled network reach function $\log_{10}(\text{Schools Impacted} + 1)$ combined with leadership tenure (`experience_years`) so individuals who influence **multiple schools** (school chains, CBSE/CISCE board leaders, turnkey consultants) rank higher than single-school staff.
    3. **Persona & Curriculum Overlap Score (20% Weight):**
       Measures Jaccard/set overlap across Indian K-12 board ecosystems (`CBSE`, `ICSE/ISC`, `IB`, `IGCSE`, `NEP 2020`) and alignment with the 5 target outreach personas.
    """)
