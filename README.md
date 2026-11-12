# 🏫 AI-Powered LinkedIn School Influencer Discovery Tool

Live Link (https://edxso-linkedin-school-influencer-discovery-hqn22qjd9cpysesqxdy.streamlit.app/)

<img width="951" height="478" alt="Screenshot 2026-11-12 114454" src="https://github.com/user-attachments/assets/fc5308a4-aec1-40e3-8531-0a2c973237cd" />

<img width="953" height="473" alt="Screenshot 2026-11-12 114513" src="https://github.com/user-attachments/assets/b237ad19-92a5-41dc-81c1-0b5544cfa4a4" />


Built for the **EDXSO AI Intern Assignment**. This application reduces the manual effort required to identify high-leverage decision-makers, consultants, former principals, trustees, and multi-school network leaders across the Indian K-12 school education ecosystem.

Given a single reference LinkedIn profile, the system executes an end-to-end workflow:
**`Reference Profile ➔ AI Persona & Context Analysis ➔ Similar Profile Discovery ➔ Hybrid ML Ranking ➔ Explainable Results`**

---

## 📑 Technical Note (Architecture, AI/ML Methodology & Compliance)

### 1. System Architecture
- **`src/dataset.py` (Lawfully Accessible K-12 Benchmark Pool):** Contains 26 real-world, publicly verifiable Indian K-12 ecosystem leaders across all 5 target personas (Former Principals/Directors/Trustees, K-12 Consultants, Education Entrepreneurs/School Chain CEOs, Multi-School Operators, and Apex Board/Network Connectors like CBSE/CISCE/NPSC/FICCI ARISE/SLN), plus 3 preset reference profiles.
- **`src/discovery_search.py` (Compliant Public Search X-Ray Layer):** Supports optional live public search engine X-Ray indexing (`site:linkedin.com/in/`) without ever bypassing LinkedIn authentication walls, CAPTCHAs, or rate limits.
- **`src/ml_ranker.py` (Hybrid AI/ML Similarity & Ranking Engine):** Combines `scikit-learn` TF-IDF N-gram vectorization, Latent Semantic Analysis (`TruncatedSVD`), and Cosine Similarity with a domain-specific Multi-School Authority graph score.
- **`src/database.py` (SQLite Persistence & CSV Export):** Stores the ranked profiles, sub-scores, and match explanations in `data/school_influencers.db` and exports `data/ranked_school_influencers.csv`.
- **`app.py` & `main.py` (Streamlit Cloud MVP & Rich CLI):** Interactive web application allowing users to test preset reference profiles or input any custom LinkedIn profile URL/bio.

---

### 2. AI/ML Similarity & Ranking Logic (Beyond Keyword Matching)
Relying solely on exact keyword matching fails when two K-12 leaders describe their work differently (e.g., *"Greenfield CBSE School Turnkey Advisory"* vs. *"K-12 Institutional Governance & Trust Management"*). Our `SchoolInfluencerMLRanker` computes a composite **Relevance Score (`0.0%` to `100.0%`)** using three weighted components:

1. **Contextual Semantic Similarity (45% Weight):**
   - Constructs a rich text representation combining `current_role`, `organisation`, `persona_category`, `boards_expertise`, `ecosystem_indicators`, and `bio_summary`.
   - Computes unigram and bigram **TF-IDF vectors** (`TfidfVectorizer(ngram_range=(1, 2))`) and projects them into a latent semantic concept space via **Truncated Singular Value Decomposition (`TruncatedSVD` / LSA)**.
   - Computes **Cosine Similarity** between the reference vector and every candidate vector (`0.65 * LSA_Cosine + 0.35 * Lexical_Cosine`).
2. **Ecosystem Authority & Multi-School Influence Score (35% Weight):**
   - Evaluates the scale of a professional's multi-school footprint using a logarithmic reach curve:
     $$\text{Reach Score} = \min\left(1.0, \frac{\log_{10}(\text{Schools Impacted} + 1)}{\log_{10}(5000)}\right)$$
   - Combines multi-school reach (65%) with career tenure in the education sector (35%) to prioritize professionals who influence **dozens or thousands of schools** (e.g., NPSC/IPSC chairs, school chain MDs, turnkey consultants).
3. **Persona & Curriculum Board Alignment (20% Weight):**
   - Evaluates curriculum overlap (`CBSE`, `CISCE/ICSE/ISC`, `IB`, `Cambridge IGCSE`, `NEP 2020`) and role persona alignment across the 5 target categories.

---

### 3. Data Retrieval Approach, Limitations & Compliant Path to Production
- **Why Direct LinkedIn Scraping is Avoided:** Scraping behind LinkedIn's login wall violates LinkedIn's Terms of Service (`hiQ vs. LinkedIn` / anti-bot directives) and triggers immediate IP/CAPTCHA blocks.
- **Current MVP Approach:** Uses publicly indexed professional metadata (`data/ranked_school_influencers.csv`) paired with an optional polite public search X-Ray (`site:linkedin.com/in/`) to demonstrate the complete AI ranking pipeline legally and reproducibly.
- **Compliant Path to Production:**
  1. **Authorized B2B Data Enrichment APIs:** Integrate **Proxycurl API**, **People Data Labs (PDL)**, or **Apollo.io / Clay** via official enterprise licenses to fetch fresh public profile JSON by LinkedIn URL.
  2. **Google Custom Search JSON API (Programmable Search Engine):** Execute automated, rate-limited `site:linkedin.com/in/` X-Ray discovery queries for K-12 personas across Indian cities via Google's official paid API.
  3. **Dense Bi-Encoder & Cross-Encoder Reranking:** Upgrade the LSA/TF-IDF layer to `sentence-transformers` (`BAAI/bge-large-en-v1.5`) stored in `FAISS` / `pgvector` for sub-millisecond retrieval across 100,000+ education profiles.

---

## 🚀 Local Setup & Execution Instructions

1. **Install dependencies:**
   `pip install -r requirements.txt`
2. **Run the CLI Discovery & Ranking Pipeline (Generates SQLite DB & CSV):**
   `python main.py`
3. **Launch the Interactive Web Application:**
   `streamlit run app.py`
