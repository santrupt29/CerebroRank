# Academic Project Report Skeleton & Rubric Mapping

> **Project Title**: Influence Ranking in the Marvel Wikipedia Link Graph using PageRank and HITS  
> **Course**: Web Data Mining — B.Tech Computer Engineering, Semester VII  
> **Institution**: Sardar Patel College of Engineering (SPCE), Mumbai  
> **Target Length**: 8–10 Pages (Standard IEEE / ACM 2-column or Single-column Technical Format)

---

## Rubric & Page Allocation Overview

| Section | Target Length | Core Artifacts from `reports/` | Primary Objective |
|:---|:---:|:---|:---|
| **1. Introduction & Problem Statement** | 0.75 – 1.0 Page | Overview stats from `reports/summary.md` | Define motivation, research questions, and scope. |
| **2. Related Work & Background** | 0.75 – 1.0 Page | Equations and theoretical contrast table | Review Brin & Page (1998) and Kleinberg (1999). |
| **3. Data Collection & Ethics** | 1.0 Page | `crawl_report.json`, `ETHICS.md` | Document hybrid BFS, robots.txt, and rate limits. |
| **4. Preprocessing & Graph Construction** | 1.0 Page | `data_quality.json`, topological stats | Navbox stripping, WCC filtering, CSR assembly. |
| **5. Algorithms: Math & Implementation** | 1.5 Pages | Mathematical derivations & algorithm boxes | Custom power-iteration, dangling fix, and SpMV. |
| **6. Ground-Truth Validation** | 0.75 – 1.0 Page | `validation.json`, validation table | Strict MSE and correlation vs NetworkX reference. |
| **7. Results & Visualization** | 2.0 Pages | Figures 01–08, Top-10 tables | Detailed analysis of convergence, damping, and PPR. |
| **8. Discussion, Bias & Limitations** | 1.0 Page | Figure 09 (`09_bias_limitations.png`), scatter | Structural shortcuts, meta-pages, negative results. |
| **9. Conclusion & Future Work** | 0.5 Page | Findings summary | Summary of insights and future temporal extensions. |
| **10. References** | 0.5 Page | Formal citations | Academic bibliography. |

---

## Detailed Section-by-Section Blueprint

### Section 1: Introduction and Problem Statement
- **Target Length**: ~1.0 Page
- **Figures / Tables to Include**:
  - High-level topology summary table (from `reports/summary.md` Section 1).
- **Key Points to Write**:
  - *Context*: Web link analysis treats hyperlinks as implicit endorsements. In encyclopaedic graphs like Wikipedia, determining the relative historical and cultural importance of fictional entities requires separating general popularity from structural centrality.
  - *Problem Definition*: How do random-walk stochastic algorithms (PageRank) differ from bipartite mutual recursion (HITS) when applied to dense, highly cross-referenced fictional universes like Marvel?
  - *Dataset Scope*: English Wikipedia link graph initialized from core seeds (*Marvel Cinematic Universe*, *Marvel Comics*, *Marvel Studios*, *Iron Man*, *Spider-Man*, *X-Men*) and bounded within verified category trees.
  - *Key Numbers*: Isolated giant connected component of $|V| = 100$ articles and $|E| = 381$ directed edges with average clustering coefficient $C = 0.6552$ and density $D = 0.0385$.

---

### Section 2: Related Work & Theoretical Background
- **Target Length**: ~1.0 Page
- **Figures / Tables to Include**:
  - Comparative table contrasting PageRank vs. HITS (random walk vs. mutual recursion, query-dependent vs. static, normalization types).
- **Key Points to Write**:
  - *PageRank Foundation (Brin & Page, 1998)*: Formulates influence as the stationary distribution of an ergodic Markov chain. The random surfer follows out-links with probability $d$ (damping factor) and teleports uniformly with probability $1-d$.
  - *HITS Foundation (Kleinberg, 1999)*: Deconstructs web topology into dual roles: *Authorities* (content-rich nodes cited by multiple hubs) and *Hubs* (index pages pointing to multiple authoritative sources) via principal eigenvectors of $A A^T$ and $A^T A$.
  - *Distinction from Degree Centrality*: Pure in-degree measures raw citation count without evaluating source prominence; both PageRank and HITS recursively weight citation quality.

---

### Section 3: Data Collection and Ethics
- **Target Length**: ~1.0 Page
- **Figures / Tables to Include**:
  - Table of crawl statistics from `reports/crawl_report.json` (Pages fetched: 94, API calls: 94, Cache hits: 262, Robots skipped: 0, Retries: 0, Runtime: 128.4s).
- **Key Points to Write**:
  - *Hybrid Crawling Strategy*: Phase 1 queries MediaWiki `categorymembers` API up to depth 2 to generate an allowed set of 8,447 Marvel articles; Phase 2 performs bounded BFS from seed articles restricted strictly to allowed titles.
  - *Edge Policy*: An edge $u \to v$ is recorded if and only if both $u \in V$ and $v \in V$ are accepted nodes, preventing artificial dangling sinks on the frontier.
  - *Ethical Enforcement*: Enforces `urllib.robotparser` against live `robots.txt` before fetching; declarative academic User-Agent; randomized uniform rate limiting ($1.0 - 1.5$s); local caching in `data/cache/` to ensure zero repeat network traffic.

---

### Section 4: Preprocessing and Graph Construction
- **Target Length**: ~1.0 Page
- **Figures / Tables to Include**:
  - Data Quality Table from `reports/data_quality.json`.
- **Key Points to Write**:
  - *DOM Sanitization*: Strips `.navbox`, `.vertical-navbox`, `.reflist`, `.hatnote`, and metadata elements prior to link parsing; drops trailing sections (*References*, *External links*, *See also*) to isolate body-text citations.
  - *Self-Loop Removal*: Strips all identity edges ($u \to u$) to prevent trivial diagonal accumulation in transition matrices ($0$ self-loops in final graph).
  - *Giant Weakly Connected Component (WCC)*: Filters isolated components ($100\%$ of raw nodes belong to the giant component) to ensure stochastic convergence across all states.
  - *Sparse Matrix Storage*: Compresses adjacency $A$ into Scipy Compressed Sparse Row (CSR) format ($N \times N$, 381 non-zero entries) and column-stochastic transition matrix $M$ where $M_{ji} = 1 / k_{out}(i)$.

---

### Section 5: Algorithms (Math and Implementation)
- **Target Length**: ~1.5 Pages
- **Figures / Tables to Include**:
  - Matrix transition equations and algorithmic pseudo-code boxes for PageRank and HITS power iteration.
- **Key Points to Write**:
  - *PageRank Power Iteration*:
    $$PR_{t+1} = d \left( M \cdot PR_t + \frac{\sum_{i \in \text{Dangling}} PR_t(i)}{N} \mathbf{1} \right) + \frac{1-d}{N} \mathbf{1}$$
    Solves dangling node mass leakage by gathering probability at out-degree 0 nodes and redistributing uniformly.
  - *HITS Alternating Iteration*:
    $$a_{t+1} = A^T h_t, \quad a_{t+1} \leftarrow \frac{a_{t+1}}{\|a_{t+1}\|_2}; \qquad h_{t+1} = A a_{t+1}, \quad h_{t+1} \leftarrow \frac{h_{t+1}}{\|h_{t+1}\|_2}$$
    Uses $L_2$ normalization to scale the principal eigenvectors of $A^T A$ and $A A^T$.
  - *Complexity Analysis*: CSR sparse matrix-vector multiplication (SpMV) runs in $O(|E|)$ per iteration, consuming $O(|V| + |E|)$ memory, scaling linearly rather than dense $O(|V|^2)$.

---

### Section 6: Validation
- **Target Length**: ~0.75 – 1.0 Page
- **Figures / Tables to Include**:
  - Validation Table from `reports/validation.json` (MSE, Spearman $\rho$, Kendall $\tau$, Top-10 overlap).
- **Key Points to Write**:
  - *Ground-Truth Cross-Check*: Custom implementations compared directly against NetworkX reference methods (`nx.pagerank` and `nx.hits`) on the exact same graph and parameters ($d=0.85$, $tol=10^{-6}$).
  - *PageRank Accuracy*: Achieves Mean Squared Error of $4.01 \times 10^{-13}$, Spearman rank correlation $\rho = 1.0000$, Kendall $\tau = 1.0000$, and exact $10/10$ top-10 rank alignment.
  - *HITS Accuracy*: Authority MSE is $2.37 \times 10^{-19}$ ($\rho = 1.0000$), Hub MSE is $4.54 \times 10^{-21}$ ($\rho = 1.0000$), confirming that custom CSR mathematical implementations match reference ground-truth to floating-point precision.

---

### Section 7: Results and Visualization
- **Target Length**: ~2.0 Pages
- **Figures / Tables to Include**:
  - `reports/figures/01_convergence_curves.png` (Convergence dynamics).
  - `reports/figures/02_damping_sensitivity.png` (Damping matrix and bump chart).
  - `reports/figures/03_metric_comparison.png` (Correlation heatmaps & Jaccard overlap).
  - `reports/figures/04_hubs_authorities_pagerank.png` (Hub vs Authority scatter plot).
  - `reports/figures/05_personalized_pagerank.png` (MCU vs X-Men PPR).
  - `reports/figures/06_community_detection.png` (Louvain community distribution).
  - `reports/figures/07_node_type_analysis.png` (Node-type bar chart).
  - Top-10 leaderboards for PageRank, Authority, Hub, and In-Degree.
- **Key Points to Write**:
  - *Convergence*: PageRank converges in 17 iterations; HITS converges in 10 iterations.
  - *Damping Analysis*: Damping parameter $d \in [0.5, 0.95]$ demonstrates monotonic iteration growth ($11 \to 14 \to 17 \to 20$ iters) while preserving high ranking stability ($\rho > 0.999$).
  - *Rank Separation*: Top PageRank entity is `Marvel_Comics` ($0.1995$), followed by `X-Men` ($0.1032$), `Spider-Man` ($0.1031$), and `Marvel_Cinematic_Universe` ($0.0762$).
  - *Personalized PageRank*: Teleportation to MCU nodes elevates `Marvel_Cinematic_Universe` to #2 and doubles `Marvel_Studios`; teleportation to X-Men elevates `X-Men` to #2 and pulls `Mutant_(Marvel_Comics)` into the top 5.
  - *Community Structure*: Louvain identifies 4 coherent modular communities cleanly separating comic publishing, character rosters, media productions, and mutant series.

---

### Section 8: Discussion, Bias, and Limitations
- **Target Length**: ~1.0 Page
- **Figures / Tables to Include**:
  - `reports/figures/09_bias_limitations.png` (Full vs Filtered graph without lists/persons).
  - Archetype Table from `reports/tables/04_archetype_examples.csv`.
- **Key Points to Write**:
  - *The Negative Result Phenomenon*: `Tsunami_(Marvel_Comics)` ranks #13 in PageRank but drops to #28 in Authority. Its incoming links originate from nodes with a low average hub score ($0.0632$). PageRank accumulates walk probability along longer multi-hop paths, whereas HITS demands direct incoming links from high-scoring hubs.
  - *Meta-Page Structural Bias*: Removing `list` and `person` nodes (4.0% of nodes) eliminates artificial cross-franchise bridges, allowing diegetic narrative continuity to govern PageRank and shifting `Marvel_Edge` into the top-10.
  - *Limitations*: Graph snapshot does not reflect real-time fandom edits; BFS horizon bounds depth to 3 hops.

---

### Section 9: Conclusion and Future Work
- **Target Length**: ~0.5 Page
- **Figures / Tables to Include**: None (summary prose).
- **Key Points to Write**:
  - *Core Conclusions*: PageRank and HITS uncover complementary structural insights in encyclopaedic networks: HITS cleanly isolates navigation directories from content anchors, while PageRank models global multi-hop authority.
  - *Future Directions*: Expanding to dynamic temporal link graphs across Wikipedia revision histories; incorporating weighted NLP text embeddings on link anchor contexts.

---

### Section 10: References
- **Target Length**: ~0.5 Page
- **Citations to Format**:
  1. Brin, S., & Page, L. (1998). *The anatomy of a large-scale hypertextual Web search engine*. Computer Networks and ISDN Systems, 30(1-7), 107-117.
  2. Kleinberg, J. M. (1999). *Authoritative sources in a hyperlinked environment*. Journal of the ACM (JACM), 46(5), 604-632.
  3. Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). *Fast unfolding of communities in large networks*. Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008.
  4. Langville, A. N., & Meyer, C. D. (2006). *Google's PageRank and Beyond: The Science of Search Engine Rankings*. Princeton University Press.
  5. Wikimedia Foundation. (2026). *Wikimedia Foundation Terms of Use and Robots Policy*.
