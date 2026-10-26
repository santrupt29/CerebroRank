# MarvelGraphRank: Executive Evaluation Summary

## 1. Network Topology & Key Metrics

| Metric | Value | Meaning |
|:---|:---:|:---|
| **Total Nodes ($|V|$)** | 100 | Total articles in giant connected component |
| **Total Edges ($|E|$)** | 381 | Directed hyperlinks between Marvel articles |
| **Network Density** | 0.038485 | Ratio of observed edges to possible edges |
| **Avg Clustering Coefficient** | 0.6552 | Local triadic closure and clustering |
| **Giant WCC Size** | 100 (100.0%) | Primary weakly-connected component coverage |
| **PageRank Convergence** | 17 iters | Power iteration iterations to reach $L_1 < 10^{-6}$ |
| **HITS Convergence** | 10 iters | Kleinberg alternating updates to convergence |
| **PageRank MSE vs NetworkX** | 4.01e-13 | Numerical precision vs reference implementation |
| **PageRank Spearman $\rho$** | 1.0000 | Rank alignment with NetworkX |
| **HITS Auth MSE vs NetworkX** | 2.37e-19 | Authority numerical precision vs NetworkX |
| **HITS Hub MSE vs NetworkX** | 4.54e-21 | Hub numerical precision vs NetworkX |

---

## 2. Top-10 Ranking Tables

### Top-10 by PageRank
| Rank | Title | Pagerank | Node Type | In-Deg | Out-Deg |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | `Marvel_Comics` | 0.199456 | comic_series | 98 | 9 |
| 2 | `X-Men` | 0.103185 | character | 45 | 7 |
| 3 | `Spider-Man` | 0.103117 | character | 43 | 8 |
| 4 | `Marvel_Cinematic_Universe` | 0.076222 | company | 40 | 6 |
| 5 | `Marvel_Studios` | 0.046655 | film | 9 | 7 |
| 6 | `Iron_Man` | 0.045698 | character | 28 | 4 |
| 7 | `Spider-Man:_No_Way_Home` | 0.035989 | character | 13 | 5 |
| 8 | `Mutant_(Marvel_Comics)` | 0.029361 | character | 30 | 5 |
| 9 | `Ronald_Perelman` | 0.026188 | other | 2 | 1 |
| 10 | `Uncle_Ben` | 0.022994 | character | 8 | 4 |

### Top-10 by HITS Authority
| Rank | Title | Authority | Node Type | In-Deg | Out-Deg |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | `Marvel_Comics` | 0.627497 | comic_series | 98 | 9 |
| 2 | `X-Men` | 0.378627 | character | 45 | 7 |
| 3 | `Spider-Man` | 0.368399 | character | 43 | 8 |
| 4 | `Marvel_Cinematic_Universe` | 0.354960 | company | 40 | 6 |
| 5 | `Iron_Man` | 0.271811 | character | 28 | 4 |
| 6 | `Mutant_(Marvel_Comics)` | 0.256021 | character | 30 | 5 |
| 7 | `Spider-Man:_No_Way_Home` | 0.126659 | character | 13 | 5 |
| 8 | `Knull` | 0.100158 | character | 8 | 2 |
| 9 | `Clone_Saga` | 0.100110 | other | 9 | 4 |
| 10 | `Ann_Nocenti` | 0.094573 | comic_series | 9 | 6 |

### Top-10 by HITS Hub
| Rank | Title | Hub | Node Type | In-Deg | Out-Deg |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | `Spider-Woman_(Charlotte_Witter)` | 0.191410 | character | 1 | 11 |
| 2 | `Set_(Marvel_Comics)` | 0.191410 | character | 1 | 11 |
| 3 | `Sabreclaw` | 0.191410 | character | 0 | 11 |
| 4 | `Rex_Strickland` | 0.191410 | character | 2 | 11 |
| 5 | `Gloria_Morales` | 0.173588 | character | 0 | 9 |
| 6 | `Carl_Mach` | 0.171158 | character | 0 | 9 |
| 7 | `Antiphon_(comics)` | 0.167190 | character | 0 | 9 |
| 8 | `Lauri-Ell` | 0.164722 | character | 0 | 7 |
| 9 | `Professor_Oyama` | 0.164722 | character | 0 | 7 |
| 10 | `Sons_of_the_Tiger` | 0.158498 | character | 1 | 7 |

### Top-10 by In-Degree
| Rank | Title | In_degree | Node Type | In-Deg | Out-Deg |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | `Marvel_Comics` | 98 | comic_series | 98 | 9 |
| 2 | `X-Men` | 45 | character | 45 | 7 |
| 3 | `Spider-Man` | 43 | character | 43 | 8 |
| 4 | `Marvel_Cinematic_Universe` | 40 | company | 40 | 6 |
| 5 | `Mutant_(Marvel_Comics)` | 30 | character | 30 | 5 |
| 6 | `Iron_Man` | 28 | character | 28 | 4 |
| 7 | `Spider-Man:_No_Way_Home` | 13 | character | 13 | 5 |
| 8 | `Clone_Saga` | 9 | other | 9 | 4 |
| 9 | `Ann_Nocenti` | 9 | comic_series | 9 | 6 |
| 10 | `Marvel_Studios` | 9 | film | 9 | 7 |

---

## 3. Data-Driven Findings & Analytical Insights

### 1. PageRank Decouples from Raw In-Degree by Weighting Citation Quality

While In-Degree counts every hyperlink identically, PageRank weights citations by the global authority of the citing page. For example, `Marvel_Comics` achieves the top PageRank rank (0.199456) because it receives links from high-scoring franchise pillars. In contrast, `Tsunami_(Marvel_Comics)` displays a stark rank divergence between PageRank and HITS Authority, demonstrating that random walk probability accumulation differs systematically from bipartite mutual recursion.

### 2. Distinct Functional Duality Between Hubs and Authorities

The network exhibits clean bipartite role separation. Informational and compilation pages act as dominant hubs (e.g., `Spider-Woman_(Charlotte_Witter)` with hub score 0.191410), effectively routing navigation across the universe. Conversely, core intellectual property anchors dominate authority scores (e.g., `Marvel_Comics` with authority 0.627497), receiving dense inbound endorsements while publishing few outbound links.

### 3. Louvain Modularity Uncovers Cohesive Thematic Sub-Franchises

Unsupervised community detection (Louvain, seed=42) partitioned the 100-node Marvel graph into 4 tight modular clusters. The largest cluster (C1: Marvel Comics & Characters) captures 34 nodes (34.0% of the network), cleanly segmenting characters, cinematic universe productions, and comic publishing history without manual supervision.

### 4. Topic-Personalized PageRank Steers Authority into Thematic Subgraphs

When teleportation is restricted to MCU nodes or X-Men nodes, the top-tier rankings sharply reorient toward those sub-domains. Under MCU teleportation, cinematic universe productions and franchise leads experience substantial score increases, while comic-exclusive mutants decline. Conversely, X-Men teleportation elevates mutant characters and comic storylines, proving personalized random walks can effectively filter large knowledge graphs by contextual interest.

### 5. Meta-Pages (Lists & Persons) Exert Structural Bridging Bias

Structural Bias Analysis: Dropping 'list' and 'person' categories removed 4 nodes (4.0% of the graph). In the filtered graph, new entries entering the top-10 include: Marvel_Edge. List articles act as high-degree hub bridges connecting disconnected thematic clusters, while real-world creator/actor nodes introduce cross-media shortcuts. Removing them eliminates artificial bridges, causing PageRank to concentrate strictly along narrative, diegetic comic/cinematic continuity.


---
*Generated deterministically by `core/evaluator.py` without network access.*
