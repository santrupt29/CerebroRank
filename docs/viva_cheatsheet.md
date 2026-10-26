# MarvelGraphRank: Comprehensive Viva Cheatsheet & Defense Guide

> **Course**: Web Data Mining — B.Tech Computer Engineering, Semester VII  
> **Institution**: Sardar Patel College of Engineering (SPCE), Mumbai  
> **Project**: Influence Ranking in the Marvel Wikipedia Link Graph using PageRank and HITS

---

## Part 1: Member Responsibilities & "Who Explains What"

| Member | Core Ownership Area | Viva Defense Topics to Lead |
|:---|:---|:---|
| **[MEMBER 1]** | **Crawler & Ethics** | Questions 10, 14, 15, 23, 24, 25 (BFS crawling, allowed-set gating, robots.txt parsing, rate limiting, Wikimedia ethics, cache management). |
| **[MEMBER 2]** | **Graph Builder & CSR Topology** | Questions 2, 3, 8, 9, 16, 17 (Navbox stripping, CSR sparse format, self-loops, giant WCC component filtering, spider-trap/dangling demo). |
| **[MEMBER 3]** | **Algorithms & Validation** | Questions 1, 4, 5, 6, 7, 18 (PageRank derivation, Perron-Frobenius convergence, HITS mutual recursion, $L_1$ vs $L_2$ norms, NetworkX validation). |
| **[MEMBER 4]** | **Evaluator, Dashboard & Findings** | Questions 11, 12, 13, 19, 20, 21, 22 (Personalized PageRank, Louvain modularity, Streamlit dashboard, archetype analysis, negative results, bias analysis). |

---

## Part 2: 25+ High-Yield Viva Questions & Mathematical Answers

### 1. Why do we need the damping factor $d$ in PageRank?
**Answer**:  
The raw transition matrix $M$ is not guaranteed to be irreducible (strongly connected) or aperiodic. By convexly combining $M$ with a uniform teleportation matrix $E = \frac{1}{N} \mathbf{1} \mathbf{1}^T$:
$$G = d M + (1 - d) \frac{1}{N} \mathbf{1} \mathbf{1}^T$$
the Google matrix $G$ becomes **strictly positive** ($G_{ij} > 0$), making the underlying Markov chain primitive, irreducible, and aperiodic. This guarantees the existence of a unique, strictly positive stationary probability distribution $\pi = G \pi$ independent of initial state.

---

### 2. What are dangling nodes, why do they cause probability leakage, and how are they fixed?
**Answer**:  
A dangling node is a vertex with out-degree $k_{out} = 0$. Its column in the transition matrix $M$ consists entirely of zeros ($M_{*, i} = \mathbf{0}$). In naive power iteration ($v_{t+1} = M v_t$), probability mass entering a dangling node vanishes, causing the vector sum to decay toward zero:
$$\sum_{j=1}^N (M v_t)_j = \sum_{j \notin \text{Dangling}} v_t(j) < 1$$
**The Fix**: In each iteration, we gather the total dangling mass $m_d = \sum_{i \in \text{Dangling}} v_t(i)$ and redistribute it uniformly across all $N$ nodes:
$$v_{t+1} = d \left( M v_t + \frac{m_d}{N} \mathbf{1} \right) + \frac{1-d}{N} \mathbf{1}$$
This guarantees $\sum_{j=1}^N v_{t+1}(j) = 1.0$ at every step.

---

### 3. What is a spider trap, and how does PageRank resolve it?
**Answer**:  
A spider trap is a closed loop of nodes (e.g. $C \leftrightarrow D$) with inbound links from the rest of the graph but zero outbound links to outside nodes. Under repeated transitions without teleportation, all probability mass eventually flows into the trap and gets permanently absorbed, leaving all other nodes with score 0.  
**Resolution**: The random teleportation probability $(1-d) = 0.15$ ensures that with probability $15\%$ at every step, the random surfer escapes the trap by jumping uniformly to any node in the network.

---

### 4. Why does PageRank power iteration always converge mathematically?
**Answer**:  
By the **Perron-Frobenius Theorem**, any irreducible, primitive, column-stochastic matrix $G$ has a unique dominant eigenvalue $\lambda_1 = 1$, and all other eigenvalues satisfy $|\lambda_i| \le d < 1$.  
Expanding the initial vector $v_0$ in the eigenvector basis $v_0 = u_1 + c_2 u_2 + \dots + c_N u_N$:
$$G^t v_0 = u_1 + \sum_{i=2}^N c_i \lambda_i^t u_i$$
As $t \to \infty$, since $|\lambda_i| \le d$, the error decays exponentially as $O(d^t)$:
$$\|v_t - u_1\|_1 \le C \cdot d^t$$
For $d = 0.85$, $0.85^{17} \approx 0.063$, guaranteeing rapid convergence to the tolerance ($10^{-6}$) in 17 iterations.

---

### 5. Why is the time complexity of each iteration $O(|E|)$ rather than $O(|V|^2)$?
**Answer**:  
Dense matrix-vector multiplication $M v$ requires $|V|^2$ operations. However, web graphs are extremely sparse ($|E| \ll |V|^2$; in our graph, $|V|=100, |E|=381$, density $\approx 3.8\%$).  
Using **Compressed Sparse Row (CSR)** format, $M v$ iterates only over non-zero matrix entries (`indices` and `data`), executing exactly $|E|$ multiplications and $|E|$ additions. The dangling sum requires $|V_{dangling}|$ operations, and scalar additions take $O(|V|)$. Thus, each iteration takes $O(|V| + |E|) = O(|E|)$ time.

---

### 6. What are the key differences between PageRank and HITS?
**Answer**:  

| Feature | PageRank | HITS (Kleinberg) |
|:---|:---|:---|
| **Underlying Model** | Random surfer on a Markov chain | Bipartite mutual reinforcement between dual roles |
| **Output** | Single stationary distribution vector ($\pi$) | Two vectors: Authority ($a$) and Hub ($h$) |
| **Mathematical Nature** | Dominant eigenvector of stochastic matrix $G$ | Singular vectors of adjacency $A$ ($A^T A$ and $A A^T$) |
| **Normalization** | $L_1$ norm ($\sum v_i = 1$) | $L_2$ norm ($\|a\|_2 = 1, \|h\|_2 = 1$) |
| **Topic Dependency** | Global (query-independent by default) | Traditionally query-dependent (root + base set) |

---

### 7. Why does HITS use $L_2$ normalization while PageRank uses $L_1$ normalization?
**Answer**:  
- **PageRank**: Represents a discrete probability distribution over states in a Markov chain. The total probability must sum to 1.0, requiring the $L_1$ norm: $\|v\|_1 = \sum |v_i| = 1.0$.
- **HITS**: Does not model probabilities. The alternating update equations:
  $$a_{t+1} = A^T h_t = A^T A a_t, \qquad h_{t+1} = A a_{t+1} = A A^T h_t$$
  correspond to the power method computing the dominant eigenvectors of symmetric matrices $A^T A$ and $A A^T$. Standard numerical linear algebra requires the Euclidean norm ($L_2$) to prevent vector magnitudes from exploding to $\infty$ or vanishing to $0$: $\|v\|_2 = \sqrt{\sum v_i^2} = 1.0$.

---

### 8. Why do Hubs and Authorities differ from Out-Degree and In-Degree?
**Answer**:  
- In-degree and out-degree are local, unweighted counts: every hyperlink contributes equally ($+1$).
- **HITS Authority**: An inbound link from a node with a high Hub score contributes substantially more than links from hundreds of low-hub or dead-end pages: $a(v) = \sum_{u \to v} h(u)$.
- **HITS Hub**: An outbound link to an authoritative node contributes significantly more than links to obscure pages: $h(u) = \sum_{u \to v} a(v)$.

---

### 9. What is the effect of Wikipedia navboxes on link graph analysis?
**Answer**:  
Wikipedia navigation templates (navboxes at the bottom of pages) contain comprehensive lists of 50–150 articles associated with a franchise. If parsed:
- Every page in a navbox links to every other page in the navbox, creating **dense artificial cliques**.
- Minor tertiary characters receive incoming links from every major character, artificially inflating their in-degree and PageRank to parity with major protagonists.
- Stripping `.navbox`, `.vertical-navbox`, and `table.navbox` in `core/graph_builder.py` ensures that links represent genuine editorial body citations.

---

### 10. Why does our crawler edge policy keep only edges between crawled nodes?
**Answer**:  
If we recorded edges $u \to v$ where $v$ is an uncrawled frontier page:
- $v$ would have out-degree $k_{out}(v) = 0$ simply because the crawler halted before fetching it.
- This would inject hundreds of **artificial dangling nodes**, distorting the transition matrix and creating artificial sinks.
- Enforcing that both endpoints $u, v \in V_{accepted}$ ensures that degrees reflect the true crawled sub-network.

---

### 11. How does Personalized PageRank (PPR) work mathematically?
**Answer**:  
Standard PageRank teleports uniformly: $p = \frac{1}{N} \mathbf{1}$.  
Personalized PageRank replaces the uniform restart vector with a non-uniform probability distribution $p$:
$$v_{t+1} = d M v_t + (1 - d) p$$
where $p$ concentrates probability mass on specific target seed nodes (e.g., $p_i = \frac{1}{|S|}$ for $i \in S$, and $0$ elsewhere). This biases the stationary distribution toward nodes reachable within short random-walk distances from $S$.

---

### 12. Why did we use Louvain community detection rather than Girvan-Newman or Spectral clustering?
**Answer**:  
- **Louvain**: Heuristic modularity-maximization algorithm running in $O(|V| \log |V|)$ time. It handles large networks efficiently and determines the optimal number of communities automatically without specifying $k$ in advance.
- **Girvan-Newman**: Relies on edge betweenness centrality recalculation after every edge removal, with prohibitive $O(|E|^2 |V|)$ complexity.
- **Spectral Clustering**: Requires computing the eigendecomposition of the graph Laplacian, which scales as $O(|V|^3)$ unless sparse approximations are used, and requires pre-specifying cluster count $k$.

---

### 13. What is a "Negative Result" in our analysis and what causes it?
**Answer**:  
In `core/evaluator.py`, an entity like `Tsunami_(Marvel_Comics)` ranks #13 in PageRank but drops to #28 in HITS Authority.  
**Cause**: The node receives links from pages that have near-zero Hub scores. PageRank accumulates mass through recursive random walks across multiple hops and from uniform teleportation. In contrast, HITS Authority strictly requires incoming links from high-scoring Hubs ($a(v) = \sum_{u \to v} h(u)$). When inbound citations originate from non-hub pages, HITS assigns near-zero authority despite high PageRank.

---

### 14. How is robots.txt technically enforced in our crawler?
**Answer**:  
`crawler/wiki_crawler.py` initializes `urllib.robotparser.RobotFileParser`. On first execution, it fetches `https://en.wikipedia.org/robots.txt` and caches it in `data/cache/_robots.txt`. Prior to issuing any HTML request, `_can_fetch(url)` calls `self.robot_parser.can_fetch(user_agent, url)`. If blocked, the crawler drops the URL and logs the event.

---

### 15. How did we validate our from-scratch PageRank and HITS against NetworkX?
**Answer**:  
In `core/validation.py`, we executed `nx.pagerank(G, alpha=d, tol=tol)` and `nx.hits(G, tol=tol)` on the identical graph. We evaluated four metrics:
1. **Mean Squared Error (MSE)**: $\frac{1}{N} \sum (v_{custom} - v_{nx})^2 < 10^{-12}$.
2. **Spearman Rank Correlation ($\rho$)**: Measures monotonic rank ordering ($\rho = 1.0000$).
3. **Kendall Tau ($\tau$)**: Pairwise concordant vs discordant rank ordering ($\tau = 1.0000$).
4. **Top-10 Ranking Overlap**: $10/10$ identical nodes in the top tier.

---

### 16. What is the Compressed Sparse Row (CSR) format and why is it optimal for PageRank?
**Answer**:  
CSR represents an $N \times N$ matrix using three 1D arrays:
1. `data`: Array of non-zero values (length $|E|$).
2. `indices`: Column indices of corresponding elements (length $|E|$).
3. `indptr`: Row index pointers (length $|V| + 1$), where row $i$ elements span `indices[indptr[i] : indptr[i+1]]`.  
CSR provides $O(1)$ row slicing, contiguous cache-friendly memory access during matrix-vector multiplication, and reduces memory from $N^2 \times 8$ bytes to $(2|E| + |V| + 1) \times 8$ bytes.

---

### 17. Why do we filter for the Giant Weakly Connected Component (WCC)?
**Answer**:  
If the graph contains disconnected components or isolated singletons:
- Nodes in disconnected subgraphs can never exchange random-walk probability with the main graph.
- Isolating the Giant WCC ($100\%$ of our 100-node graph) ensures every node is structurally reachable in the undirected projection, providing a well-conditioned graph for community detection and ranking.

---

### 18. What is the Markov chain interpretation of PageRank?
**Answer**:  
We define a finite-state discrete-time Markov chain where states correspond to Wikipedia articles, and transition probabilities are given by the stochastic matrix $G$. A state vector $\pi_t$ represents the probability distribution of the surfer's position at step $t$. The stationary distribution satisfies:
$$\pi = \pi G \quad \iff \quad G^T \pi = \pi$$
PageRank is the unique normalized left-eigenvector of $G$ with eigenvalue $\lambda = 1$.

---

### 19. Why do List pages have high Hub scores but low Authority scores?
**Answer**:  
List articles (e.g. *List of Marvel Cinematic Universe films*) curate and link to dozens of individual character and film articles, giving them large out-degrees to authoritative targets. However, individual character articles rarely hyperlink back to general list index pages in their narrative body text. Consequently, lists receive near-zero inbound citations, resulting in minimal authority scores.

---

### 20. How does damping factor $d$ affect convergence speed and ranking?
**Answer**:  
The asymptotic convergence rate is governed by the second eigenvalue $|\lambda_2| \le d$.
- When $d=0.5$, $|\lambda_2| \le 0.5$, error shrinks rapidly ($0.5^{11} \approx 0.00049$), converging in 11 iterations.
- When $d=0.95$, error decays slowly ($0.95^{20} \approx 0.358$), requiring 20 iterations.
Lower $d$ values blur graph topology toward the uniform distribution, while higher $d$ values allow deep multi-hop paths to dictate prestige.

---

### 21. How did we demonstrate the structural bias of meta-pages?
**Answer**:  
In `core/evaluator.py`, we created a filtered subgraph removing `list` and `person` nodes (4.0% of the graph). Recomputing PageRank revealed that list pages and creator/actor nodes act as non-diegetic shortcuts bridging unrelated comic and film franchises. Removing them forced PageRank to flow strictly through narrative storylines, elevating character entities like `Marvel_Edge`.

---

### 22. How does the Streamlit dashboard maintain snappy performance offline?
**Answer**:  
- Uses `@st.cache_resource` for the static `Config` and `GraphData` objects.
- Uses `@st.cache_data` for PageRank computations parameterized by damping $d$. Shifting the slider triggers instant lookups for previously calculated values.
- Precomputed HITS and Louvain communities are cached in memory; all assets are loaded locally with zero network I/O.

---

### 23. What are the advantages of using the MediaWiki API over raw HTML scraping?
**Answer**:  
The MediaWiki API (`/w/api.php`) returns clean JSON with pre-parsed metadata. Querying `list=categorymembers` with `cmlimit=500` allows discovering 500 categorized articles in a single HTTP request, whereas scraping category HTML would require parsing pagination links and processing megabytes of presentation markup.

---

### 24. What happens when Wikipedia returns an HTTP 429 Too Many Requests status?
**Answer**:  
Our crawler catches HTTP 429 and 503 errors in `_get_with_retry()`. It inspects the `Retry-After` header. If present, it sleeps for the designated seconds; otherwise, it applies exponential back-off ($wait = \min(2^{attempt+1}, 120)$s). It increments `stats["backoff_events"]` and retries up to 5 times before failing gracefully.

---

### 25. How does the project guarantee that running the evaluation pipeline twice gives identical results?
**Answer**:  
1. All random processes in graph algorithms (Louvain community detection) are pinned with deterministic seeds (`seed=42`).
2. PageRank and HITS power iterations initialize with deterministic uniform vectors ($1/N$).
3. All file outputs (figures, CSVs, markdown) write deterministic floats formatted to consistent decimal precisions.
4. Verified via MD5 checksum hashing across all 16 CSV tables and `summary.md`.

---

### 26. Quick Formula Cheat Sheet

| Metric | Governing Formula | Key Invariant |
|:---|:---|:---|
| **PageRank ($v$)** | $v_{t+1} = d \left( M v_t + \frac{m_d}{N} \mathbf{1} \right) + \frac{1-d}{N} \mathbf{1}$ | $\|v\|_1 = \sum_{i=1}^N v_i = 1.0$ |
| **HITS Authority ($a$)** | $a_{t+1} = \frac{A^T h_t}{\|A^T h_t\|_2}$ | $\|a\|_2 = \sqrt{\sum a_i^2} = 1.0$ |
| **HITS Hub ($h$)** | $h_{t+1} = \frac{A a_{t+1}}{\|A a_{t+1}\|_2}$ | $\|h\|_2 = \sqrt{\sum h_i^2} = 1.0$ |
| **Graph Density ($D$)** | $D = \frac{\|E\|}{\|V\|(\|V\| - 1)}$ | $0 \le D \le 1$ |
| **Clustering Coeff ($C$)**| $C_i = \frac{2 T_i}{k_i(k_i - 1)}$ | Ratio of actual triangles to possible triangles |
| **Modularity ($Q$)** | $Q = \frac{1}{2m} \sum_{i,j} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$ | Fraction of edges within communities minus expected fraction |
