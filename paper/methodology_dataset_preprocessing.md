# Methodology: Dataset and Knowledge Graph Construction

---

## A. Document Corpus

The knowledge base is constructed from the publicly archived meeting documents of **3GPP TSG-RAN Working Group 1 (RAN WG1)**, the body responsible for defining the physical layer of 5G New Radio (NR) and its successive enhancements. RAN WG1 is the highest-volume working group in 3GPP and the primary site of technical competition among equipment vendors, operators, and chipset manufacturers proposing physical-layer innovations [CITE: Baron 2025 SSRN]. Each meeting generates between 1,500 and 2,000 Technical Documents (TDocs), covering five principal document genres: (*i*) **Contribution documents**, carrying company proposals for specification text; (*ii*) **Change Requests (CRs)**, encoding line-level modifications to an existing Technical Specification (TS) or Technical Report (TR); (*iii*) **Session notes**, recording agreed outcomes and remaining work items; (*iv*) **Liaison statements (LS/LS-Reply)**, formalising communication between working groups or external bodies; and (*v*) normative **Technical Specifications and Reports** themselves, which carry the final standardised text.

The corpus spans **six consecutive meetings**: RAN1 \#116, \#116-bis, \#117, \#118, \#118-bis, and \#119, covering the period from early 2024 through early 2025 — the active development window of 3GPP Release 19 and the later phases of Release 18. Source archives were obtained from the publicly accessible 3GPP FTP server (`https://www.3gpp.org/ftp/`) where each meeting's contributions are stored as per-document ZIP archives. Every document is identified by its canonical 3GPP identifier, following the pattern `R1-YYNNNNN` (RAN WG1, year, sequential number).

After extraction and processing, the corpus comprises **10,356 documents** distributed as follows:

| Meeting | Documents |
|---|---|
| RAN1 \#116 | 1,843 |
| RAN1 \#116-bis | 1,782 |
| RAN1 \#117 | 1,848 |
| RAN1 \#118 | 1,698 |
| RAN1 \#118-bis | 1,659 |
| RAN1 \#119 | 1,525 |
| **Total** | **10,356** |

The dominant document type is *Contribution* (approximately 73% of the corpus), followed by *Change Request* (~12%), *Meeting Document* (~8%), and *Liaison Statement* (~3%). By release, the corpus is concentrated in Release 19 (~47% of documents with a recognised release string) and Release 18 (~35%), reflecting the active specification work during the covered period.

---

## B. Structured Information Extraction

### B.1 Ingestion Pipeline

Raw documents arrive as `.doc` and `.docx` files nested inside ZIP archives. Each archive is extracted to a temporary directory, and constituent Word documents are loaded using `UnstructuredWordDocumentLoader`. Document text is tokenised using the `cl100k_base` encoding (tiktoken); documents exceeding **65,000 tokens** are excluded from processing due to context-window constraints, discarding a small fraction of very large session-note aggregates.

Processing is parallelised across a `ThreadPoolExecutor` of **100 workers**, with one worker assigned per ZIP archive. Within each archive, documents are processed sequentially to avoid token-budget contention on the LLM API.

### B.2 LLM-Based Entity and Relation Extraction

Structured information is extracted from document text using a **two-model, three-tier pipeline**. The primary extractor is **DeepSeek-R1** (`deepseek-reasoner`), invoked via structured-output completion with the `DataModel` Pydantic schema as the target type. DeepSeek-R1's chain-of-thought reasoning capability is particularly suited to dense technical text, where acronym disambiguation, cross-reference resolution, and entity boundary detection require multi-step inference rather than simple span extraction [CITE: DeepSeek-R1 2025]. When the structured output call fails, a second attempt is made using the same model in free-form completion mode, with the full schema definition embedded in the prompt; the response is then parsed by regex extraction targeting JSON blocks. If parsing still fails, **DeepSeek-Chat** (`deepseek-chat`) acts as a JSON repair model, correcting formatting errors in the raw output before a final parse attempt. This three-tier fallback ensures high extraction coverage across the heterogeneous formatting found in 3GPP TDoc contributions.

### B.3 Extraction Schema

The extraction target is defined as a Pydantic v1 `DataModel` comprising six node types and five edge types:

**Node types:**

- **Document** — the central entity, carrying `doc_id`, `title`, `release`, `type`, `tags`, `summary`, `topic`, `keywords`, `meeting_id`, `status`, and `source_path` (the reconstructed 3GPP FTP URL)
- **Contributor** — a submitting organisation or individual, with `name` and `aliases`
- **TechnologyEntity** — a named concept, signal, procedure, or standard document referenced in the text, with `canonical_name`, `aliases`, and `description`
- **WorkingGroup** — a 3GPP working group, with `id`, `name`, and `description`
- **Meeting** — a specific 3GPP plenary or working group meeting instance, with `meeting_id`, `venue`, `wg`, and `topic`
- **Agenda** — a numbered agenda item at a specific meeting, with `agenda_id`, `meeting_id`, `topics`, and `descriptions`

**Edge types:**

| Relationship | Direction | Key Properties |
|---|---|---|
| `AUTHORED` | `(Contributor) → (Document)` | `contribution_type` |
| `MENTIONS` | `(Document) → (TechnologyEntity)` | `context`, `frequency` |
| `BELONGS_TO` | `(Document) → (WorkingGroup)` | `role_in_group` |
| `REFERENCES` | `(Document) → (Document)` | `type_of_reference`, `details` |
| `APPEARS_IN` | `(Document) → (Agenda)` | `release`, `page_range` |

The `source_path` property on Document nodes is constructed at ingestion time by mapping each document's local ZIP path back to its canonical 3GPP FTP URL via directory suffix extraction, enabling downstream retrieval of original source files.

---

## C. Knowledge Graph Construction

### C.1 CSV Generation and Deduplication

Following LLM extraction, the per-document JSON outputs stored in the `Results/` directory hierarchy are aggregated into eleven normalised CSV files by `generate_csv.py`. Deduplication during this stage is non-trivial given that the same technology entity, contributor, or agenda item may be extracted independently from hundreds of documents with slight textual variation.

Deduplication keys are as follows:

- **Document**: exact `doc_id` (unique per document in the 3GPP naming convention)
- **Contributor**: `(name, aliases)` tuple
- **TechnologyEntity**: `canonical_name` (deduplicated across all documents; aliases pipe-delimited)
- **WorkingGroup**: `(id, name)` pair; the LLM-generated `description` is excluded from the key since it varies across documents despite referring to the same WG
- **Agenda**: composite key `(agenda_id, meeting_id)` — this is critical because agenda numbering resets at each meeting; agenda item "9" at RAN1 \#116 and agenda item "9" at RAN1 \#118 are distinct agenda items covering different work items, and conflating them corrupts both the graph structure and the full-text index

For Agenda nodes, the `topics` and `descriptions` fields are not taken from any single document's extraction output; instead, they are aggregated as semicolon-delimited sets of unique strings collected from all documents that reference a given `(agenda_id, meeting_id)` pair. This aggregation gives each Agenda node a rich, multi-perspective textual representation of its scope, improving the recall of agenda-based retrieval.

The `REFERENCES` relationship presents a particular attribution challenge: the LLM extraction schema places outgoing references in a top-level list without a `source_doc_id` field. Attribution is resolved by assigning all references in a given extraction output to the first document in that file, under the practical assumption that most ZIP archives contain a single TDoc.

### C.2 Neo4j Import and Indexing

The eleven CSV files are imported into a **Neo4j 5.x** graph database via `LOAD CSV` with node-level `MERGE` operations keyed on the deduplication fields above. Uniqueness constraints are created before import on all primary keys. Agenda nodes use a **composite node key constraint** on `(agenda_id, meeting_id)`, explicitly preventing cross-meeting merging of structurally identical but semantically distinct agenda items.

Three **full-text (BM25/Lucene) indexes** are built over the graph to support retrieval:

| Index | Node Label | Indexed Properties |
|---|---|---|
| `docIndex` | `Document` | `title`, `summary`, `keywords`, `topic`, `tags` |
| `agendaIndex` | `Agenda` | `topics`, `descriptions`, `release` |
| `techEntityIndex` | `TechnologyEntity` | `canonical_name`, `aliases`, `description` |

BM25 is chosen as the primary retrieval mechanism — rather than dense vector similarity — for a corpus dominated by technical identifiers (`TS 38.214`, `PDCCH`, `TCI framework`), which are poorly handled by general-purpose embeddings that have not been fine-tuned on 3GPP vocabulary [CITE: Telco-DPR 2024]. The `docIndex` operates directly over the rich document fields extracted by the LLM, including the generated `summary`, `keywords`, and `topic`, which substantially extend the indexable signal beyond the document's literal title.

### C.3 Post-Import Graph Cleaning

The LLM extraction process introduces several systematic artefacts that require post-import correction before the graph can be used for retrieval:

1. **Duplicate relationships**: The same `(Document, TechnologyEntity)`, `(Contributor, Document)`, or `(Document, Document)` pair may appear multiple times due to the same document being referenced across multiple JSON extraction outputs. All duplicate edges are collapsed to a single relationship, retaining the highest observed `frequency` value for `MENTIONS` edges.
2. **Self-referencing citations**: 40 `REFERENCES` edges where a document cites itself — attributable to the reference attribution heuristic — are deleted.
3. **Orphan TechnologyEntity nodes**: 5,592 of 7,043 initially extracted `TechnologyEntity` nodes (79.4%) had no `MENTIONS` edge, caused by a systematic mismatch between the LLM's free-form `entity_name` in the `MENTIONS` model and the formal `canonical_name` in the `TechnologyEntity` model (e.g., abbreviation `"SRS"` versus canonical `"Sounding Reference Signal (SRS)"`). These unlinked nodes pollute the `techEntityIndex` — they can match full-text queries but cannot route back to any document — and are removed.
4. **Duplicate Agenda nodes**: Despite the composite key constraint on `(agenda_id, meeting_id)` in the corrected pipeline, the original import created up to 55 duplicate Agenda nodes per `agenda_id` due to (a) meeting-scoped keying not being enforced and (b) the `topics` field not being aggregated. These are merged using `apoc.refactor.mergeNodes()` with `mergeRels: true`, collapsing 1,469 nodes to 420 while preserving all `APPEARS_IN` edges.

After cleaning, the graph contains the following:

| Entity | Count |
|---|---|
| Document nodes | 10,356 |
| TechnologyEntity nodes | 1,451 |
| Agenda nodes | 420 |
| Contributor nodes | 359 |
| `MENTIONS` relationships | 13,050 |
| `AUTHORED` relationships | 12,388 |
| `APPEARS_IN` relationships | 9,843 |
| `REFERENCES` relationships | 5,970 |

---

## D. Graph Coverage and Statistics

**Node connectivity**: 99.4% of documents have at least one `AUTHORED` edge; 90.7% are linked to at least one `Agenda` node via `APPEARS_IN`; 83.6% carry at least one `MENTIONS` edge to a `TechnologyEntity`. After orphan removal, all 1,451 `TechnologyEntity` nodes are connected to the document subgraph, with an average of 8.99 `MENTIONS` edges per entity.

**Contributor network**: 359 unique contributor organisations are represented, with an average of 34.5 documents per contributor. The ten most prolific contributors are Huawei (767 documents), Ericsson (721), Samsung (641), vivo (485), Nokia (482), CATT (474), OPPO (431), ZTE (426), HiSilicon (355), and NTT DOCOMO (344), reflecting the known distribution of intellectual property activity in 3GPP RAN WG1 [CITE: Baron 2025].

**Technology entity network**: The most frequently mentioned technology entity is **SSB** (Synchronisation Signal Block), appearing in 838 unique documents, followed by **PRACH** (Physical Random Access Channel, 578 documents), **CSI-RS** (Channel State Information Reference Signal, 404 documents), **LP-WUS** (Low Power Wake-Up Signal, 381 documents), and **SRS** (Sounding Reference Signal, 289 documents). These frequencies reflect the active work items on MIMO enhancement, Ambient IoT, beam management, and AI/ML for air interface that dominated the RAN1 \#116–\#119 period.

**Citation network**: 4,442 documents carry outgoing `REFERENCES` edges (citing 5,970 edges total), while 1,775 documents are cited at least once, reflecting the typical ratio in standards contribution literature where normative specifications, study item reports, and feature lead summary documents are heavily referenced while individual company contributions are rarely cited back.

**Document connectivity**: On average, each document is linked to 0.95 agenda items, 1.26 technology entities, 1.20 contributors, and 0.58 cited documents. The maximum agenda linkage for a single document is 63, corresponding to broad session-level summary documents that span the entire agenda. The maximum entity mentions per document is 43.

---

## E. Three-Branch Graph Retrieval

At query time, document retrieval is implemented as a single Cypher query combining three complementary full-text search branches, executed within a single Neo4j transaction:

**Branch A — Direct document retrieval** operates on `docIndex`, scoring documents by BM25 against the LLM-extracted `title`, `summary`, `keywords`, `topic`, and `tags` fields. This branch identifies documents whose *content* is directly relevant to the query.

**Branch B — Agenda-mediated retrieval** queries `agendaIndex` on aggregated `topics`, `descriptions`, and `release`, then traverses the graph pattern `(Document)−[:APPEARS_IN]→(Agenda)` to surface documents associated with the matched agenda items. This branch recovers documents whose topic is the subject of the agenda item even if the query terms do not appear prominently in the document's own text fields. The agenda-derived score is weighted by a cross-index bonus: documents already retrieved by Branch A receive a multiplier of **2.3×**, exploiting the meeting-agenda topology as a structural relevance signal for documents that appear in both direct and contextual retrieval paths; documents retrieved only via agenda context receive a reduced multiplier of **0.8×** to prevent agenda-noise from overwhelming direct matches.

**Branch C — Entity-mediated retrieval** queries `techEntityIndex` and traverses `(Document)−[:MENTIONS]→(TechnologyEntity)` to recover documents that prominently reference the queried technology concept. Entity-derived scores are attenuated by a weight of **0.7×**, reflecting that entity co-occurrence is a weaker relevance signal than direct document content match.

Scores across the three branches are aggregated per document by summation:

$$\text{score}(d) = s_A(d) + w_B(d) \cdot s_B(d) + 0.7 \cdot s_C(d)$$

where $s_A$, $s_B$, $s_C$ are the BM25 scores from each branch, $w_B(d) = 2.3$ if $d$ appears in Branch A and $0.8$ otherwise. A final domain-specific boost is applied: documents whose title contains "Feature Lead Summary" (the 3GPP convention for the designated rapporteur's synthesis document at each agenda item) receive an additional **2.0× multiplier** on their total score; documents titled as "Feature Lead" documents receive **1.5×**. Feature Lead Summary documents represent expert synthesis of all proposals on a work item and are systematically the highest-value retrieval target for technical queries. The top-15 documents by boosted score are returned to the generation stage.

This multi-branch design is analogous in spirit to the hybrid BM25 + graph traversal approaches in Neo4j's `HybridRetriever` [CITE: Neo4j GraphRAG 2024] and to the two-level retrieval in LightRAG [CITE: LightRAG 2024], but is tailored to the specific topology of the 3GPP contribution graph: the `APPEARS_IN` and `MENTIONS` edges encode domain-specific relevance signals (meeting agenda co-occurrence and technology concept co-mention) that general-purpose GraphRAG frameworks do not exploit.

---

## F. Retrieval-Augmented Generation

Documents retrieved by the graph query are passed to a Pathway-based DocumentStore [CITE: Pathway] indexed with **voyage-3-large** embeddings (1024-dimensional; 32K context window) for dense retrieval within the downloaded document collection. Before passing context to the generator, retrieved passages are optionally reranked using **VoyageAI rerank-2** [CITE: VoyageAI rerank-2 2024], a cross-encoder model that improves retrieval accuracy by 13.89% over the underlying embedding model on held-out technical documentation benchmarks. The final ranked context (up to 10 passages, maximum 5,000 output tokens) is passed to **DeepSeek-Chat** for answer generation with temperature 0.3 and top-p 0.9, with a system prompt instructing the model to perform comparative and critical analysis, cite document names and page numbers, and flag conflicting information across sources.

---

## References

[CITE: Lewis 2020] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. arXiv:2005.11401.

[CITE: Chat3GPP 2025] "Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents," arXiv:2501.13954, 2025.

[CITE: TelecomRAG 2024] "TelecomRAG: Taming Telecom Standards with Retrieval Augmented Generation and LLMs," *ACM SIGCOMM CCR*, 2024.

[CITE: DeepSpecs 2024] "DeepSpecs: Expert-Level Questions Answering in 5G," arXiv:2511.01305, 2024.

[CITE: TelcoAI 2026] "TelcoAI: Advancing 3GPP Technical Specification Search through Agentic Multi-Modal Retrieval-Augmented Generation," arXiv:2601.16984, 2026.

[CITE: Telco-DPR 2024] M. Saraiva et al., "Telco-DPR: A Hybrid Dataset for Evaluating Retrieval Models of 3GPP Technical Specifications," arXiv:2410.19790, 2024.

[CITE: GraphRAG 2024] D. Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," arXiv:2404.16130, Microsoft Research, 2024.

[CITE: GraphRAG Survey 2024] B. Peng et al., "Graph Retrieval-Augmented Generation: A Survey," arXiv:2408.08921, 2024.

[CITE: LightRAG 2024] Z. Guo et al., "LightRAG: Simple and Fast Retrieval-Augmented Generation," arXiv:2410.05779, 2024.

[CITE: HippoRAG 2024] B. Jiménez Gutiérrez et al., "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models," *NeurIPS*, arXiv:2405.14831, 2024.

[CITE: KG Survey 2025] H. Bian, "LLM-empowered knowledge graph construction: A survey," arXiv:2510.20345, 2025.

[CITE: DeepSeek-R1 2025] D. Guo et al. (DeepSeek-AI), "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," arXiv:2501.12948, 2025.

[CITE: DeepSeek-KG 2025] "Knowledge Graph-Driven Retrieval-Augmented Generation: Integrating DeepSeek-R1 with Weaviate," arXiv:2502.11108, 2025.

[CITE: I40KG 2021] "Analyzing a Knowledge Graph of Industry 4.0 Standards," arXiv:2107.01910, 2021.

[CITE: REBEL 2021] P. Cabot and R. Navigli, "REBEL: Relation Extraction By End-to-end Language generation," *EMNLP Findings*, 2021.

[CITE: VoyageAI voyage-3-large 2025] VoyageAI, "voyage-3-large: State-of-the-Art General Embedding Model," blog.voyageai.com, January 7, 2025.

[CITE: VoyageAI rerank-2 2024] VoyageAI, "rerank-2: Superior Reranking Model," blog.voyageai.com, September 30, 2024.

[CITE: Baron 2025] J. Baron, M. Bergallo, and M. Gamarra, "A Text-Based Analysis of Technical Contributions to 3GPP," SSRN:5369199, 2025.

[CITE: Neo4j GraphRAG 2024] Neo4j, "Hybrid Retrieval for GraphRAG Applications Using the GraphRAG Python Package," neo4j.com/blog, 2024.

[CITE: Pathway] Pathway, *Pathway: Python Data Processing Framework for Real-Time Applications*, pathway.com.
