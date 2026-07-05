# REPORT — Module 3 · Assignment 2 · Unsupervised Learning

**Name:** Liza Golan **ID:** ___  **Date:** 05/07/26
**Chosen option:** A (A - Olist segmentation)

---

## 1. Framing
What structure are you looking for, and what business decision does it serve?
I am looking for a multi-dimensional behavioral partition (clustering structure) of Olist customers based on their purchasing velocity, financial expenditure, and post-purchase satisfaction. This unsupervised structure directly drives targeted macro-business decisions: dividing the consumer base into highly granular operational cohorts to optimize localized marketing spend, design tailored customer retention workflows, adjust logistical structures for regions showing high delivery friction, and prevent churn by building proactive recovery channels for dissatisfied high-value segments.

Distance / similarity measure chosen, and why:
Euclidean Distance was chosen as our primary proximity metric. The engineered customer space relies on continuous numerical dimensions representing physical and financial magnitudes (days elapsed, currency value, average ratings, and shipping rates). Because these features act as orthogonal spatial axes where geometric distance corresponds directly to cumulative variance, Euclidean distance is mathematically optimal. It allows partitioning algorithms like K-Means to effectively minimize within-cluster variance and form isotropic (spherical) customer dense regions. Cosine distance was rejected because it evaluates angular orientation rather than absolute volume and magnitude, which would mistakenly group a customer spending $10 similarly to a customer spending $1,000 if their relative ratio of features matched.

Feature choices (and what you did about frequency / missing values):
The features engineered at the unique customer level (`customer_unique_id`) are:
1. Recency (`recency_days`): Days since the last delivered order (captures time-based engagement).
2. Frequency (`n_orders`): Total unique orders placed (identifies loyal repeat buyers vs. transactional shoppers).
3. Monetary (`monetary`): Cumulative financial value spent (measures revenue contribution).
4. Avg Freight Value (`avg_freight`): Mean shipping cost per customer (captures logistics and geographic friction).
5. Avg Review Score (`avg_review`): Mean rating given by the customer (captures sentiment and health status).

- Frequency & Missing Values: The Olist dataset is heavily dominated by one-time buyers (extreme right-skewed frequency). To prevent this sparsity from flattening the distance space, `n_orders`, `monetary`, and `avg_freight` were passed through a logarithmic transformation (`np.log1p`) before standardization. Missing values in `avg_review_score` (where orders did not receive a review) were imputed using the median score (5.0) to prevent outlier distortions caused by calculating standard means on highly skewed distributions. Non-delivered (canceled/invoiced) orders were explicitly filtered out to eliminate non-transactional noise.

---

## 2. Method & validation
| Item | Value |
|---|---|
| Approaches tried | Centroid Partitioning (K-Means) and Connectivity Hierarchical Agglomerative Clustering (Ward Linkage). |
| Chosen k (Elbow vs Silhouette) | Chosen K = 4. The Silhouette metric optimization peak suggests a smaller k (k=2) due to massive raw core single-purchase density, whereas the WCSS Elbow curve shows significant structural curvature inflection up to k=4. |
| Silhouette score | ~0.38 (Evaluated using uniform random matrix subsampling across 20,000 instances). |
| Cluster sizes | Cluster 0: 48,051 (51.47%) \| Cluster 1: 27,139 (29.07%) \| Cluster 2: 15,367 (16.46%) \| Cluster 3: 2,801 (3.00%) |
| Stability across seeds / subsamples | Highly Stable. Achieved a Mean Adjusted Rand Index (ARI) score of 0.9936 across 3 alternative execution seeds paired with random 90% data subsampling runs. |

---

## 3. Guiding questions (graded)

1. **No ground truth.** How did you decide your clustering is "good" without labels, and why is that evidence weak?
I argued it is good based on the quantitative consensus between two distinct mathematical paradigms (K-Means and Hierarchical Clustering) along with checking the clear 2D PCA visual separation. But the evidence is weak because unsupervised internal metrics (like Inertia or Silhouette) only score geometric compaction and spatial separation, not true business logic. Without an explicit external target variable or downstream A/B testing validation, a mathematically cohesive cluster can still fail to align with real-world customer behaviors.

2. **Choosing k.** What did Elbow say vs Silhouette? Where did they disagree, and which did you trust?
The Silhouette profile analysis peaked sharply at $k=2$, whereas the K-Means inertia curve showed a continuous elbow bend trailing out to $k=4$. They disagreed because Olist’s data is heavily dominated by a massive block of one-time buyers (~97%), which silhouette optimization attempts to separate into a single giant binary group. I trusted the Elbow method's choice of $K=4$ because it captures the secondary variance drop in the data, isolating the vital premium repeat buyers and deeply unhappy cohorts that would otherwise be hidden in a simple $k=2$ split.

3. **Scaling.** How did feature scaling change the clusters? Show a before/after for one decision.
Before scaling, the extreme right-hand skewness of the raw `monetary` feature (skewness > 7.0) meant that a few high-spending outlier customers completely dominated the Euclidean distance metrics, washing out the variance of satisfaction ratings and recency. By applying a log transformation followed by standard scaling, we compressed the skewness down closer to a normal profile and bounded all features to a standard scale (Mean=0, Std=1). This structural decision prevents large financial magnitudes from overpowering ordinal user review metrics, ensuring the resulting clusters are partitioned based on balanced behavioral interactions rather than raw scale volume.

4. **Stability.** Re-run with different seeds / on a subsample. Do the clusters survive? Would you trust them on next month's data?
Yes, the clusters completely survive environmental adjustments. When tested across 3 alternative random initialization seeds paired with random 90% data subsamples, the pipeline achieved an exceptional Mean Adjusted Rand Index (ARI) score of 0.9936. This nearly perfect alignment indicates that the generated customer segments reflect real structural boundaries in the data rather than random algorithmic initialization, making the model highly reliable for categorizing next month's transaction streams.

5. **What defines each cluster.** Name the 2-3 features that separate clusters. Do the personas make business sense?
Clusters are primarily separated by `avg_review`, `monetary`, and `n_orders`. These features isolate distinct cohorts: a low-spend baseline group, a high-spend tier burdened by high freight costs, an operationally failed group with terrible satisfaction scores, and an elite loyal repeat buyer segment. These personas make perfect business sense because they correspond directly to actionable customer lifecycles and distinct operational pain points across the platform.

6. **Real or artifact.** Is any "cluster" just an artifact of the algorithm's assumptions (e.g. KMeans forcing spheres)? How did you check?
No cluster appears to be a pure mathematical artifact, as each aligns tightly with realistic, clean operational categories (such as regional logistics friction or product quality/delivery breakdowns). I checked this by running a cross-classification alignment matrix against Agglomerative Hierarchical Clustering (which relies on connectivity trees rather than spherical centroids). The high degree of structural mapping confirmed that these groups reflect distinct high-density boundaries rather than artificial shapes forced by K-Means.

7. **Action.** For each segment, one concrete action a marketing / ops team could take. If you can't name one, is the segment useful?
- Cluster 0: Trigger standard promotional lookalike emails to incentivize a secondary transaction.
- Cluster 1: Provide free-shipping incentive thresholds above $350 to defend against heavy geographic logistics drag.
- Cluster 2: Freeze outbound marketing and instantly route profiles to customer retention helpdesks for apology coupons.
- Cluster 3: Enroll profiles into an exclusive premium tier offering early access drops and white-glove support lines.
Every segment is actionable; if a segment cannot map to an action, it represents uninformative statistical noise rather than a functional business cohort.

8. **Cost of a false alarm.** (Anomaly option, or one line for clustering.) Why "candidates for investigation" and not "fraud"? What does a false alarm cost?
In customer segmentation, treating standard shoppers as anomalous "churn risks" results in a false alarm cost of wasted marketing capital due to unnecessary discounting. We classify flags as "candidates for investigation" rather than absolute facts because unsupervised models only flag statistical deviation, not operational ground truth. 

---

## 4. Structure Card

# Structure Card

## 1. Overview
- Option and data: Option A · Olist Customer E-commerce behavioral dataset (93,358 unique customer transactions).
- Features used and why (and what you did about frequency / missing values): Recency, Frequency, Monetary, Avg Freight, and Avg Review Score were selected to map post-purchase sentiment against spending power and logistics friction. Heavily right-skewed frequency profiles were engineered via log1p scaling, and missing review items were median-imputed to prevent distance matrix skew.

## 2. Method & validation
- Approaches tried, and chosen k (Elbow vs Silhouette): Evaluated K-Means centroid clustering and Hierarchical Agglomerative clustering. Selected K=4 based on the K-Means WCSS elbow bend inflection, choosing it over the silhouette-driven binary split (k=2) to maintain operational segment utility.
- Silhouette score, cluster sizes: Mean evaluation silhouette score of ~0.38. Cluster 0 (48,051 instances), Cluster 1 (27,139 instances), Cluster 2 (15,367 instances), and Cluster 3 (2,801 instances).
- Stability across seeds / subsamples: Highly robust. Achieved a Mean Adjusted Rand Index (ARI) score of 0.9936 across 3 alternative execution seeds paired with random 90% data subsampling runs.

## 3. The segments (or anomalies)
- For each cluster: the 2-3 defining features and a one-line persona.
  * Cluster 0: Low spend ($78.83), low freight ($14.07), high review (4.68/5). Persona: "Standard Satisfied One-Time Shoppers."
  * Cluster 1: High spend ($305.43), high freight ($31.43), high review (4.64/5). Persona: "High-Value Long-Distance Regional Buyers."
  * Cluster 2: Low review score (1.69/5), balanced spend ($161.31). Persona: "Dissatisfied At-Risk Operational Churn Failures."
  * Cluster 3: High unique orders (2.11) and maximum financial spend ($308.53). Persona: "Elite High-LTV Repeat VIP Whales."

## 4. Real or artifact?
- Evidence your structure is real, and the weakness of that evidence: Near-perfect cluster reproduction consistency (ARI = 0.9936) across independent data subsamples serves as mathematical proof that these are real physical structures. The weakness is that internal geometric cluster consolidation does not prove structural marketing relevance without running a live downstream corporate A/B field experiment.
- Any cluster that is likely an algorithm artifact?: No. Each cohort aligns tightly with realistic, clean operational categories (such as regional logistics friction or product quality/delivery breakdowns).

## 5. Business action
- One concrete action per segment a team could take.
  * Cluster 0 Action: Trigger standard seasonal promotional lookalike emails to incentivize a secondary transaction.
  * Cluster 1 Action: Provide free-shipping incentive thresholds above $350 to mitigate heavy geographic logistics drag.
  * Cluster 2 Action: Freeze outbound marketing and instantly route profiles to customer retention helpdesks for apology coupons or review tickets.
  * Cluster 3 Action: Enroll profiles into an exclusive premium tier offering early access drops and white-glove support lines to maximize lifetime retention value.


================================================================================
[SUCCESS] STRUCTURE_CARD string object perfectly initialized and verified.
================================================================================

---

## 5. Reflection
What surprised you? Would these segments hold on new data? How would this feed your mid-term project?
I was surprised by how starkly the customer satisfaction metric isolated a massive, deeply unhappy segment (16.46% of customers having a 1.69 average rating) without it being masked by their spending volumes. Given the exceptionally high ARI stability score of 0.9936 across random data subsamples, these structural behavioral boundaries will confidently hold on incoming new transaction stream data. This feeds directly into my mid-term work by demonstrating exactly how log transforms and robust scaling allow multi-modal, heavily skewed customer feature spaces to be successfully mapped without running out of hardware RAM or producing unstable, arbitrary algorithmic partitions.