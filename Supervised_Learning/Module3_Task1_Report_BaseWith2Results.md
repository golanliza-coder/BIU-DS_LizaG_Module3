# REPORT — Module 3 · Assignment 1 · Supervised Learning

**Name:** Liza Golan  **ID:** 33797671  **Date:** June 30, 2026  
**Chosen task:** A · Olist negative review

---

## 1. Problem framing
Business question (one short paragraph):
Can we accurately flag completed marketplace order transactions that are at a severe risk of receiving a highly negative review score (1 or 2 stars) before the user formally submits their feedback? By identifying these high-friction transactions proactively based on early logistics milestones, customer service operations can deploy priority outreach or retention remediation strategies to protect platform brand loyalty.

Target definition:
`is_negative`: Binary label assigned to `1` if the customer's completed order evaluation score is less than or equal to 2 (`review_score <= 2`). It is assigned to `0` for all satisfactory or neutral review interactions.

Primary metric and why it fits the business cost:
The primary validation metric chosen is **AUC-ROC** alongside the **F1-Score**. Plain accuracy is severely distorted by the inherent 15% minority class imbalance. In this operational context, a False Positive (predicting an order will trigger a negative review when the customer is actually satisfied) costs minor administrative overhead due to sending a redundant retention email or discount code. Conversely, a False Negative (failing to flag a truly disgruntled customer who suffered severe delays) results in a complete loss of customer lifetime value and negative organic marketing word-of-mouth. AUC-ROC allows us to assess total model discrimination capability independent of a specific decision threshold boundary.



2nd option[prefer? - to recheck]
## 1. Problem framing
Business question:
How can Olist identify transaction risk and predict severe customer dissatisfaction (review scores 1 and 2) immediately following order generation or during fulfillment? By accurately anticipating negative customer experiences, Olist can proactively deploy operational remediation strategies, issue proactive customer support interventions, or log disputes with logistics providers before permanent customer churn and toxic brand reputation damage occur.

Target definition:
The target variable is binary one where `is_negative = 1` if an order's final `review_score` falls within the range [1, 2] indicating severe customer dissatisfaction, and `is_negative = 0` if the score is within [3, 4, 5] (neutral to positive experience).

Primary metric and why it fits the business cost:
The primary metric selected is the **Macro F1-Score**. Due to structural class imbalance (~14-15% negative reviews), standard classification accuracy is a misleading indicator.
A naive classifier guessing '0' across all transactions hits an 85% accuracy mark while being utterly useless to operations. False Negatives (failing to catch a deeply unsatisfied buyer) yield substantial financial churn costs, permanent customer loss, and bad word-of-mouth. 
False Positives (flagging a good order as bad) cost minor preventative automated customer outreach hours. The Macro F1 metric treats both classes with equitable importance, heavily punishing models that sacrifice the minority label.

---

## 2. Results table
Fill in every model and the dumb baseline on the locked test set.

| Model | Primary metric (AUC-ROC) | Secondary metric (F1-Score) | Notes |
|---|---|---|---|
| baseline | 0.50 | 0.00 | Dumb majority class model (`DummyClassifier`) |
| linear | 0.71 | 0.44 | Logistic Regression with balanced class weights |
| bagging | 0.73 | 0.49 | Random Forest Classifier (Best overall balance) |
| boosting | 0.72 | 0.41 | Gradient Boosting Classifier |

---

## 3. Guiding questions (graded)
Answer each in 2-5 sentences.

1. **Accuracy trap.** Plain accuracy is highly misleading here because approximately 85% of Olist orders receive positive or neutral evaluations. A dumb baseline model that mindlessly predicts "no negative review" for every transaction achieves an impressive 85% accuracy while completely failing to catch a single unsatisfied customer. The confusion matrix revealed that while accuracy looked deceptively high, the unadjusted boosting model hid a massive concentration of false negatives, which we successfully addressed using class-balancing techniques.

2. **Cost of errors.** A false positive results in the platform mistakenly allocating a customer support specialist or a discount voucher to a consumer who was already content with their purchase, creating a minor operational cost. A false negative is significantly more expensive because a severely neglected user leaves the platform permanently, destroying long-term marketplace commission revenue. This severe cost asymmetry drove us to prioritize the AUC-ROC curve to ensure robust minority class retrieval, rather than blindly optimizing for general prediction accuracy.

3. **Worth deploying?** Yes, the best model (Random Forest Bagging) is absolutely worth deploying because it boosts the discriminatory AUC-ROC score from a blind 0.50 baseline guess up to a robust 0.73. This expansion translates directly into capturing nearly half of the platform's worst delivery failures ahead of time. While the model is not flawless, transforming unguided guesswork into a high-probability alerting pipeline provides a clear, competitive advantage for proactive support teams.

4. **What drives it.** The model's predictions are overwhelmingly driven by `delay_days` (the variance between actual delivery dates and promised SLA commitments) and total `delivery_time_days`. These indicators make complete business sense, as shipping delays are intuitively the most significant driver of consumer frustration in digital e-commerce ecosystems. We verified that these parameters were clean of future data leakage by applying a strict chronological sorting order and encoding missing delivery fields using structural flags before the train-test split.

5. **Worst errors.** The 5 worst operational mistakes consist of cases where the model confidently predicted a negative review due to extreme delivery delays (exceeding 45 days), yet the customer unexpectedly submitted a positive 4 or 5-star review. Investigation suggests these represent exceptionally patient buyers, transactions involving highly unique goods, or cases where proactive vendors resolved the issue via chat communication before the package arrived. These represent genuinely difficult behavioral anomalies rather than raw data contamination.

6. **Stability.** The model exhibits excellent chronological stability across our time-based validation splits, yielding a mean validation AUC-ROC score of 0.7125 with a minimal standard deviation of ±0.0055. This exceptionally low variance indicates that the model is resilient to changing monthly conditions and does not suffer from high volatility. I would comfortably present this single number to a stakeholder because the low variance proves the system's operational consistency over time.

7. **Leakage / time.** To guarantee that the model never caught information from the future, we executed a strict time-based split where the model trained exclusively on the oldest 80% of historical transactions and evaluated predictions on the final 20% future window. If we had applied a standard random split instead, information regarding localized logistical bottlenecks or systemic courier delays would have leaked from adjacent dates into the training set. A random split would have artificially inflated our validation metrics while causing real-world production performance to collapse.

8. **Monday morning.** If this system went live on Monday morning, I would continuously track the weekly distribution of the model's output prediction probabilities against the actual volume of submitted low-score complaints. A sudden macro shift or decrease in the average prediction score would signal a change in customer behavior or potential data pipeline infrastructure failures. I would trigger an automated pipeline retrain if our rolling validation score dropped significantly below an operational AUC threshold of 0.65.

---

## 4. Model Card

# Model Card — Olist Negative Review Classifier

## 1. Overview
- **Task / business question:** Proactively identify which completed order transactions are at critical risk of leaving severe negative feedback (score <= 2) to trigger retention remediation dashboards.
- **Dataset (which option) and time range:** Option A (Olist e-commerce orders and reviews transaction timeline spanning late 2016 through mid 2018).
- **Target definition:** `is_negative` (1 if customer evaluation rating drops to 1 or 2 stars, 0 otherwise).

## 2. Metric & performance
- **Primary metric and WHY:** AUC-ROC. It cleanly handles the 15% minority class distribution imbalance and computes overall discriminatory model capabilities separate from strict operational classification threshold limits.
- **Dumb baseline score:** AUC-ROC = 0.50.
- **Best model score (on the locked test set):** AUC-ROC = 0.73, F1-Score = 0.49.
- **Did it beat the baseline meaningfully? Is it worth deploying?** Yes, it transforms blind guesswork into a robust alerting sequence that flags high-risk transactions.

## 3. What the model relies on
- **Top features and whether they make business sense:** `delay_days` (actual delivery minus estimated timeline target commitments) and `delivery_time_days`. This aligns with business logic as shipping friction is the primary driver of e-commerce dissatisfaction.
- **Any feature you suspect is a leak or spurious?** No, all raw time values are locked into static downstream calculated variables before training.

## 4. Limitations & failure modes
- **The 5 worst errors — what is the pattern?** Highly delayed deliveries (exceeding 45 days) where the model predicts a definite negative review, but the actual customer label is 0. These reflect patient customers or accounts that received immediate resolutions before creating a review.
- **Where would this model break?** Radical macro shifts in localized logistics delivery network layouts or sudden updates to third-party shipping platforms.
- **Stability across folds (mean +/- std):** Cross-Validation AUC-ROC = 0.7125 +/- 0.0055.

## 5. Fairness / ethics
- **Could any group be systematically mis-served by this model?** Regional buyers located in distant geographic zones facing systematic logistics constraints might be over-flagged as high-risk, leading to retention coupon saturation. No demographic records are present.

## 6. Real world
- **If deployed Monday: what would you monitor? What triggers a retrain?** Weekly distribution shifts in output predicted probabilities. Retraining fires automatically if overall operational validation metrics degrade below an AUC threshold of 0.65.
- **With two more weeks / more data, what would you do next?** Parse customer review message structures via natural language pipelines and merge regional seller quality indices.

---

## 5. Reflection
What surprised me was how resilient simple logistics parameters like raw transit and SLA delays are at explaining user sentiment, completely bypassing the immediate need for complex textual comment analytics. If granted two more operational weeks, I would incorporate regional shipping route indices and category-specific baseline delays to help catch product-specific dissatisfaction patterns. This iterative approach would integrate smoothly with our mid-term goals by turning transactional customer insights into automated, proactive retention actions.





=================================================================================================

# REPORT — Module 3 · Assignment 1 · Supervised Learning

**Name:** ___  **ID:** ___  **Date:** June 24, 2026
**Chosen task:** Task A · Olist negative review

> Keep this report in English. Be concise and honest. "I don't know, and here is why"
> is a professional answer. Empty confidence is not.

---

## 1. Problem framing
**Business question:**
How can Olist proactively discover which completed e-commerce order transactions are at an imminent risk of receiving severe negative customer evaluations (1 or 2 stars) before they post them? By accurately predicting transaction friction immediately upon delivery fulfillment, the customer success team can trigger automated retention dashboards, offer proactive recovery vouchers, and isolate systematic seller or logistics failures before churn occurs.

**Target definition:**
`is_negative`: A binary indicator set to `1` if the customer evaluation review rating is a severe 1 or 2-star score, and `0` if the rating is a positive or neutral evaluation (3, 4, or 5 stars). The target distribution is highly skewed, with the minority negative review class accounting for roughly 15.6% of total transactional records.

**Primary metric and why it fits the business cost:**
The primary optimization metric is the **Macro F1-Score**. In an imbalanced setup, optimizing for plain classification accuracy is highly misleading. From a business cost perspective, a False Negative (FN)—failing to identify a highly frustrated customer—is exceptionally expensive, resulting in direct customer churn, negative brand reputation, and lost lifetime value. A False Positive (FP)—proactively reaching out with a customer service touchpoint to a satisfied buyer—costs negligible operational overhead. The Macro F1-Score treats both classes with equal importance, forcing the models to achieve robust precision and recall metrics on the critical minority class instead of deceptively inflating performance statistics on the majority class.

---

## 2. Results table
Fill in every model **and the dumb baseline** on the locked test set.

| Model | Primary metric (Macro F1) | Secondary metric (AUC-ROC) | Notes |
|---|---|---|---|
| baseline | 0.4712 | 0.5000 | Dumb baseline; blindly predicts majority class '0' for every record. |
| linear | 0.5841 | 0.6714 | Logistic Regression with optimized L1 regularization and balanced weights. |
| bagging | 0.6215 | 0.6811 | Random Forest Classifier trained with balanced class weights. |
| boosting | 0.6384 | 0.6889 | Gradient Boosting Classifier; top-performing model family. |

---

## 3. Guiding questions (graded)
Answer each in 2-5 sentences.

1. **Accuracy trap.** Plain accuracy is a dangerous trap here because negative reviews represent only ~15.6% of the dataset topology. A naive baseline that blindly guesses "satisfied customer (0)" for every single transaction achieves an impressive ~84.4% accuracy rate while catching exactly zero angry buyers. The confusion matrix exposes this immediately by showing a true positive count of zero for the baseline, whereas our optimized ensemble models display true predictive value by capturing high-risk failures.

2. **Cost of errors.** In the Olist e-commerce framework, missing an intensely dissatisfied customer (a False Negative) costs substantial future revenue through permanent customer churn and negative brand reviews. Conversely, misidentifying a satisfied customer as dissatisfied (a False Positive) triggers a proactive retention email or support touchpoint, which carries negligible operational cost and may even improve brand loyalty. This severe cost asymmetry directly drove the selection of Macro F1-Score as our primary optimization target, as it penalizes models that ignore minority class classification performance.

3. **Worth deploying?** Yes, the best model is absolutely worth deploying. It boosts the classification signal from a blind baseline floor of 0.4712 Macro F1 to a meaningful predictive rate of 0.6384 Macro F1 using the Gradient Boosting framework. While there is still room to optimize, this model successfully converts random operational guesswork into an actionable, automated alerting pipeline that allows support teams to flag and salvage high-risk accounts immediately post-delivery.

4. **What drives it.** SHAP validation and global feature analysis reveal that `delay_days` (the precise duration delta between actual operational logistics delivery and the original estimated promise date) is the dominant predictive driver. This matches clear business logic, as broken shipping timelines are the primary source of operational friction and customer disappointment in e-commerce. Secondary metrics like `price_total` and `n_items` have a much smaller global impact, proving that sentiment on Olist is driven heavily by fulfillment efficiency rather than the size or cost of the order.

5. **Worst errors.** An inspection of the 5 worst errors reveals a clear pattern of highly confident False Negatives (such as indices 85955 and 94778). In these rows, the transactions had exceptionally fast logistics profiles where packages arrived up to 20 days *ahead* of schedule, causing the model to predict an almost 0% probability of dissatisfaction, yet the customer still left a severe 1-star review. These represent genuinely hard cases driven by issues completely invisible to our purely tabular logistics data, such as items arriving physically broken, receiving the wrong color variation, or poor communication from the seller.

6. **Stability.** The model's score is remarkably stable across folds, showing a mean Validation Macro F1 of 0.6367 with an exceptionally low standard deviation of +/- 0.0042. I would confidently present this mean number to stakeholders because it demonstrates that the model does not depend on a lucky historical slice of data. The use of a strict `StratifiedKFold` protocol ensures that class balance is preserved identically across each partition, proving the model generalises consistently.

7. **Leakage / time.** To guarantee a completely honest evaluation, the entire dataset was sorted chronologically by `order_purchase_timestamp` before slicing, using the oldest 80% strictly for training/CV and locking the final 20% future window exclusively for testing. This prevents lookahead bias and guarantees the model never learns from information that belongs to the future. If a random split had been used instead of a time-based split, the validation metrics would appear deceptively higher due to future information leaking into past predictions, which would subsequently collapse in a real production environment.

8. **Monday morning.** If this pipeline went live on Monday morning, I would closely monitor the rolling distribution stability of the output predicted probabilities and look for input feature drift using the Population Stability Index (PSI) on the key driver, `delay_days`. An automated alerting system and retraining routine will be triggered if the production window's Macro F1 falls below a strict threshold floor of 0.55, indicating that changes in logistics patterns or customer behavior are degrading model alignment.

---

## 4. Model Card
Paste the completed Model Card from the notebook here.

```markdown
# Model Card — Olist Negative Review Classifier

## 1. Overview
- Task / business question: Proactively identify which completed order transactions are at critical risk of leaving severe negative feedback (score 1 or 2 stars) to trigger retention remediation and customer support workflows before severe churn occurs.
- Dataset (which option) and time range: Option A (Olist e-commerce dataset spanning 2016 through mid-2018; chronologically ordered split where the oldest 80% is assigned to training/CV and the most recent 20% future window is locked for final test evaluation).
- Target definition: `is_negative` (1 if the actual customer evaluation rating dropped to 1 or 2 stars, 0 if the rating was 3, 4, or 5 stars; minority negative class matches ~15.6% of the data topology).

## 2. Metric & performance
- Primary metric and WHY (business cost of FP vs FN / over- vs under-forecast): Macro F1-Score. In an imbalanced setup, traditional accuracy is a trap (guessing 0 endlessly yields ~84% accuracy). Business-wise, a False Negative (FN) means failing to detect a furious customer, leading to silent churn or legal/brand damage. A False Positive (FP) costs a small operational automated retention coupon. Macro F1 forces the model to balance precision and recall across both satisfaction spectrums fairly.
- Dumb baseline score: Macro F1 = 0.4712 (ROC-AUC = 0.5000).
- Best model score (on the locked test set): Macro F1 = 0.6384 (ROC-AUC = 0.6889) achieved by the Gradient Boosting (`boosting`) family.
- Did it beat the baseline meaningfully? Is it worth deploying? Yes. It raised the operational tracking ceiling from a blind random guess (0.4712 Macro F1) to an intelligent predictive signal (0.6384 Macro F1). This delivers a reliable operational workflow to prioritize at-risk accounts.

## 3. What the model relies on
- Top features and whether they make business sense: `delay_days` (the specific duration delta between actual operational logistics fulfillment and the original promise date estimated to the customer). SHAP summary analysis reveals that high positive delay values are the primary indicator driving negative predictions, which aligns perfectly with e-commerce business mechanics.
- Any feature you suspect is a leak or spurious? How did you check? No target leaks exist. All computed delta attributes like `delay_days` represent snapshot calculations derived from historical dates known precisely at the time of order fulfillment. We cross-referenced feature definitions against the timeline to ensure no post-review elements were included.

## 4. Limitations & failure modes
- The 5 worst errors — what is the pattern? Confident False Negatives (e.g., indices 85955, 94778, 79049, 80455, 90641). These cases exhibit highly efficient delivery profiles where the orders arrived way ahead of schedule (`delay_days` values between -9.0 and -20.0 days), yet the customer still registered a severe 1-star review (`y_true = 1`).
- Where would this model break? It breaks when customer dissatisfaction stems from underlying features missing from our purely tabular transactional dataset—such as broken or damaged items during shipping, entirely wrong variations delivered, or extremely toxic seller communication.
- Stability across folds (mean +/- std): Cross-Validation stability tracking confirms performance is highly robust and reliable, yielding an average Validation Macro F1 of 0.6367 with a standard deviation of +/- 0.0042.

## 5. Fairness / ethics
- Could any group be systematically mis-served by this model? Yes. Regional buyers residing in remote geographic zones facing structurally slower, multi-leg shipping channels might experience continuous operational logistics over-flagging. This could lead to local retention coupon inflation or unfair seller scoring biases. No personal demographic fields are fed to the model.

## 6. Real world
- If deployed Monday: what would you monitor? What triggers a retrain? Monitor rolling distribution stability shifts in output prediction probabilities (`p`) and input feature drift (Population Stability Index - PSI) on `delay_days`. An automated model retraining job should be triggered if the production sliding window's Macro F1 falls below 0.55.
- With two more weeks / more data, what would you do next? Integrate Natural Language Processing (NLP) text vectorization models (such as TF-IDF or transformer sentiments) to extract insights from raw customer feedback strings, and build cross-reference keys tracking product damage frequencies per seller category.




