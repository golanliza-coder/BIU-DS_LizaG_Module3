
# REPORT — Module 3 · Assignment 3 · Deep Learning Foundations

**Name:Liza Golan  **ID:** 033797671  **Date:** 03/08/26
**Chosen option:** __A_ (A · Olist MLP / B · Fashion-MNIST CNN / C · Olist Autoencoder)

Comment: The notebook was executed on google collab only [Trying to run it from git - yet issue not resolve].

---

## 1. Framing
For this assignment, I chose **Option A**: predicting negative review risk for e-commerce orders using the Olist tabular dataset. 

My goal was to build a end-to-end binary classification pipeline using standard-scaled features (8 numerical features in total) and explore whether introducing a neural network provides a genuine edge over linear models on tabular data. Throughout the project, I set out to evaluate not just raw metric performance, but also the real-world trade-offs: training speed, parameter footprint, hyperparameter sensitivity, and model interpretability.

---

## 2. Results

| Model | Test metric | Params | Train time | Notes |
|---|---|---|---|---|
| simpler baseline (Default LR) | 0.7327 ROC-AUC | 9 | ~0.10s | Baseline linear model on standard-scaled features |
| simpler baseline (Improved LR) | 0.7454 ROC-AUC | ~45 | ~1.20s | Enhanced with PowerTransform, Degree-2 interaction features & GridSearch |
| neural net (PyTorch MLP) | 0.7496 ROC-AUC | 321 | ~25.54s | 2-layer MLP: 8 -> 32 (ReLU, Dropout 0.2) -> 1 |

---

## 3. Guiding questions (graded)

1. **Did DL win?** 
   Yes, the PyTorch MLP achieved the highest Test ROC-AUC score of **0.7496**, beating the initial default Logistic Regression baseline of **0.7327** by **+1.69%**. What was most interesting to discover during experimentation was *why* it won: the MLP's hidden layer automatically learned non-linear feature interactions (like how high freight ratios combined with delivery delays impact customer dissatisfaction). However, when I explicitly engineered pairwise feature interactions and non-linear power transforms for the Logistic Regression model, its performance jumped to **0.7454**, narrowing the gap to a minor **0.42%**.

2. **Logits / loss.** 
   I used `nn.BCEWithLogitsLoss()` as the loss function. During my PyTorch setup, I learned that applying a Sigmoid activation directly in the model's `forward()` pass before sending outputs to standard BCE loss can lead to numerical instability. Extremes in predicted values cause floating-point underflow or overflow when taking logarithms near 0 or 1. `BCEWithLogitsLoss` takes raw, unscaled logits and uses the log-sum-exp trick internally, ensuring stable gradient updates.

3. **Overfitting.** 
   Looking at my training and validation loss curves, both metrics decreased smoothly together and plateaued around ~0.321 by epoch 5 without diverging. To keep the model grounded and prevent memorization, I incorporated Dropout ($p=0.2$) between layers, $L_2$ weight decay ($1\text{e-}4$), and saved the best model weights based on peak validation ROC-AUC. As a result, the final generalization gap between train ROC-AUC ($0.7462$) and validation ROC-AUC ($0.7348$) was less than 0.012, confirming no severe overfitting occurred.

4. **Learning rate.** 
   Through hyperparameter testing, I confirmed that the learning rate is the most critical knob in training neural networks because it dictates step size across the optimization landscape. When I tested a large learning rate ($LR = 0.1$), the optimization became unstable, causing loss to bounce around and diverge in later epochs. Conversely, a tiny learning rate ($LR = 1\text{e-}5$) resulted in severe underfitting, leaving the loss stuck around ~0.44 after 20 epochs. Settling on $LR = 1\text{e-}3$ provided the ideal balance of swift and stable convergence.

5. **Regularization.** 
   I implemented two main regularization techniques: Dropout ($p=0.2$) and Weight Decay ($L_2 = 1\text{e-}4$). Weight decay penalizes excessively large weight values to keep the decision boundary smooth, while Dropout randomly zeroes out 20% of node activations during training to prevent neurons from co-depending on one another. Tracking the metrics confirmed their value—keeping validation ROC-AUC stably aligned with training performance.

6. **Cost / benefit.** 
   The MLP requires 321 trainable parameters and ~25.5 seconds of training time compared to standard Logistic Regression’s 9 parameters and sub-second execution (~0.10s). Furthermore, Logistic Regression gives us direct, interpretable coefficients, whereas the MLP operates as a black box. In a high-stakes customer retention scenario where detecting dissatisfied buyers matters most, a +1.69% ROC-AUC gain justifies the modest extra compute cost, though the engineered LR model remains a strong lightweight candidate.

7. **When DL.** 
   Given the choice for this specific task, I would choose to deploy the PyTorch MLP. In modern production environments, an extra ~25 seconds of training time and 321 parameters present virtually zero compute overhead, and the extra predictive edge translates directly to better detection of at-risk e-commerce orders.

8. **Monday morning.** 
   If I were taking this model to production on Monday morning, I would monitor three key areas: input data drift (e.g., shifts in average delivery times or shipping pricing), output prediction distribution shifts, and real-world ROC-AUC performance on new labeled customer reviews. I would establish automated retraining triggers set on a monthly cycle or whenever rolling test ROC-AUC falls below $0.71$.

---

## 4. DL Model Card

# DL Model Card

## 1. Overview
- Option / task / data: Option A - Olist tabular MLP (predicting negative review risk). Dataset consists of tabular e-commerce order attributes with 8 preprocessed and standard-scaled input features.
- Architecture (layers, params): Multi-Layer Perceptron (MLP) architecture: Linear(8, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1). Total trainable parameters: 321.

## 2. Setup
- Loss and why (logits handling): BCEWithLogitsLoss. Receives raw, unscaled output logits directly from the model without an explicit Sigmoid layer in forward pass. This guarantees numerical stability by combining Sigmoid and Binary Cross-Entropy via the log-sum-exp trick.
- Optimizer, learning rate, regularizer: Adam optimizer with learning rate = 1e-3. Regularization mechanisms include Dropout (p=0.2), Weight Decay (L2 = 1e-4), and Early Stopping / Best-Model Checkpointing based on peak validation ROC-AUC.

## 3. Performance
- Simpler-model baseline: Default Logistic Regression Test ROC-AUC = 0.7327 (9 parameters).
- Neural-net test metric: PyTorch MLP Test ROC-AUC = 0.7496 (321 parameters).
- Did DL win? By how much, at what cost? Yes, the Deep Learning model won by +0.0169 (+1.69% in ROC-AUC score). The costs involved include higher parameter complexity (321 vs 9), increased training duration (~25.5s vs ~0.1s), and reduced model interpretability (matrix weights vs direct feature coefficients).

## 4. Diagnostics
- Learning curves: where do train and val diverge? Training and validation curves track closely together throughout 20 epochs without divergence. Both loss curves plateau smoothly around ~0.321 and ROC-AUC around ~0.735, confirming zero overfitting.
- Learning-rate sensitivity: LR = 0.1 was unstable and oscillated/diverged toward later epochs. LR = 1e-5 learned far too slowly, remaining stuck at a high loss of ~0.44 after 20 epochs. LR = 1e-3 proved optimal, yielding rapid and stable convergence.

## 5. Decision
- Would you deploy DL or the simpler model here? Deploy the Deep Learning model (MLP). The +1.69% increase in ROC-AUC provides superior detection of dissatisfied customer orders, which outweighs the minor computational cost (~25.5 seconds training time) and low parameter count (321 parameters).
- Production: what to monitor, what triggers a retrain. Monitor input feature drift (e.g., freight ratio shifts, delivery delay increases), prediction probability distribution shifts, and real-world ROC-AUC degradation. Trigger automated retraining monthly or if rolling validation ROC-AUC falls below 0.71.

---

## 5. Reflection

What stood out to me most in this assignment was seeing how much performance could be squeezed out of a simple linear algorithm once I applied intentional feature engineering. By introducing non-linear power transformations and explicit 2nd-degree polynomial interactions, Logistic Regression improved from **0.7327** to **0.7454**, nearly bridging the gap with the neural network.

As a Data Science student, this experiment reinforced a core principle: Deep Learning isn't a silver bullet for every problem, but it shines when dealing with raw unstructured inputs (like images or text) or complex tabular problems where manual feature engineering becomes too complex. Working through this pipeline gave me a much clearer appreciation for balancing model complexity with practical data engineering.