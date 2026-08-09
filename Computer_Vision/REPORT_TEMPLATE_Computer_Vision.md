# REPORT — Module 3 · Assignment 4 · Computer Vision

**Name:** _Liza Golan__  **ID:** _033797671__  **Date:** _07/07/26__
**Chosen option:** __A_ (A · PlantVillage / B · Oxford Flowers-102 / C · PlantVillage -> PlantDoc)

Comment: The notebook was executed on google collab only [Trying to run it from git - yet issue not resolve].

---

## 1. Framing

**Business Question & Metric:**  
The primary objective of this project is to build an automated diagnostic pipeline capable of identifying 38 distinct crop-disease combinations from leaf imagery. In agricultural deployment, early detection is critical to preventing widespread crop loss. From a data science perspective, relying solely on **Overall Accuracy** is dangerous because it can be heavily inflated by well-represented classes while masking catastrophic failures on rare or severe pathogens. Therefore, I prioritized **Per-Class Recall** and **Macro F1-Score** to ensure balanced diagnostic reliability across all disease categories.

**Why Transfer Learning Over Training from Scratch:**  
Training a deep Convolutional Neural Network (CNN) from scratch on tens of thousands of leaf images requires massive compute resources, long convergence times, and a significant risk of overfitting. By utilizing transfer learning with a pre-trained **ResNet-18** backbone, I leverage visual feature extractors (edges, gradients, textures) already optimized on ImageNet. Fine-tuning these pre-trained representations allowed the model to adapt to subtle plant lesion patterns rapidly, reaching high validation accuracy in just 3 epochs.

---

## 2. Results

| Split | Accuracy | Notes |
|---|---|---|
| Lab Test Set (Overall) | **95.31%** | Achieved in 3 epochs using ResNet-18 fine-tuning with AdamW optimizer (`lr=0.001`, `batch_size=256`). |
| Worst Classes | **88.0% – 91.5%** | Fungal spot lesions within the Solanaceae family (e.g., `Tomato___Septoria_leaf_spot` vs. `Tomato___Target_Spot`). |
| Real-World / Field Images | **30.0% – 40.0%** | Tested on unstructured outdoor photos; severe performance drop driven by domain shift and background shortcut learning. |

---

## 3. Guiding Questions (Graded)

1. **Why transfer learning? Why not train from scratch? What does the pretrained backbone give you?**  
   Training a deep architecture from scratch requires millions of domain-specific samples to avoid memorization and build hierarchical representations from scratch. A pre-trained ResNet-18 backbone provides robust low- and mid-level feature maps learned from ImageNet. By keeping these feature extractors and fine-tuning the classification head, it drastically reduce training time, stabilize gradient updates, and achieve over 95% accuracy in just 3 epochs without needing millions of parameters to be learned from a cold start.

2. **Per-class performance. Which classes does it confuse, and why (visual similarity, class imbalance)?**  
   Analyzing the confusion matrix reveals that misclassifications primarily cluster around morphologically similar fungal leaf spots within the same plant family, such as *Tomato Septoria Leaf Spot* versus *Tomato Target Spot*. Because early-stage lesions share almost identical color palettes and circular geometries, downsampling images to 224x224 input resolution blurs the fine micro-textures required to cleanly separate these subtle class boundaries.

3. **Lab-to-field reality check. Run it on real-world images. What happened, and why? (This is the core of the assignment.)**  
   When evaluating the trained model on unstructured, real-world field photos, top-1 accuracy dropped significantly to roughly 30%–40%. This drastic performance collapse stems from **shortcut learning**: during training on the PlantVillage dataset, the network implicitly learned to associate clean, uniform studio backgrounds (solid grey/black) with valid leaf features. When exposed to real field conditions with outdoor clutter, direct sunlight glare, and soil, the model's feature activations broke down.

4. **Augmentation strategy. Which augmentations did you use, and did they help generalization? Show numbers.**  
   The baseline pipeline used standard evaluation transforms: resizing (`Resize((224, 224))`) and ImageNet normalization. While these transforms preserved strong lab test set accuracy (95.31%), the lack of heavy domain-randomization augmentations (such as random background substitution, CutMix, or severe brightness/contrast jitter) meant the model remained brittle and incapable of generalizing beyond studio-like laboratory conditions.

5. **Overfitting diagnosis. Learning curves — is the model memorizing the clean lab background?**  
   The training and validation loss curves tracked closely together (Train Loss: 0.1577, Val Loss: 0.1529), indicating no traditional internal overfitting within the PlantVillage distribution. However, the model suffered from **out-of-domain overfitting**. Rather than learning invariant biological leaf pathology, it memorized the high-contrast boundary between isolated studio leaves and sterile backdrops as an unintended shortcut feature.

6. **The domain gap. What specifically differs between lab and field images that breaks the model?**  
   Laboratory images feature isolated, centered leaves laid flat against uniform backdrops with perfectly controlled lighting. Field images introduce complex environmental factors: dynamic shadows, direct sunlight overexposure, soil clutter, overlapping foliage, and varied camera distances. These external variables create a severe covariate shift that shifts the input feature distribution far beyond what the laboratory-trained model ever experienced.

7. **Real-world deployment. What data and steps would you need to make this work in a farmer's hand?**  
   To build a production-ready mobile application for agronomists, we must retrain the network on native in-the-wild datasets (such as PlantDoc) combined with aggressive background-swapping augmentations. Furthermore, the deployment pipeline should incorporate Out-of-Distribution (OOD) uncertainty checks (e.g., softmax entropy thresholds) so that when confidence is low, the app prompts "Uncertain / Consult Agronomist" rather than returning a confident wrong answer.

8. **Cost of error analysis. In agriculture, what is the cost of a false "healthy" vs a false "diseased"? How should the threshold reflect that?**  
   The risk matrix in agricultural diagnosis is highly asymmetric. A **False Negative** (predicting "healthy" when a pathogen is present) is catastrophic—an unchecked disease like Late Blight can spread rapidly and destroy an entire crop yield, threatening a farmer's livelihood. Conversely, a **False Positive** (predicting "diseased" when healthy) results in unnecessary localized pesticide application—a financial cost, but one that protects the primary harvest. Consequently, decision thresholds must be calibrated to maximize recall for disease classes.

---

## 4. Model Card (lab-to-field)

```text
# Model Card · lab-to-field

## 1. Overview
- Option / dataset / classes: PlantVillage dataset featuring 38 fine-grained classes covering healthy crops and specific fungal, bacterial, and viral plant pathogens across multiple crop species (e.g., Tomato, Potato, Apple, Corn).
- Backbone and what you froze vs fine-tuned: Transfer learning using a pre-trained ResNet-18 architecture. The feature extraction layers were adapted alongside a fully connected classification head, optimized using AdamW and Cross-Entropy Loss over 3 epochs.

## 2. Performance (lab test set)
- Overall accuracy: 95.31% on the held-out validation/test set.
- Worst classes (confusion matrix) and why: Minor misclassifications occurred between morphologically similar spot diseases within the Solanaceae family (e.g., Tomato___Septoria_leaf_spot vs. Tomato___Target_Spot). The confusion matrix indicates that fine texture variations become ambiguous when downsampled to 224x224 input resolution.

## 3. The reality check
- Accuracy on real field images: Performance drops significantly when evaluated on unstructured outdoor field photos.
- What specifically broke between lab and field?:
  1. Background Shortcut Learning: PlantVillage images feature solid grey/black studio backgrounds. The model implicitly relies on these clean backgrounds during lab training.
  2. Domain Shift & Noise: Real-world field conditions introduce complex soil textures, sunlight glare, overlapping leaves, and shadows that distort feature activations.
  3. Framing & Scale: Studio images center isolated leaves, whereas mobile field photos contain partial leaves and scale variations.

## 4. Limitations & ethics
- Cost of a false 'healthy' vs a false 'diseased' in agriculture:
  - False 'Healthy' (False Negative): High risk. An unchecked infection (e.g., Late Blight) can rapidly destroy an entire crop yield, leading to severe financial loss.
  - False 'Diseased' (False Positive): Moderate risk. Leads to unnecessary pesticide application costs and environmental runoff, but protects the overall yield. Decision thresholds should prioritize recall for disease classes.
- Who could be harmed if this were deployed as-is?: Farmers relying autonomously on mobile diagnostic predictions without agronomic verification.

## 5. Real world
- What data and steps would make this work in a farmer's hand?:
  1. Field Data Fine-tuning: Retrain using native field datasets (e.g., PlantDoc) containing complex outdoor backgrounds.
  2. Data Augmentation: Implement heavy color jitter, random background substitution, and CutMix/Mixup.
  3. Uncertainty Calibration: Incorporate Out-of-Distribution (OOD) detection so the model prompts "Consult an Agronomist" when prediction confidence is low.
  4. Edge Deployment: Quantize the model to INT8 for fast, offline mobile inference in rural areas.

```
---

## 5. Reflection
**What surprised you in the reality check? What would it take to make this trustworthy in the field?**

This assignment provided a clear, practical lesson in the gap between benchmark accuracy and real-world model deployment. Reaching **95.31% accuracy** on the PlantVillage validation set initially suggested a near-production-ready model. However, stress-testing the pipeline against unstructured field photos exposed how easily deep neural networks can exploit dataset artifacts—in this case, relying on sterile studio backgrounds as a shortcut feature.

To make computer vision models truly trustworthy in agricultural settings, It must look beyond clean laboratory benchmarks. Real-world reliability demands training on unconstrained field environments (e.g., PlantDoc), applying heavy domain randomization, implementing out-of-distribution detection, and optimizing pipelines for offline edge execution on low-cost mobile hardware.


---

