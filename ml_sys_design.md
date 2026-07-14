# ML System Design

org certificate:

* [AI Platform on Microsoft Azure Specialization](https://teams.public.onecdn.static.microsoft/evergreen-assets/safelinks/2/atp-safelinks.html)
* [AWS Prescriptive Guidance](https://docs.aws.amazon.com/pdfs/prescriptive-guidance/latest/gen-ai-workload-assessment/gen-ai-workload-assessment.pdf)
* [GCP Audit Manager](https://cloud.google.com/products/audit-manager?hl=en)
* [ISO/IEC 42001](https://www.iso.org/standard/42001) [Azure ISO 42001](https://learn.microsoft.com/en-us/compliance/regulatory/offering-iso-42001) [AWS ISO 42001 FAQS](https://aws.amazon.com/compliance/iso-42001-faqs/) [GCP ISO 42001](https://cloud.google.com/security/compliance/iso-42001?hl=en)

## Discover & Clarify: Business Objective, Scope, Requirements, Constraints

Business Objective, vision & goal. What is success? KPIs:

* annual recurring revenue ARR, net profit margin NPP
* customer aquisition cost CAC, churn rate
* customer satisfaction score CSat, net promoter score NPS
* average order value AOV, customer lifetime value CLtV
* order fulfillment cycle time OFCT, inventory turnover COGS/avgINV

Functional Requirements: UX, features, session memory, continuous improvement

Scale: Daily Active Users (DAU) x Click Per User / 100K = Query Per Second (QPS), ingestion vs query frequency, rate of growth, data volume of knowledge base,

Constraints:

* latency budget (time to first meaningful byte),
* cost of data gathering (label) or storage
* cost of processing on CPU/GPU/RAM
* interpretability, explainability, and simplicity of baseline
* build vs buy (cloud) vs reuse

Data: source, size, type, ground truth


## Business Metrics

Note: business KPI ← online metric ← offline metric ← validation metric ← training loss. Each arrow is a surrogate relationship with a gap. Imperfect and comes with a risk.

* Business KPI: the bottom line metric to improve
* Online metric: the real impact number to optimize for
* Offline metrics: the complete holdout report performance
* Validation metric: the single number to early-stop and model-select on
* Training loss: the model's loss on the training data

Online metric:

* User feedback
    * Explicit feedback: User likes, valid dislike/escalation/complain/appeal (hard negative), Perceptual Mean Opinion Score (MOS), appeal rate, overturn rate on appeals
    * Implicit feedback: glance view (impression), query reformulation rate, Click-Through Rate (CTR, risk clickbait), watch time, completed watch, successful clicks, retention A/B
* Operational performance:
    * time-to-detect/process, dwell time, stockout / overstock cost, ETA
    * SLA breach rate, fraud $ prevented, revenue error
    * self-service (deflection) rate
* Safety
    * field incident / miss rate
* HITL:
    * expert confirm rate
* Prevalence

Guardrails: fairness and bias (age, gender, ethnicity) with constrastive evaluation


## Solution Design

I/O spec:
 
* input entities → output type (continous value | (ordinal or categorical) class | ranked list | segment/structure | generated)
* at what granularity (user / item / pair / session),
* what cadence (one-shot vs sequential).

Label spec: exact positive definition, source (explicit / implicit / synthetic), proxy risks (eg. clickbait).

API type: 

* REST (simple CRUD with HTTP),
* GraphQL (selective query),
* WebSocket (real-time two-way over TCP),
* SOAP (secure XML),
* gRPC (high-speed service-to-service)

Non-ML baseline first: rules / popularity / heuristic. ML must beat this.


###  ML Task Framing (business → ML) top 2 with rejection criteria:

1. Paradigm from label availability: supervised / self-supervised / unsupervised / RL.
2. Candidate framings (2–3 of the SAME problem), e.g. recommendation sys as pointwise classification vs pairwise ranking vs retrieval+ranking.
3. Pick one; reject runner-up against: label cost, metric alignment, latency, data volume, interpretability.


| Task framing | Offline test metrics | Data object (training example) | Output head | Validation metric (select/stop) | Training loss |
|---|---|---|---|---|---|
| Regression | [MAE](#mae), [RMSE](#rmse), [MAPE](#mape), [R²](#r); [interval coverage](#interval-coverage) | (x, y∈ℝ) | single linear unit; [quantile heads for intervals](#quantile-heads-for-intervals) | [RMSE](#rmse) on [temporal val split](#temporal-val-split) | [MSE](#mse) / [Huber](#huber); [pinball for quantiles](#quantile-heads-for-intervals) |
| Binary clf w/o imbalance / multi-task (K binary heads) | [P](#precision)/[R](#recall)/[F1](##f1)@(0..1), [P@fixed-FPR](#precision), [PR-AUC](#pr-auc), [ROC-AUC](#roc-auc), [calibration (ECE)](#calibration-ece) | (x, y∈{0,1}) | [sigmoid p(y)](#sigmoid-function) | [PR-AUC](#pr-auc) | ([Weighted](#weighted-bce)) [BCE](#bce); [Focal loss](#focal-loss) |
| Multi-class / multi-label clf | per-class/macro [P](#precision)/[R](#recall)/[F1](#f1), [Accuracy](#accuracy), [Hamming loss](#hamming-loss) | (x, y∈K$) / (x, y⊆K) | [softmax](#softmax); K independent [sigmoids](#sigmoid-function) | macro-[F1](#f1) | [CE](#ce); per-label [weighted BCE](#weighted-bce); [KL-divergence](#kl-divergence) |
| Clustering | --- | --- | --- | --- | --- |
| Dimensionality reduction | --- | --- | --- | --- | --- |
| Representation learning | --- | --- | --- | --- | --- |
| Recommendation: rule-based, content-based (similar content), collaborative (similar client), hybrid sequential or parallel, two-stage retrieval + ranking | precision @ k, mAP, diversity (Avg pair-wise embedding distance) | --- | --- | --- | --- |
| Retrieval / candidate generation | recall@k, precision@k, MRR, catalog coverage, ANN latency | (query, positive item, sampled negatives) | two embeddings + dot/cosine | recall@k | InfoNCE / sampled softmax / triplet |
| Pointwise ranking (CTR, feed) | AUC, nDCG@k on sessions, calibration (critical for auctions) | (user, item, context, y=click); multi-task labels | sigmoid pCTR (+ per-engagement heads) | ROC-AUC / logloss | BCE; weighted multi-task sum |
| Information Retrieval / Learning to rank | nDCG, MRR, mAP | (query, doc list, graded relevance) | scoring function s(q,d) | nDCG@10 | pairwise RankNet / listwise LambdaRank |
| Sequence / Image segmentation | text sequence pariwise F1/ARI, img obj boundary mAP@IOU, 1-vs-all recall, FPS | --- | --- | --- | --- |
| Object detection | mAP@[.5:.95], per-class recall, FPS | (image, boxes + classes) | box regression + class scores + objectness | mAP@IoU 0.5 | composite: IoU/smooth-L1 + focal CE |
| Generative | groundedness, relevance, coherence in logical flow (LLM-judge + human), consistency, language fluency, safety rates (self-harm content, hateful or unfair content, violent content, sexual content, protected material, indirect attack/jailbreak), ROUGE recall oriented for summarization, image generation (FID, Inseption score) | (prompt + context, target text); preference pairs | token softmax over vocab | perplexity / judge win-rate on eval set | next-token CE (SFT); DPO on preferences |
| Translation, STT, TTS | BLEU n-gram comparison for translation, METEOR extended BLEU, GLEU sentence-level BLEU, WER for STT/TTS | --- | --- | --- | --- |
| Strategy Development (RL) | --- | --- | --- | --- | --- |
| Combinatorial optimisation | asymm costs | --- | --- | --- | --- |
| Forecasting (time series) | wMAPE, MASE, interval coverage | (series history + covariates, horizon values) | per-horizon point / quantile outputs | rolling-origin wMAPE | MSE / pinball |
| Anomaly detection | precision@k alerts, PR-AUC, alert volume | unlabeled x + few labeled anomalies | anomaly score | PR-AUC on labeled slice | reconstruction error / one-class objective |



## Data Engineering

Clarification: Data source, size, type, ground truth labels

Data Objects: names

Data Gathering and Cost:

* hand labeling (explicit feedback, human judge, select or order)
* natural labeling (implicit feedback, interactions, results, if applicable)
* synthetic data (RLHF)

Data schema: field name, data type (numerical discrete vs continuous, categorical ordinal vs nominal), is pkey, is nullable, foreign key.

Data store:

* SQL: RDBS
* Column-based: RedShift, Cassandra, HBase
* Key/Value: Redis, DynamoDB, CosmosDB
* Document: MongoDB, CouchDB
* Graph: Neo4J

Feature Engineering: 

* balance dataset (over vs under sampling),
* handle missing values (deletion, imputation)
* data transform:
    * numerical: scaling (normalization = min-max scaling, standardization = z-score normalisation, log scaling), discretization (bucketing)
    * categorical: encoding (enumerating, one-hot encoder, embedding learning), 
    * natural language: stemming & lemmatization (remove stop words), tokenization, embedding
    * image: resize, crop, rotate, color shift, color mode (RGB/CMYK)
    * audio: ?
    

# System Architecture

Realistic system: add complexity only if justified

Modeling craft (task → models → pitfalls)

| Task framing | Baseline (incl. non-ML) | Candidate models | Training notes (splits, sampling, pitfalls) |
|---|---|---|---|
| Regression | segment mean/median; last value | linear reg → GBDT → MLP | log-transform skewed targets; clip outliers; temporal split against leakage |
| Binary clf, imbalanced | rules engine; logistic regression | GBDT; DNN | class weights over naive oversampling; recalibrate after any resampling; threshold from cost matrix, not 0.5; label delay (chargebacks arrive weeks late) |
| Multi-class / multi-label | one-vs-rest logistic on TF-IDF | fine-tuned DistilBERT; CNN | per-class thresholds; upsample rare classes; taxonomy drift over time |
| Retrieval | popularity; BM25; co-visitation | matrix factorization (ALS); two-tower + HNSW/ScaNN | in-batch negatives + hard-negative mining; logQ sampling correction; cold-start via content features |
| Pointwise ranking | logistic w/ feature crosses; GBDT | DCN / DLRM; transformer over user history | position bias (position feature at train, fixed at serve, or IPS); temporal split; negative downsampling then recalibrate |
| Learning to rank | BM25 | LambdaMART; cross-encoder re-ranker | split by query, never by doc; click labels are biased vs human judgments; runs as stage 2 after retrieval |
| Object detection | pretrained YOLO, frozen backbone | YOLO family; Faster R-CNN; DETR | transfer-learn, don't train from scratch; flip/crop/color augmentation; NMS and anchor tuning; small-object failure mode |
| Generative | prompt-engineered frozen LLM + RAG | LoRA fine-tune; RAG + re-ranker | prompt-first, fine-tune only when evals prove the gap; curate eval set, guard against contamination; hallucination guardrails; cost/latency per token |
| Forecasting | seasonal naive; ETS/ARIMA | GBDT on lag features; DeepAR / TFT | never random split — rolling-origin backtest; leakage through future-known covariates; hierarchical reconciliation |
| Anomaly detection | per-metric z-score / quantile threshold | isolation forest; autoencoder; one-class SVM | "normal" training data is contaminated; threshold set by alert budget, not statistics; feed confirmed alerts back as labels |

Model selection:

* baseline
* number of stages: one vs multi-stage, early vs late fusion (simpler vs native combined understanding)
* representation: 
    * statistical: Bag of Words (key-count, sparse representation), TF-IDF (normalized BoW), BM25, n-gram
    * learned: embedding (hashmap,matrix factorization), Word2vec (shallow NN, Continuous BoW + Skip-gram), transformer (context aware, DistilBERT), 3D Convolution (frame/video-level)
* similarity:
    * exact (kNN)
    * approximate (tree and/or cluster based, locally sensitive hashing, ScaNN, DiscScann, HNSW)
* search, fusing layer, re-ranking service, guardrails
* classification: decision tree with boosted gradients, random forest, logistic regression, naive bayes, SVM
* regression: linear regression
* ensemble: begging vs boosting
* neural networks: transformer (encoder-decoder), convolutional NN CNN), region proposal network (RPN)
* ranking: rule base, embedding based, learn to rank (point wise Q-V, pairwise Q-V(V,V), list wise Q-(V, V, …, V)) 
* Generative: watermarking (Coalition for Content Provenance and Authenticity,  C2PA)

Model training/fitting:

* Elasticsearch
* dataset: data split to training, validation (k-fold), test (hold-out)
* regularization: dropout, weight regularization (lasso k*sum|w|, ridge k*sum w^2, elastic net), early stopping, batch normalization
* approach: forward feed (layers, bias, activation function, softmax) and backward propagation, contrastive training (feature= query + n objects, label = idx of object among n with highest similarity)
* loss function: 
   * classification: cross entropy loss (log loss, softmax vs ground truth), KL divergence
   * regression: MAE, MSE, RMSE
   * region overlap w Non-Maximum Suppression (NMS), 
* batch, epoch, checkpoint
* distributed training: PyTorch Distributed Data Parallel (copy model, forward minibatch, aggregate loss, sync gradients, updata all)
* Optimisation: stochastic gradient descent, weighted alternating least squares

Serving & Inference:

* data ingestion, data indexing pipeline
* latency budget: batch vs real-time, streaming output
* Caching
* Re-training cadence
* host: cloud vs on-device
* pipeline: stages, gates, release strategy (shadow = dual deployment, A/B testing, p-value)
* model compression: quantization, knowledge distillation, pruning
* monitoring, hardware utilisation, requests/responses, drift in performance (model/data/context drift)

Risk & Safety:

* Failure modes
* Misuse, fairness, human-in-the-loop

# Roadmap & Summary

* MVP to V1 to V2
* Priorities to de-risk
* What to delegate to others


# Definitions

## Accuracy

Accuracy is the fraction of all predictions that are correct. It counts correct decisions. For binary classification, with total samples $n$:

$$
\text{Accuracy} = \frac{TP+TN}{n}
$$

## Hamming loss

Hamming loss is the fraction of all predictions that are wrong. It counts mistakes. In a binary setting it is:

$$
\text{Hamming loss} = \frac{FP+FN}{n} = 1 - \text{Accuracy}
$$

In multi-label tasks, Hamming loss is often preferred over Accuracy because it measures per-label error rate across all label decisions.

## MAE

Mean Absolute Error is the average absolute difference between predictions and true values. Same units as the target (easy to interpret). Every error contributes linearly, so a Kx bigger error counts Kx more.

$$
MAE = \frac{1}{n} \sum |y-\hat{y}|
$$

## MSE

Mean Squared Error: quares the error, so large mistakes get penalized heavily. But it is sensitive to outliers.

$$
MSE=\frac{1}{n}∑(y−\hat{y})^2
$$

## Huber loss

Huber loss is a hybrid of squared error and absolute error. For small errors it behaves like MSE, and for large errors it behaves more like MAE. Use when labels are noisy or occasional extreme values would otherwise dominate training.

$$
L_δ(r)=\begin{cases}
\frac{1}{2} r^2 & \text{if} |r| ≤ δ \\ 
δ(∣r∣−(1/2)δ) & \text{if} |r| > δ
\end{cases}
$$

$$
L_{Huber} = \frac{1}{n} \sum L_δ(y−\hat{y})
$$
 
where $r=y−\hat{y}$.

## RMSE

Root Mean Squared Error is the square root of average squared error, so it penalizes large errors more heavily. It is in the same units as y, and is sensitive to outliers.

$$
RMSE = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}
$$

## MAPE

Mean Absolute Percentage Error is the average absolute error as a percentage of the true value. It is scale-free (%), easy to explain to business users. But it can be unstable or undefined when $y \approx 0$.

$$
MAPE = \frac{100\%}{n} \sum |\frac{y_i-\hat{y_i}}{y_i}|
$$

## R²

$R^2$ is the coefficient of determination, a regression metric for how much variance in $y$ your model explains.

Interpretation:
* $R^2 = 1$: perfect prediction
* $R^2 = 0$: no better than predicting the mean
* $R^2 < 0$: worse than the mean baseline

$$
R^2 = 1 - \frac{ \sum (y-\hat{y})^2}{ \sum (y-\bar{y})^2}
$$

where $\bar{y} = \frac{1}{n} \sum y$ 

## Interval coverage

Interval coverage measures how often the true value falls inside the predicted interval.

For prediction intervals $[L,U]$ and targets $y$, empirical coverage is:

$$
Coverage = \frac{1}{n} \sum 1(L \le y \le U)
$$

where $1(.)$ is the indicator function.


## temporal val split

“Temporal val split” means a validation split that respects time order: train on older data, validate on newer data.

Instead of randomly splitting examples, you split by timestamp so the model is evaluated the way it will be used in production.

Why it matters:

* It prevents leakage from the future into the past.
* Random splits can make results look better than they really are when adjacent events are highly correlated.


## quantile heads for intervals

Quantile heads give you prediction intervals directly, without assuming a distribution. Instead of one output predicting the mean, you output several quantiles (say 0.05, 0.5, 0.95), and the gap between the low and high ones is your interval.

The whole trick is the loss function — the pinball loss (a.k.a. quantile / check loss). For a target quantile τ:
```python
import torch
def pinball_loss(y_true, y_pred, tau):
    e = y_true - y_pred
    return torch.mean(torch.max(tau * e, (tau - 1) * e))
```
For τ=0.5 it's symmetric and you recover the median (MAE).
The asymmetry is what makes it work for quantiles: for τ=0.95 it penalizes under-prediction ~19× harder than over-prediction, so the model learns to sit above 95% of the data.

Architecture — one shared trunk, one linear head per quantile:
```python
import torch; import torch.nn
class QuantileRegressor(nn.Module):
    def __init__(self, in_dim, quantiles=(0.05, 0.5, 0.95)):
        super().__init__()
        self.quantiles = quantiles
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.heads = nn.ModuleList(nn.Linear(64, 1) for _ in quantiles)

    def forward(self, x):
        z = self.trunk(x)
        return torch.cat([h(z) for h in self.heads], dim=1)  # (B, n_quantiles)
```

Training loop sums the pinball loss across heads:
```python
def multi_quantile_loss(preds, y_true, quantiles):
    y_true = y_true.unsqueeze(1)
    losses = []
    for i, tau in enumerate(quantiles):
        e = y_true - preds[:, i:i+1]
        losses.append(torch.max(tau * e, (tau - 1) * e))
    return torch.mean(torch.cat(losses, dim=1))
```
At inference, pred[:, 0] and pred[:, 2] are your 90% interval bounds, pred[:, 1] is the median.
The one gotcha: quantile crossing. Since the heads are independent, nothing stops the model from predicting q05 > q95 in some regions. 

Fixes, cheapest first:
* Sort the outputs post-hoc (torch.sort along the quantile dim) — simple, works fine for most cases.
* Monotone parameterization: predict the lowest quantile plus non-negative increments, e.g. q_k = q_0 + cumsum(softplus(deltas)). Guarantees ordering by construction.
* Single-head conditional quantile: feed τ as an input and train on random τ per batch — one head, infinitely many quantiles, no crossing if you enforce monotonicity in τ.

## Sigmoid function

The sigmoid function maps any real number to a value between 0 and 1:

$$
\sigma (z) = \frac{1}{1+e^{-z}}
$$

$σ(0)=0.5$; as $z → +∞, σ(z) → 1$; az $z → -∞, σ(z) → 0$; 

## Precision

Precision answers: “Of all items predicted positive, how many were truly positive?” High precision means few false alarms.

$$
P = \frac{TP}{TP+FP}
$$

Precision@fixed-FPR means:

$$
\text{Precision}|\text{FPR=α}
$$

1. Choose an acceptable false positive rate target, say $FPR=α$ (for example 0.01).
2. Tune the score threshold so the model operates at that FPR on validation data.
3. Report precision at that operating point.

If you cap $FPR$ at 0.5%, and precision there is 40%, then about 4 in 10 selected items are real positives while staying within your false-alert budget.


## Recall

Recall answers: “Of all truly positive items, how many did we catch?” It is the true positive rate. High recall means few misses.

$$
R = \frac{TP}{TP+FN}
$$

## F1

The harmonic mean of precision and recall.

$$
F1 = \frac{2PR}{P+R}
$$

Why it is used:

* Balances precision and recall in one number.
* Useful when class imbalance exists and both false positives and false negatives matter.
* Range is [0,1]; higher is better.

## PR-AUC

Area under the Precision-Recall curve: plot precision vs recall across thresholds, then take the area. The baseline is roughly the positive class prevalence: 

$$
\text{baseline}=\frac{\text{nb.positive}}{\text{nb.all.samples}}
$$

Why it is useful:
* Focuses on positive-class performance.
* Higher is better;  1.0 is perfect.
* More informative than ROC-AUC on highly imbalanced data.

$$
\text{PR-AUC} = \int_{0}^{1} P(R)\,dR
$$

where $P(R)$ is precision as a function of recall.

## ROC-AUC

The Area Under the Receiver Operating Characteristic curve. It plots:

True Positive Rate (ie. recall): $TPR = \frac{TP}{TP+FN}$

False Positive Rate: $FPR=\frac{FP}{FP+TN}$

for all classification tresholds.

$$
\text{ROC-AUC} = \int_{0}^{1}\mathrm{TPR}(\mathrm{FPR})\,d(\mathrm{FPR})
$$

Interpretation:

* $AUC = 1.0$: perfect ranking.
* $AUC=0.5$: random ranking.
* Probability view: ROC-AUC equals the probability a random positive gets a higher score than a random negative.

It measures ranking quality, not calibration. Two models can have the same ROC-AUC but very different predicted probabilities.

## Calibration (ECE)

Calibration means predicted probabilities should match observed frequencies. Expected Calibration Error (ECE) is a summary of mismatch.

Example intuition: predictions around 0.8 should be correct about 80% of the time. If they are correct only 65%, the model is overconfident in that region.

## KL-divergence

Kullback-Leibler divergence measures how one probability distribution $Q$ differs from a reference distribution $P$:

$$
D_{KL}(P||Q) = \sum P(x) \log \frac{P(x)}{Q(x)}
$$

(continuous version uses an integral).

Intuition:

* $D_{KL}=0$ only when $P=Q$ (almost everywhere)
* Larger value means $Q$ is a worse approximation of $P$.
* Not symmetric: generally $D_{KL}(P||Q) ≠ D_{KL}(Q||P)$

## CE

(Categorical) Cross-Entropy is used for one-of-K classes with true class vector $y_k$ (usually one-hot) and predicted class probs $p_k$ (typically the softmax output). It pushes probability mass onto the correct class.

$$
CE = - \sum y \log p
$$
​
For one-hot labels, this simplifies to 
$− \log p_{\text{true class}}$

## BCE

Binary Cross-Entropy is the binary case of CE. It is used when target is $y∈\{0,1\}$ and model outputs probability $p=P(y=1∣x)$. It heavily penalizes being confidently wrong in binary classification, while creating a smooth (difentiable) curve. It treat every training example equally. For multi-label classification (several can be 1) we use independent BCE per label.

$$
BCE = - [y \log p + (1-y) \log (1-p)]
$$


## Weighted BCE

Weighted Binary Cross-Entropy is a simple baseline used to counter class imbalance or asymmetric mistake costs. Weighted BCE says: “Mistakes on positives hurt more (or less), based on business need.” 

$$
L_{\text{WBCE}} = - \frac{1}{n} \sum [w_1 y \log p + w_0 (1-y) \log(1-p)]
$$

where $p$ is the predicted probability and label $y \in \{0, 1 \}$, $w_{\{1, 0\}}$ are up/down-weight positive and negative classes

## Focal loss

Focal loss is for heavy imbalance with many easy negatives (e.g., detection, rare-event classification). It says: “Stop spending so much effort on easy examples you already get right. Focus on hard ones.” Easy examples get down-weighted automatically. Hard/misclassified examples keep large influence.

$$
L_{\text{focal}} = - \frac{1}{n} \sum α (1-p)^{\gamma} \log (p)
$$

where $p = yp + (1-y)(1-p)$ and $α = yα + (1-y)(1-α)$

## mAP

Mean Average Precision is used for ranking/retrieval/detection quality. $mAP$ rewards putting true positives early in the ranked list, not just finding them eventually

For one query or one class, Average Precision (AP):

$$
AP = \sum P(k) \Delta R(k)
$$

where $P(k)$ is precision at cutoff k, and $\Delta R(k)$ is recall increase at k.

$$
mAP = \frac {1}{K} \sum AP
$$

## Softmax

Softmax converts logits $z_1, ... z_K$ into a probability distribution over $K$ classes:

$$
softmax(z)_i = \frac{e^{z_i}}{ \sum e^z}
$$

Properties:

* output is in (0,1)
* outputs sum to 1
* larger logit → larget class probability
