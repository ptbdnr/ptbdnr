# ML System Design


## Discover & Clarify: Business Objective, Scope, Requirements, Constraints

Business Objective, vision & goal. What is success? (KPI: customer aquisition cost CAC, customer lifetime value CLTV, revenue, net profit margin, order fulfillment time, inventory turnover, average order value AOV, customer satisfaction score CSAT, net promoter score NPS, churn rate)

Functional Requirements: UX, features, continuous improvement

Scale: Daily Active Users (DAU), click per user, rate of growth, data volume of knowledge base

Constraints:

* latency budget (time to first meaningful byte),
* cost of data gathering (label) or storage
* cost of processing on CPU/GPU/RAM
* interpretability, explainability, and simplicity of baseline
* build vs buy (cloud) vs reuse

Data: source, size, type, ground truth


## Business Metrics

Note: business KPI ← online metric ← offline metric ← validation metric ← training loss. Each arrow is a surrogate relationship with a gap.

* Business KPI: the salient number you optimize for
* Online metric: the single number you optimize for at the edge (e.g. latency, throughput)
* Offline metrics: the complete holdout report performance
* Validation metric: the single number you early-stop and model-select on
* Training loss: the model's loss on the training data

Online metric:

* Explicit feedback: User likes, valid dislike/complain/appeal (hard negative), Perceptual Mean Opinion Score (MOS)
* Implicit feedback: Click-through rate (CTR), glance view (impression), watch time, completed watch,
* Prevalence

Guardrails: fairness and bias (age, gender, ethnicity)


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


| Task framing | Data object (training example) | Output head | Training loss | Validation metric (select/stop) | Offline test metrics | Online proxy |
|---|---|---|---|---|---|---|
| Regression (ETA, LTV, watch time) | (x, y∈ℝ) | single linear unit; quantile heads for intervals | MSE / Huber; pinball for quantiles | RMSE on temporal val split | MAE, RMSE, MAPE, R²; interval coverage | SLA breach rate, revenue error |
| Binary clf, imbalanced (fraud, churn, abuse) | (x, y∈{0,1}), heavy skew | sigmoid p(y) | weighted BCE / focal | PR-AUC | PR-AUC, precision@fixed-FPR, calibration (ECE), confusion at chosen threshold | fraud $ prevented, appeal rate |
| Multi-class (1 of n) / multi-label (k of n) / multi-task (n binary variants) (tagging, moderation) | (x, y = 1-of-K) or (x, y ⊆ K) | softmax / K independent sigmoids | CE / per-label weighted BCE | macro-F1 / mAP | per-class P/R/F1, micro-macro avg, Hamming loss | overturn rate on appeals |
| Clustering |---|---|---|---|---|---|
| Dimensionality reduction |---|---|---|---|---|---|
| Representation learning |---|---|---|---|---|---|
| Recommendation: rule-based, content-based (similar content), collaborative (similar client), hybrid sequential or parallel, two-stage retrieval + ranking |---|---|---|---|---|---|
| Retrieval / candidate generation | (query, positive item, sampled negatives) | two embeddings + dot/cosine | InfoNCE / sampled softmax / triplet | recall@k | recall@k, MRR, catalog coverage, ANN latency | downstream CTR, freshness |
| Pointwise ranking (CTR, feed) | (user, item, context, y=click); multi-task labels | sigmoid pCTR (+ per-engagement heads) | BCE; weighted multi-task sum | ROC-AUC / logloss | AUC, nDCG@k on sessions, calibration (critical for auctions) | CTR, dwell, retention A/B |
| Information Retrieval / Learning to rank | (query, doc list, graded relevance) | scoring function s(q,d) | pairwise RankNet / listwise LambdaRank | nDCG@10 | nDCG, MRR, MAP | query reformulation rate, successful clicks |
| Sequence / Image segmentation |---|---|---|---|---|---|
| Object detection | (image, boxes + classes) | box regression + class scores + objectness | composite: IoU/smooth-L1 + focal CE | mAP@IoU 0.5 | mAP@[.5:.95], per-class recall, FPS | field incident / miss rate |
| Generative (LLM, RAG) | (prompt + context, target text); preference pairs | token softmax over vocab | next-token CE (SFT); DPO on preferences | perplexity / judge win-rate on eval set | groundedness, relevance, coherence (LLM-judge + human), safety rates | thumbs-up, deflection rate, escalations |
| Translation |---|---|---|---|---|---|
| Strategy Development (RL) |---|---|---|---|---|---|
| Combinatorial optimisation |---|---|---|---|---|---|
| Forecasting (time series) | (series history + covariates, horizon values) | per-horizon point / quantile outputs | MSE / pinball | rolling-origin wMAPE | wMAPE, MASE, interval coverage | stockout / overstock cost |
| Anomaly detection | unlabeled x + few labeled anomalies | anomaly score | reconstruction error / one-class objective | PR-AUC on labeled slice | precision@k alerts, PR-AUC, alert volume | analyst confirm rate, time-to-detect |



Offline evaluation (imperfect connection to business):

* Regression: MAE, RMSE
* Classification
    * hard label with discrete classes: confusion matrix, overall metrics (accuracy (TP+TN)/n, Hamming Loss (FP+FN)/n), class-level one-vs-rest (precision, recall, F1), average (macro/micro/weighted avg of precision/recall/F1), precision at fixed FP rate (for async error costs), PR-AUC for binary classification with various thresholds for precision and recall, ROC-AUC for binary classification for recall vs FP)
    * soft label with continuous probabilities: cross-entropy (negative sum of the true probabilities multiplied by the log of the predicted probabilities), KL-divergence 
* Clustering:
* Content segmentation: sequence segmentation (pairwise F1, ARI), image object boundary (mAP @ intersection over union IOU, per-class recall,  FPS, )
* Dimensionality reduction
* Representation learning
* Information retrieval: search/ranking (precision @ k, recall @ k, Mean Reciprocal Rank MRR, mAP, nDCG)
* Generative: quality (groundedness rate, relevance to request, coherence in logical flow and consistency, fluency in language w LLM-judge and human eval) risk and safety (self-harm content, hateful or unfair content, violent content, sexual content, protected material, indirect attack/jailbreak), image generation (FID, Inseption score)
* Translation: NLP (BLEU n-gram comparison for translation, METEOR extended BLEU, GLEU sentence-level BLEU, ROUGE recall oriented for summarization)
* Recommendation: precision @ k, mAP, diversity (avg pairwise embedding distance)
* Reinforcement learning:
* Combinatorial optimisation: asymmetric costs

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
* distributed training
* Optimisation: stochastic gradient descent, weighted alternating least squares

Serving & Inference:

* data ingestion, data indexing pipeline
* latency budget: batch vs real-time, streaming output
* Caching
* Re-training cadence
* host: cloud vs on-device
* pipeline: stages, gates, release strategy (A/B testing, p-value, shadow = dual deployment)
* model compression: quantization, knowledge distillation, pruning
* monitoring, hardware utilisation, requests/responses, drift in performance (model/data/context drift)

Risk & Safety:

* Failure modes
* Misuse, fairness, human-in-the-loop

# Roadmap & Summary

* MVP to V1 to V2
* Priorities to de-risk
* What to delegate to others
