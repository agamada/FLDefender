# Reviewers' Comments

## Reviewer 1

### Recommendation: Reject

#### Comments:
Federated learning is vulnerable to model poisoning attacks. This paper proposes two defense methods against model poisoning attacks: differential privacy-based defense (DPD) and selective aggregation-based defense (SAD). DPD and SAD are evaluated on the Fashion-MNIST dataset (using MLP models) and the CIFAR-10 dataset (using CNN models). They are compared against Krum, Median, and Trimmed Mean defenses to demonstrate effectiveness in maintaining high accuracy (ACC) and low attack success rate (ASR).

While the topic of FL poisoning is popular and the paper’s methods seem intuitive, unfortunately, I do not think the strength of the contribution justifies a journal publication. More specifically, my concerns are as follows:

##### **Low Technical Depth and Novelty**  
   In DPD, the main idea is to clip the models and then add appropriately scaled Gaussian noise. In SAD, a subset of parameters in W is aggregated instead of all parameters. Similar approaches (i.e., selective aggregation, noise addition, DP) have been used in many FL works. The paper just uses these approaches without diving deep into them (tailoring them) or advancing them. Thus, it seems like the paper is reusing well-known ideas.

    We readily acknowledge that the core techniques employed in our DPD scheme—differential privacy (DP) and adaptive gradient clipping—are themselves not novel inventions. However, as articulated in the earlier sections of our paper, traditional detection-based defenses are inherently vulnerable to being circumvented by adaptive attacks. The primary contribution of our work lies in the introduction and comprehensive evaluation of the DPD framework across various settings, demonstrating the potent effectiveness of obfuscating malicious features for defending against model poisoning attacks.

    Furthermore, within our SAD component, we introduce a more novel, sign-based scheme that performs probabilistic obfuscation on model updates by leveraging the sign consistency of gradients. This approach offers a fresh perspective on feature aggregation and enhancement of security.

    We will revise the manuscript to more prominently highlight this conceptual advancement and the empirical validity of our probabilistic defense paradigm.

##### **Intuitive but Obvious Defenses**  
   It is unsurprising that if we add noise to parameters or aggregate subsets of poisoned parameters, then ASR will decrease. It would have been much more interesting if these defenses were tailored to the attacks under consideration, or there was formal proof/evidence showing that they are a good fit for certain attacks.

    We thank the reviewer for this insightful comment. We agree that the expectation that noise reduces the Attack Success Rate (ASR) is intuitive. However, our work demonstrates that the central challenge is not just adding noise, but strategically managing the fundamental privacy-utility trade-off to maintain robustness without sacrificing model accuracy.

    Firstly, as investigated in our experiments in Section 5.1, blindly adding noise, while decreasing ASR, significantly degrades the main model's accuracy. Our contribution lies in the careful empirical analysis of this trade-off, demonstrating that a defender must precisely calibrate the noise level to achieve a practical equilibrium. This necessity for careful tuning is a non-trivial insight for deploying such defenses in practice.

    To further strengthen our approach and its analysis, we have conducted additional experiments by introducing a clustering step based on the selection metrics of SAD. Crucially, the results demonstrate that our approach achieves a significantly higher level of robustness compared to existing state-of-the-art defenses such as FLAME and FLDetector under the same threat model.

##### **Weak Experimental Setup**  
   An MLP model is used for Fashion-MNIST. The Dirichlet distribution, which is typically used to simulate non-iid in FL, is not used. The number of clients is k = 20. I would have expected a broader and more advanced experiment setup with 3-4 datasets, SOTA models (e.g., more complex CNNs, fine-tuned MobileNet-type models, etc.), a higher number of clients, SOTA non-iid methods, and so forth.

    We thank the reviewer for their constructive suggestion. We have strengthened the experimental section by:
    (1) adopting the Dirichlet distribution for non-IID partitioning,
    (2) replacing the MLP with a CNN on Fashion-MNIST, and
    (3) increasing the number of clients to 40.

    The updated results confirm our method’s robustness under these more realistic settings. Please see the revised manuscript for details.


##### **Limited Baseline Attacks and Defenses**  
   Baseline attacks used in experimentation are label flipping (with scaling), sign flipping, and min-max. Defenses are Krum, Median, and Trimmed Mean. There are many attacks and defenses published in the FL literature after 2020-2021, but they are not included in this work (example defenses: FLAME, FLARE, FL-Defender, FLTrust, …). It is an important weakness that recent attacks and defenses are not included in the paper, and experimental comparisons against SOTA defenses are not provided.

    Thank you for raising this important point. We agree that comparisons with SOTA methods are crucial.

    In response to your suggestion, we have expanded our experimental analysis to include comparisons with FLAME and FLDetector - two widely-recognized and commonly used defense methods in the literature. The new results demonstrate that our approach achieves superior performance against min-max attacks compared to all baseline methods, while maintaining competitive effectiveness against other attack types.

    These additions, now included in Section 5.3 of the revised manuscript, provide a more comprehensive comparison with mainstream defense approaches and further validate the practical value of our method.

#### Additional Questions:
- **Does the paper present innovative ideas or material?** No
- **In what ways does this paper advance the field?**
- **Is the information in the paper sound, factual, and accurate?** Yes
- **If not, please explain why.**
- **Does this paper cite and use appropriate references?** Yes
- **If not, what important references are missing?**
- **Is the treatment of the subject complete?** No
- **If not, What important details/ideas/analyses are missing?**
- **Should anything be deleted from or condensed in the paper?** No
- **If so, please explain.**
- **Please help ACM create a more efficient time-to-publication process: Using your best judgment, what amount of copy editing do you think this paper needs?** Moderate
- **Most ACM journal papers are researcher-oriented. Is this paper of potential interest to developers and engineers?** Maybe

## Reviewer 2

### Recommendation: Major Revision

#### Comments:
The paper addresses model poisoning attacks in Federated Learning (FL). It introduces a defense mechanism based on eliminating malicious features. The main idea is to identify and remove malicious features that adversaries inject into their local updates to corrupt the global model. The proposed defense methodology consists of three steps:

1. Gradient analysis to identify the malicious features.
2. Model pruning to eliminate those features.
3. Reconstruction to restore utility on benign data.

The defense methodology does not require prior knowledge of attack strategies; thus it is applicable to a wide range of poisoning attacks, including backdoor and untargeted ones.

The paper provides an interesting contribution, but it needs a major revision before being ready for publication.

#### Comments:
##### **Theoretical Guarantees**  
   While intuitively justified, the decomposition assumption (malicious = benign + noise) lacks strong theoretical guarantees under adversarial settings. The paper must provide provable bounds on the error rate of feature filtering.

    We thank the reviewer for their feedback. However, we believe there may be a misunderstanding regarding the core contribution of our work.
    
    After careful consideration, we are unable to identify a "decomposition assumption (malicious = benign + noise)" or a "feature filtering" mechanism in our manuscript. Our method, as presented in Section 4, is primarily based on the obfuscation and mitigation of malicious features through selective aggregation and noise addition, without relying on any explicit assumptions about the composition of malicious updates.

    Therefore, it appears this specific criticism may not directly apply to our proposed approach. We would be grateful if the reviewer could elaborate on which part of our work prompted this comment, and we are fully prepared to address any specific concerns upon clarification.

##### **Scalability**  
   The proposed methodology executes PCA and cosine similarity computations across feature gradients of all clients. This may not scale efficiently to large models or large numbers of clients without distributed computation. The paper should provide a scalability analysis of the proposed methodology.

    We thank the reviewer for raising this important point regarding scalability.

    We kindly note that the concern about computational cost related to PCA and cosine similarity may stem from a misunderstanding. Our proposed approach, comprising the SAD and DPD mechanisms, does not utilize PCA or cosine similarity computations.

    The core operations of our method primarily involve sign-based comparison of gradients and calibrated noise addition—both of which are computationally efficient and introduce negligible overhead compared to standard federated averaging. That said, we fully agree with the reviewer on the value of explicit scalability analysis.

##### **Assumption of Feature Gradient Independence**  
   The key assumption is that malicious features are linearly independent from benign ones. This might not hold for sophisticated adaptive adversaries who embed their attack within the same subspace.

##### **Reconstruction**  
   The reconstruction phase seems a bit underdeveloped in the description. The paper must provide details on the mechanism used to preserve benign utility after pruning and its overhead.

##### **Limited Attack Diversity**  
   Although three attack types are used, more adaptive or stealthy attack types (e.g., ones designed to avoid the detection criteria of the proposed defense methodology) must be considered.

##### **Limited Reproducibility**  
   Parameters and settings are mostly well-documented, but the code is not open-sourced.

#### Additional Questions:
- **Does the paper present innovative ideas or material?** Yes
- **In what ways does this paper advance the field?** The paper proposes two new ideas:
  1. It introduces a feature-level filtering approach, which is more fine-grained than traditional update-level or client-level filtering.
  2. The method described in the paper includes a clever use of cosine similarity of feature gradients to isolate potentially malicious ones.
- **Is the information in the paper sound, factual, and accurate?** Yes
- **If not, please explain why.**
- **Does this paper cite and use appropriate references?** Yes
- **If not, what important references are missing?**
- **Is the treatment of the subject complete?** Yes
- **If not, What important details/ideas/analyses are missing?** See comments to the authors
- **Should anything be deleted from or condensed in the paper?** No
