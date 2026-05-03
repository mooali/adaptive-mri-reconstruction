# Adaptive MRI Acquisition — Extension Guide

## 1. Overview

This section proposes an extension to the current MRI reconstruction pipeline by introducing an **adaptive acquisition strategy**.

Instead of always acquiring the full MRI volume, the system dynamically decides whether reconstruction is sufficient or whether full acquisition is required.

The objective is to:

- reduce scan time  
- maintain diagnostic safety  
- leverage AI as a decision-making component  

---

## 2. Core Idea

The system follows a **conditional acquisition strategy**:

1. Acquire only part of the scan (fast)  
2. Analyze the acquired data using AI  
3. Decide:
   - reconstruct missing slices (fast path)  
   - or continue full acquisition (safe path)  

---

## 3. Proposed Pipeline

| Phase | Action | Outcome |
|------|--------|--------|
| **Phase 1 — Fast Scout** | Acquire every 2nd slice (50% sampling) | Reduced scan time |
| **Phase 2 — Detection** | Run anomaly detection / uncertainty estimation | Normal vs suspicious |
| **Phase 3a — Normal** | Reconstruct missing slices using U-Net | Fast full volume |
| **Phase 3b — Suspicious** | Acquire remaining slices normally | Full real scan |

---

## 4. Decision Mechanism

The key component is a **safety gate** that determines whether reconstruction is acceptable.

### 4.1 Uncertainty-Based Decision

We use:

- Monte Carlo Dropout for uncertainty estimation  

Procedure:

1. Perform multiple forward passes through the U-Net  
2. Compute prediction variance  
3. Aggregate into a global uncertainty score  

Decision rule:

```python
if uncertainty < threshold:
    decision = "RECONSTRUCT"
else:
    decision = "FULL_ACQUISITION"
```
### 4.2 Optional: Anomaly Detection

A more advanced system includes an anomaly detector:

- trained on healthy vs pathological MRI scans  
- outputs normal vs suspicious classification  

Final decision:

SAFE = (low uncertainty) AND (no anomaly detected)

---

## 5. Why This Approach Is Viable

### 5.1 Anatomical Prior

- Peripheral regions are structurally simpler  
- Central brain regions contain more complex structures  

This makes reconstruction more reliable in large parts of the scan.

---

### 5.2 Asymmetric Risk Design

- Reconstruction is applied only in low-risk cases  
- Suspicious cases trigger full acquisition  

This ensures safety while still achieving acceleration.

---

### 5.3 Uncertainty as a Safety Signal

Uncertainty estimation enables the model to:

- identify difficult reconstruction regions  
- detect when predictions are unreliable  
- trigger fallback automatically  

---

## 6. Required Components

### 6.1 Reconstruction Model

- Existing U-Net architecture  
- No modification required  

---

### 6.2 Uncertainty Estimator

- Monte Carlo Dropout  
- Produces pixel-wise and global uncertainty  

---

### 6.3 Anomaly Detector

- Requires pathological dataset  
- Not implemented in current project  

---

### 6.4 Acquisition Controller

- Interfaces with MRI scanner  
- Enables conditional acquisition  

⚠️ Not implementable in this project — conceptual only

---

## 7. Minimal Proof-of-Concept (Recommended)

To demonstrate feasibility within this project, implement:

- Monte Carlo Dropout  
- Uncertainty map visualization  
- Global uncertainty score  
- Threshold-based decision output  

Example output:

Decision: SAFE — reconstruction applied  

or  

Decision: UNSAFE — full acquisition recommended  

---

## 8. Limitations

- No pathological data available  
- Threshold selection is heuristic  
- No scanner integration  
- Evaluation limited to PSNR and SSIM  

---

## 9. Research Contribution

This extension introduces:

AI-driven conditional MRI acquisition based on uncertainty

Unlike standard pipelines:

- AI does not only reconstruct  
- AI actively controls the acquisition process  

This represents a shift from passive reconstruction to active decision-making.

---

## 10. Future Work

- Train anomaly detector on pathological MRI data  
- Evaluate lesion preservation  
- Add uncertainty calibration  
- Integrate with scanner acquisition systems  
- Perform clinical validation studies  

---

## 11. Key Takeaway

The goal is not only better reconstruction  
but smarter acquisition using AI.