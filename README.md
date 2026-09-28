# Failure detection and uncertainty in learned spacecraft pose estimation

Biswarup Mukherjee: MSc IPCVai, 2025–2027

This is a work on monocular relative pose estimation for spacecraft rendezvous and
proximity operations. 

## The problem

**Relative pose estimation** means: given an image of a target spacecraft taken by a
camera on a chaser spacecraft, compute where the target is and how it is oriented,
relative to the camera. The answer has six numbers, three of translation in metres,
three of rotation, hence **6-DoF**.

A Guidance, Navigation and Control system consumes those numbers continuously to steer
the chaser. What it needs is not only an estimate but an indication of how far to trust
it. A wrong pose reported with high confidence propagates into the control loop
unchecked. A wrong pose reported with low confidence can be rejected, retried, or
cross-checked against another sensor.

That distinction is the thread through all seven experiments: **can the perception stage
tell when it is wrong, using only quantities available at runtime?**

The question throughout is not which model is most accurate, but
whether a model can tell when it is wrong, at runtime, without ground truth, and
whether that signal survives contact with hardware-in-the-loop imagery, a navigation filter, and a
flight-processor budget.

---

## Summary of findings

Four confidence mechanisms were measured, on two targets and two datasets. All four
fail in the same direction: they report high confidence where the system is most wrong.

| Study | Confidence mechanism | Result |
|---|---|---|
| 1, 2, 3 | RANSAC inlier ratio, three detectors(ORB, SIFT and DISK) | The learned detector (DISK) produced the most detections and was the worst calibrated of the three, ECE 0.232 against 0.091 for ORB |
| 4 | Predicted range distribution, SPEED+ | ResNet-18 collapses to a near-constant prediction on hardware-in-the-loop imagery; translation error rises 14.5× and 15.4× |
| 5 | Monte Carlo dropout, epistemic term | Epistemic uncertainty *falls* to 0.39× and 0.57× on the domains where this network's own error rises 13.4× and 13.9×; ROC area 0.076 and 0.203, below chance |
| 6 | Inlier ratio inside an EKF | Identical confidence on frames whose attitude is wrong by 180°; filter inconsistency traced to a measurement bias: median NEES 8.88 before correction, 3.07 after, against 2.37 for a consistent filter |

Two further results focus on deployment and architecture:

- **The ranking of backbones reverses across the domain gap.** ResNet-18 is more
  accurate on synthetic imagery (0.271 m against 0.366 m median translation error) and
  roughly twice as *inaccurate* as a frozen DINOv2 ViT-S/14 on hardware-in-the-loop
  imagery (3.9–4.2 m against 1.6–2.1 m).
- **The cost of uncertainty depends on where the dropout sits.** A single forward pass
  uses well under half of an assumed 1 Hz perception budget. With dropout only in the
  head, 25 Monte Carlo passes cost little more than one pass; running the whole network
  25 times exceeds the budget for both models.

A geometric result underlies part of this: the synthetic target used in Study 1 and
Experiment 6 is *exactly* symmetric under a 180° rotation about its z axis
(permutation is a bijection, maximum residual 0.000000 m). No method using only these untextured keypoints
can resolve that from a single frame, and the geometric confidence signal is blind to
the resulting failure.

---

## Repository layout

```
src/
    sc_common.py                       target model, camera, renderer, degradations
    speedplus_data.py                  SPEED+ loader, schema handling
    speedplus_fetch.py                 archive verification, subset extraction
    pose_models.py                     backbones, heteroscedastic head, MC dropout
    exp6_ekf_mc.py                     HCW dynamics, EKF, campaign runner
    exp7_efficiency.py                 profiling, quantisation, budget checks

Experiments_Report.pdf                                           concepts, methods and results
Exp1_2_3_pose_estimation_benchmarking_confidence_study.ipynb     ORB, SIFT, DISK detector benchmark, confidence calibration
Exp4_cnn_vs_vit_pose_regression_speedplus.ipynb                  CNN vs ViT on SPEED+, domain gap measured
Exp5_uncertainty_across_domain_gap.ipynb                         aleatoric / epistemic split, calibration
Exp6_hcw_linearised_relative_motion_ekf_monte_carlo.ipynb        HCW dynamics, EKF, Monte Carlo campaign
Exp7_deployment_efficiency.ipynb                                 parameters, MACs, latency, quantisation
```

All notebooks are committed with outputs.

---

## Data

Experiments 4 and 5 use **SPEED+** (Park, Märtens, Lecuyer, Izzo, D'Amico, IEEE
Aerospace Conference, 2022): 59,960 labelled synthetic images of the Tango spacecraft
from the PRISMA mission, plus two hardware-in-the-loop domains captured in Stanford
SLAB's TRON facility — *lightbox*, simulating Earth albedo, and *sunlamp*, simulating
direct sunlight.

The dataset is published on Zenodo as a single 16.9 GB archive
under a CC BY-NC-SA 4.0 licence and is **not** included here.
`src/speedplus_fetch.py` verifies the archive and extracts a working subset.

Studies 1 and 6 use a programmatic wireframe target defined in `src/sc_common.py`, with ground truth.

---

## Reproducing

```bash
pip install -r requirements.txt
```

Notebooks 6 and 7 need no dataset.

Notebooks 4 and 5 need SPEED+. Archive can be downloaded once, then set `ZIP` to its path
and run; extraction of a working subset takes a few minutes. Experiments were run on a Colab
T4.

---

## Scope and limitations

- Training used 8,000 of the 47,966 available SPEED+ synthetic images for 12 epochs (Experiment 4) and 15 epochs (Experiment 5).
  Absolute errors are well above published SPEC2021 entries and should not be compared
  with them; what is compared here is two backbones under an identical budget.
- Attitude is regressed directly as a quaternion, with no keypoint or geometric stage.
  This is the weaker of the two standard approaches and shows in the rotation errors.
- No domain adaptation of any kind was applied. This is deliberate: the object was to
  observe the untreated gap.
- The DINOv2 backbone is frozen, so the comparison is a fine-tuned convolutional
  network against a frozen self-supervised representation with a trained head.
- Experiment 6 uses a simulated trajectory and rendered imagery, because no public
  dataset provides labelled rendezvous sequences at the scale a Monte Carlo campaign
  requires.
- Latency in Experiment 7 was measured on general-purpose hardware. The numbers are
  relative.
- Monte Carlo dropout is a cheap approximation to a posterior. A deep ensemble is the
  stronger estimator, and the negative result in Experiment 5 does not settle the
  question for epistemic uncertainty in general.

---

## References

1. T. H. Park, M. Märtens, G. Lecuyer, D. Izzo, S. D'Amico, "SPEED+: next-generation
   dataset for spacecraft pose estimation across domain gap", IEEE Aerospace
   Conference, 2022.
2. T. H. Park et al., "Satellite pose estimation competition 2021: results and
   analyses", *Acta Astronautica*, vol. 204, 2023.
3. L. Pauly, W. Rharbaoui, C. Shneider, A. Rathinam, V. Gaudillière, D. Aouada,
   "A survey on deep learning-based monocular spacecraft pose estimation: current
   state, limitations and prospects", *Acta Astronautica*, vol. 212, 2023.
4. A. Kendall, Y. Gal, "What uncertainties do we need in Bayesian deep learning for
   computer vision?", NeurIPS, 2017.
5. Y. Gal, Z. Ghahramani, "Dropout as a Bayesian approximation: representing model
   uncertainty in deep learning", ICML, 2016.
6. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern neural
   networks", ICML, 2017.
7. M. Oquab et al., "DINOv2: learning robust visual features without supervision",
   *TMLR*, 2024.
8. A. Martinez, J. Ramirez, F. Cacciatore, P. Gonzalez, "Guidance, Navigation and
   Control for the autonomous rendezvous and docking of cooperative targets",
   ESA GNC-ICATT, 2023.
9. M. J. Tyszkiewicz, P. Fua, E. Trulls, "DISK: learning local features with
   policy gradient", NeurIPS, 2020.
10. K. Cosmas, K. Asami, "Utilization of FPGA for onboard inference of landmark
    localization in CNN-based spacecraft pose estimation", *Aerospace*, vol. 7, 2020.
