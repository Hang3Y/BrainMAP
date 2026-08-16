# BrainMAP: Modality-aware Anatomy-guided Predictive Representation Learning for Brain MRI

BrainMAP is a domain-specific foundation representation-learning framework for brain MRI. It incorporates anatomy-frequency-guided masked latent prediction, modality supervision, and acquisition-perturbation consistency within a student-teacher framework.

## Abstract

Magnetic resonance imaging (MRI) plays a central role in the diagnosis and follow-up of brain tumors, neurodegenerative diseases, stroke, and other neurological disorders. Foundation models offer a new representation-learning paradigm for brain MRI analysis across multiple tasks. However, existing approaches often adopt generic self-supervised objectives and overlook the explicit modeling of heterogeneous anatomical structures, sequence-dependent tissue contrast, and acquisition-related variation in brain MRI. We propose BrainMAP (Modality-aware Anatomy-guided Predictive Representation Learning), a domain-specific foundation representation-learning framework for brain MRI. Built on masked latent prediction, BrainMAP integrates anatomy-frequency-guided masking, modality supervision, and acquisition-perturbation consistency within a unified student-teacher framework. This design prioritizes information-rich structural regions, preserves tissue contrast across MRI sequences, and reduces the influence of nonpathological acquisition variation. We pretrain BrainMAP on a large-scale brain MRI pool assembled from 14 public datasets and evaluate its transferability in MRI modality classification, molecular-status classification, cognitive classification, tumor segmentation, and overall survival prediction.

## Motivation

![BrainMAP motivation](assets/fig1.png)

**Figure 1.** Generic self-supervised pretraining versus BrainMAP for brain MRI. Both learn representations from large-scale brain MRI data and transfer them to disease classification, survival prediction, cognitive classification, and lesion segmentation. Generic approaches use general-purpose objectives without explicitly modeling brain MRI properties. BrainMAP incorporates heterogeneous anatomy, acquisition-related appearance variation, and sequence-dependent tissue contrast into pretraining.

## Framework

![BrainMAP framework](assets/fig2.png)

**Figure 2.** BrainMAP domain-specific pretraining framework. On the left, three views are constructed from the same brain MRI: an anatomy-frequency-guided masked view based on brain-region intensity, tissue boundaries, local variance, and Fourier response, a sequence-labeled clean view, and a perturbed artifact view. In the center, the online student encodes all three views, whereas the EMA teacher encodes only the clean view and provides patch-level targets and a global reference representation. On the right, structure-weighted latent prediction, sequence classification, and clean-artifact representation consistency jointly optimize the student network. Teacher parameters are updated as an EMA of student parameters.

## Repository Status

This repository currently provides the public method implementation and data-preprocessing utilities:

- Model architecture and EMA teacher.
- Anatomy-frequency-guided masking and pretraining objective definitions.
- WebDataset data contract and modality mapping.
- Brain MRI preprocessing and WebDataset packaging utilities.
- Reference configuration for BrainMAP pretraining.

Additional materials will be released according to the progress of the paper.

## Directory Structure

```text
brainmap/
|-- assets/          # Repository figures
|-- config/          # Reference pretraining configuration
|-- data/            # Data contract, loader, and modality mapping
|-- masking/         # Anatomy-frequency-guided masking
|-- model/           # BrainMAP model architecture
|-- objectives/      # Pretraining objective definitions
|-- preprocessing/   # Brain MRI preprocessing and WebDataset packaging
`-- utils/           # Shared utility modules
```


## Notice

This repository is intended for academic research communication. Do not redistribute private medical images, non-public annotations, checkpoints, trained weights, or experiment logs unless they are explicitly approved for public release.
