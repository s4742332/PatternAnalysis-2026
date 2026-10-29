
# HipMRI Prostate Segmentation for Radiotherapy Planning using 2D U-Net CNN Architecture

## Introduction

### Problem

In prostate radiotherapy planning, imaging specalist must be able to correctly classify and delinate between the prostate, the femoral hip, the bladder, the rectum, and the body and background to plan the path of an external-beam. This a high precision, high context task to delinate target volumes with milimeter level precision. In the case of under-countering the prostate region, tumours have a chance to regrow and reoccur in the patient. In the case of over-countouring, suurounding organs can become irratiated due to the radiation, causing high levels of patient pain and potential radiation toxicity. Both scenarios ultimately lead to reduced patient health outcomes. As such tools that can assist in this planning process are desired.

### 2D U-Net Architecture



## Solution

### Dependencies

### Reproducibility

### Data and Preprocessing

### Example Inputs and Outputs

## Research Dilemma

## Limitations

## Artifical Intelligence Disclourse

# Required Feasibility

This feasibility assessment is for section 2D U-Net for the HipMRI 2D Slices annotation assistant.

### User Need

The end users are imaging specialists, they require specific, defined and clear boundaries between the prostate and OARs (Organs at Risk) around it. As such, under-contouring or over-contouring poses significant risk to the patients health outcomes after radiotherapy. This is an assistant tool that outlines contours to help inform a specialists decision in pathing the external radiotherapy beam. In summary, this tool needs to be precise, explainable, and be able to be used on varying patients.

### Scope

- Use HipMRI2D Slices provided
- Target DICE $\geq$ 0.75 on prostate label on test set with the 2D U-Net Model
- Apply 2D U-Net

### Acceptance Criterea

- DICE $\geq$ 0.75 on prostate label on test set containing only patients that were not seen in training
- No patient appears in more than once in a training, validating and testing session
- U-NET beats baseline model on prostate DICE under identical conditions
- Report DICE must be seperate for the first, central, and last slice per scan
- Report must be able to label per-class DICE for all 6 labels
- Must be able to be trained on a 6GB GTX 1660 Ti at fp32 with an inference of under 50 ms per slice

### Model Choice and Concepts

*Difficulty Band:* Easy. Despite the implementation cap placed upon this band, this was chosen due to perceived difficulty with the content from a personal standpoint.

*Data:* Patient data was provided in keras_slices_datas. This data is formatted by patient case number, week of scan, and the slices that occured within that scan. These slices are formatted as .nii.gz file. There is a pre-provided patient split between training, validation and testing. However, it should be recognised that any patient should be able to be resplit should there be any particular bias towards any particular case which could reduce a model's ability to generalise the dataset. Between the segmented masks and images, they will need to be jointly transformed for data augmentation for model training. Cases with little representation in the data set may be removed or weighted sampling may be applied so that no slice over-contributes.

*Task:* In order to be able to define boundaries between OARs and the prostate, the model must be able to capture both fine and bigger picture details to classify with high spatial precision. This suggests that a model with an encoder-decoder architecture is suited for this task due to it's ability to downsample and reconstruct high resolution images, with accuracy depending on the model chosen.

*Models:*

2D U-Net will be used as the comparative model. U-Net is an CNN encoder-decoder archtecture specific to segementation tasks. U-Net is particularlly popular in medical imaging and has been widely used for similar tools and tasks. First it successively convolves and pools each layer, then upsamples each convolution layer. Skip connections at mirrored stages of convolving and pooling then allow the network to capture both contextual and spatial features. As such this symmetry then allows for efficient training and preservation of spatial information.

Ronneber's paper on 2D U-Net's claim that skips provide precise localisation. In theory, if skip connections carry across fine details from the encoder to the decoder, they increase the amount of knowledge the model has about the boundaries within that image. This is what makes it particularly ideal for tasks where minute differences in inference can mean catastrophic consequeunces for the patient. As such, *a baseline model of a no-skip 2D U-Net (keeping the same symmetric architecture)* allows for further investigation into segmentation behaviour at boundaries. It then can be seen, by comparing the baseline against the compartive model the extent skip connections being able to reduce leakage at OARs and prostate boundary interfaces.

Although the original paper compared a sliding-window CNN with it's U-Net architecture, as it was the prior best performing method, from an investigative standpoint, it is of significant interest as to the extent skip connection's recover spatial details lost in pooling. As this is the central paper's claim. It is interesting to see the impact on this particular dataset.

Both model's will be evaluated on the same patient test split, on prostate DICE overall, on base/apex/centrak slices, and on OAR boundary leakage.

### Preliminary Evidence

*Data Investigation:*

| Split    | Patients | Scans | Slices | Image Pairs |
| -------- | -------- | ----- | ------ | ----------- |
| train    | 31       | 191   | 11,460 | yes         |
| validate | 4        | 11    | 660    | yes         |
| test     | 3        | 9     | 540    | yes         |

| Split        | Shapes                                        | dtype          | Header zooms                               |
| ------------ | --------------------------------------------- | -------------- | ------------------------------------------ |
| seg_train    | (256, 128):11400 slices, (26, 144): 60 slices | uint8:  11460 | (np.float32(1.0), np.float32(1.0)): 11460} |
| seg_validate | (256, 128): 660 slices                       | uint8: 660     | (np.float32(1.0), np.float32(1.0)): 660    |
| seg_test     | (256, 128): 540 slices                       | uint8: 540     | (np.float32(1.0), np.float32(1.0)): 540    |

| Split           | 0                         | 1                   | 2                                           | 3          | 4         | 5         |
| --------------- | ------------------------- | ------------------- | ------------------------------------------- | ---------- | --------- | --------- |
| seg_train       | 17.2706%                  | '67.5106%'          | '12.4949%'                                  | '1.9416%', | '0.3567%' | '0.4256%' |
| seg_validate    | '13.7591%'                | '70.3316%'          | '1.7475%'                                   | '0.5610%   | '0.5610%' | 0.4958%   |
| seg_test        | 16.1678%',                | '68.2893%'          | '12.2517%'                                  | 2.3716%',  | '0.5804%' | 0.3393%'  |
| Assumed Labels: | Background, not body-part | General Body Tissue | Femoral Heads, left and right labelled as 1 | Rectum     | Bladder   | Prostate  |

![1793063917096](image/Feasability/1793063917096.png)

![Comparison of volumetric-modulated arc therapy and intensity-modulated radiation  therapy prostate cancer plans accounting for cold spots | Radiological  Physics and Technology | Springer Nature Link](https://media.springernature.com/m685/springer-static/image/art%3A10.1007%2Fs12194-019-00502-0/MediaObjects/12194_2019_502_Fig1_HTML.jpg)

Patient Cases Inequalities:

train case 004: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 005: 5 weeks [0, 1, 2, 3, 4]
train case 006: 2 weeks [0, 1]
train case 007: 1 weeks [0]
train case 008: 1 weeks [0]
train case 009: 1 weeks [0]
train case 010: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 011: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 012: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 013: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 014: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 015: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 016: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 017: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 018: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 019: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7] *Good dataset, week 01 will be cropped
train case 020: 4 weeks [0, 1, 2, 3]
train case 021: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 022: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 023: 1 weeks [0]
train case 024: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 025: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 026: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 027: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 028: 1 weeks [0]
train case 029: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 030: 6 weeks [0, 1, 2, 3, 4, 5]
train case 031: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 032: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 033: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
train case 035: 1 weeks [0]
validate case 036: 1 weeks [0]
validate case 037: 1 weeks [0]
validate case 038: 1 weeks [0]
validate case 039: 8 weeks [0, 1, 2, 3, 4, 5, 6, 7]
test case 040: 7 weeks [0, 1, 2, 3, 4, 5, 7]
test case 041: 1 weeks [0]
test case 042: 1 weeks [0]

Notes on the dataset:

- No patient appears in more than one split
- Every scan has exactly 60 slices
- Missing Patient 034 and 001-003 are absent from the test set - assume these are patients whose data was already taken out of the study
- Test set is only 3 patients, if using original patient splitting then report DICE score per patient/scan/(BASE, CENTRAL, APEX)
- 019_week_1 all have 256x144; all others are 256x128
- Labels 0-5 were not originally labelled to their associated segmentation classification. As such reference from [link.springer.com/article/10.1007/s12194-019-00502-0](https://link.springer.com/article/10.1007/s12194-019-00502-0) (image attached) was used for comparisont to correctly identify labels

Other: [github.com/cosmic-cortex/pytorch-UNet/tree/master](https://github.com/cosmic-cortex/pytorch-UNet/tree/master) will be used as a primary reference for model and data augmentation. Their dataset contained PIL images, so this will have to be altered or a wrapper will have to be created along with a no skip connections feature to create baseline

### Risks, Budget and Fallback

Risk include but are not limited to:

*Data Based*

- Missing, corrupted, and non-uniform case data of a patient over the course of the 60 slices per scan. This has been identified as keras_slices_train missing patient 034. Inequal case lengths between patients (1 week vs 8 weeks for 1 patient case for example). Non alignment of folder IDs between the segmented masks and the base images.
- Repeated patient data in sets that lead to defeneralisation of anatomy
- Small tests sets leading to biases towards one patient's slices
- Insufficient scan resoultion leading to inaccurate segmentation at the boundaries
- Base/Apex slices have reduced contrast on cross-sections leading to lower DICE scores and inprecise segmentation at the boundaries

*Time and Hardware Constraints*

- Rangpur A100 GPUs are available for usage for training. However, it should be considered to train for local GTX 1660 Ti in case of unavailablities.
- Final date of project completion: 30/10/2026

*Fallbacks*

- if DICE is below 0.75:
  - training on binaray prostate vs OARs set
  - DICE loss, class weighting, or augmentation of data sets
- If memory runs out:
  - halve batch size or base channel count
- if time allocation runs out prior to completion:
  - increase training time
  - reduce epochs
