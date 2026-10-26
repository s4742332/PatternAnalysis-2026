# Required Feasibility

This feasibility assessment is for section 2D U-Net for the HipMRI 2D Slices annotation assistant.

### User Need

The end users are imaging specialists, they require specific, defined and clear boundaries between the prostate and OARs (Organs at Risk) around it. This is an assistant tool, as such it's to increase speed of information about a scan for a user to know.

### Scope

- Use HipMRI2D Slices
- Target DICE $\geq$ 0.75 on prostate label on test set
- Apply 2D U-Net

### Acceptance Criterea

- DICE $\geq$ 0.75 on prostate label on test set containing only patients that were not seen in training
- No patient appears in more than once in a training and testing session
- U-NET beats baseline model on prostate DICE under identical conditions
- DICE must be seperate

### Model Choice and Concepts

2D U-Net will be used as the base model. U-Net is an CNN archtecture specific to segementation tasks. The task requires the identification and seperation of propstate and OARs within a set of 2D images and thus is suited to the task.

### Preliminary Evidence

no implementations as of now

### Risks, Budget and Fallback

As the user need is for speed of information to help, it should be



Risks include but are not limited to:

- Missing, corrupted, and non-uniform case data of a patient over the course of the 60 weeks. This has been identified as keras_slices_train missing patient 034. Inequal case lengths between patients (1 week vs 8 weeks for 1 patient case for example). Non alignment of folder IDs between the segmented masks and the base images.
