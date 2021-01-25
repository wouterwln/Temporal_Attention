<h1>Temporal Attention Module implementation</h1>
This repository contains a first implementation of Temporal Attention Modules as described in my seminar report. The class TemporalAttentionModule can be added to any pytorch model and will facilitate Temporal Attention. This class utilizes helper classes VisualAttentionHead and TemporalAttentionHead, which are implemented in the same file. During training, batches have to be individual videos, as we assume that batches are still nx3xwxh in shape. This approach is therefore not fit for single image segmentation, as it will draw information from other images in the batch we are processing. 
<h2>Important details</h2>
To run this, you'll need an installation of pytorch. Furthermore, due to the short longevity of the seminar project and the low emphasis it has on implementation, some of the code in this repository is not as desired. For example, the layers are hardcoded to only work on CUDA devices, and also the amount of Temporal Attention Heads as well as the stabbing depth are hardcoded. Both of these details were hardcoded as these were not the emphasis of the project, and the general robustness of the code in the repo had smaller emphasis than generally being able to realize what I described in my seminar report.
Anyway, I hope that, even though there are some minor imperfections, you are still able to appreciate this code.