# Carve/tracer_b7Cog model

[![Try a demo on Replicate](https://replicate.com/lucataco/remove-bg/badge)](https://replicate.com/lucataco/remove-bg)

This is an implementation of the [Carve/tracer_b7](https://huggingface.co/Carve/tracer_b7) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i image=@shoe.jpg

## Example

Input shoe image

![alt text](shoe.jpg)

Output shoe image with background removed

![alt text](output.png)

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The model weights are not covered by this license. The [Carve/tracer_b7](https://huggingface.co/Carve/tracer_b7) segmentation weights are Apache 2.0. The FBA matting weights used for edge refinement were trained on the Adobe Deep Image Matting dataset and are for non-commercial use only (see [FBA_Matting](https://github.com/MarcoForte/FBA_Matting)).
