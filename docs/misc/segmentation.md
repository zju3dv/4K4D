# Segmentation

All scripts described in this document assume that the dataset has been restructured according to the [main readme](../../readme.md#dataset-structure).

## [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)

We provides a script for running the robust video matting model in [`inference_robust_video_matting.py`](../../scripts/segmentation/inference_robust_video_matting.py).
Since we're loading from `torchvision`, there's no need to manually setup the trained model weight.
We perform the matting on batches of images, thus if you encountered OOM errors, please try using a smaller batch size by passing in the `--batch_size` parameter to the script.

```shell
# Perform video matting using RVM
python scripts/segmentation/inference_robust_video_matting.py --data_root ${data_root}
```

## [Self Correction Human Parsing](https://github.com/dendenxu/Self-Correction-Human-Parsing)

This repo provides a script ([`inference_schp.py`](../../scripts/segmentation/inference_schp.py)) to use the SCHP model for better segmentation results on humans. Compared to RVM, SCHP might produce cleaner segmentation on humans.

Pure SCHP might not work on images where the human is small since their model is mainly trained on cropped images. Thus [chingswy](https://chingswy.github.io/) modified the process with a detect-crop-schp-restore pipeline and later [dendenxu](https://zhenx.me) ported this to ***EasyVolcap***.

SCHP requires some setup to run correctly.
1. You'll first need to clone our modified version into `3rdparty/self-correction-human-parsing`.
2. Then, download the [official model](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) and place it somewhere not to be picked up by git. Our script takes a `--chkpt_dir` to find your downloaded models.
3. Make sure `detectron2` is correctly installed following the guide in the [main readme](../../readme.md#installation).
4. Finally, you can run the script with the following command.

```shell
# Clone modified SCHP into the 3rdparty directory
git clone https://github.com/dendenxu/Self-Correction-Human-Parsing 3rdparty/self-correction-human-parsing

# Place the official model somewhere (like in 3rdparty/self-correction-human-parsing/data?)
# Make sure detectron2 is installed correctly
python -c "import detectron2" # no error -> OK

# Run the detect-crop-schp-restore pipeline
python scripts/segmentation/inference_schp.py --data_root ${data_root}
```

## [Background Matting V2](https://github.com/PeterL1n/BackgroundMattingV2)

As the name suggests, this model requires background images as input.
If your dataset does not contain those, you should first try to extract a coarse mask using RVM or SCHP and then follow the [background extraction](#background-extraction) section to create a coarse background estimation to be used for BGMTV2.

My general experience is that BGMTV2 usually produces the best results compared to other methods, thus it's worth at least trying it out before making the final decision about which mask to use.

A script [`inference_bkgdmattev2.py`](../../scripts/segmentation/inference_bkgdmattev2.py) is provided to make the process of running the model easier. Along with the `--batch_size` parameter, which controls how much VRAM you'll be using, you can also tune the `--chunk_size` parameter to reduce RAM usage cost. The `--chunk_size` parameter controls how many images will be loaded and saved at once. By loading more images, we can keep the GPU fed and get higher throughput.

Similar to SCHP, you'll also need to download the pretrained model from [BGMTV2's official site](https://github.com/PeterL1n/BackgroundMattingV2). Our script takes a `--chkpt_path` to find your downloaded model.

```shell
# Extract mask with BGMTV2
python scripts/segmentation/inference_bkgdmattev2.py --data_root ${data_root}
```


## Background Extraction

We implemented a simple background extraction script for dynamic scenes with masks (or with extracted masks).
This script assumes a typical ***EasyVolcap*** camera setup, where there're multiple or one fixed camera and multiple video frames for a sequence.
We compute the weighted average of the pixels using the masks as weights to get the background.

This is useful when you want to reconstruct a separated static background model of a dynamic scene or just requires background images for better mask extraction.

```shell
# Fuse background from masks
python scripts/segmentation/extract_backgrounds.py --data_root ${data_root}
```