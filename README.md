# Tiger Keypoint Detection

This pipeline was been coded for the [Computer Vision for Wildlife Conservation (CVWC) challenge](https://cvwc2019.github.io/challenge.html).

Some facts: 
- inference speed is ~18.5 frames/sec on a single 1080Ti, given the resized image and including postprocessing; 
- the solution is mostly inspired FPN encoder-decoder architecture; 
- Densenet201 feature extractor is used; 
- the training took ~2h 30m using 2x1080Ti setup;
- private LB mAP will be updated later.
