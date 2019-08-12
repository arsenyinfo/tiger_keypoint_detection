# Tiger Keypoint Detection

This pipeline was been coded for the [Computer Vision for Wildlife Conservation (CVWC) challenge](https://cvwc2019.github.io/challenge.html).

Some facts: 
- inference speed is ~18.5 frames/sec on a single 1080Ti, given the resized image and including postprocessing; 
- the solution is mostly inspired FPN encoder-decoder architecture; 
- Densenet201 feature extractor is used; 
- the training took ~2h 30m using 2x1080Ti setup;
- public LB mAP is ~0.74 although there was a stupid padding issue https://github.com/arsenyinfo/tiger_keypoint_detection/commit/8943d061a995b622c802c8b0080d136a499c9cae :( 
