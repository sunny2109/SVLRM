# [Spatially Variant Linear Representation Models for Joint Filtering](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Spatially_Variant_Linear_Representation_Models_for_Joint_Filtering_CVPR_2019_paper.pdf)

## Dependencies

- Python 3.8
- PyTorch = 1.5
- numpy
- os
- cv2
- glob
- logging

## Training
I trained and tested the model on a single NVIDIA RTX 2080Ti GPU, and this process took about 2 days for 20w iterations.The training strategy is the same as paper.
```bash
#x4
python train.py --upscaling_factor 4
#x8
python train.py --upscaling_factor 8
#x16
python train.py --upscaling_factor 16
```

## Testing

```bash
python test.py --upscaling_factor 8 --model weights/X8/model_130000_iter.pth
```

## Results

## Acknowledgements
[SVLRM](https://www.dropbox.com/s/1z9ps20welw3c9a/CVPR19_SV_code.zip?dl=0)
