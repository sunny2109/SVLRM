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
python test.py --upscaling_factor 8 --model weights/X8/model_195000_iter.pth
```

## Results

- X8

on the left is output of the model, on the right is the corresponding ground truth image
<img src="./results/X8/001065.png" width="400"/> <img src="./results/gt/001065.png" width="400"/>
RMSE:3.2915470145958894 || PSNR:37.78280235489618 || SSIM:0.9761888256026665

<img src="./results/X8/001101.png.png" width="400"/> <img src="./results/gt/001101.png.png" width="400"/>
RMSE:5.210280611855403 || PSNR:33.79358133117786 || SSIM:0.9628326188482944

<img src="./results/X8/001215.png.png" width="400"/> <img src="./results/gt/001215.png.png" width="400"/>
RMSE:4.1548492643583925 || PSNR:35.75969815984982 || SSIM:0.9701610108526342

<img src="./results/X8/001320.png.png" width="400"/> <img src="./results/gt/001320.png.png" width="400"/>
RMSE:6.202878047820046 || PSNR:32.278938753328994 || SSIM:0.9564319583736038

<img src="./results/X8/001436.png.png" width="400"/> <img src="./results/gt/001436.png.png" width="400"/>
RMSE:4.407425071944234 || PSNR:35.24710485172616 || SSIM:0.9789862490518304

## Acknowledgements
[SVLRM](https://www.dropbox.com/s/1z9ps20welw3c9a/CVPR19_SV_code.zip?dl=0)
