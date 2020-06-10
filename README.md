# [Spatially Variant Linear Representation Models for Joint Filtering](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Spatially_Variant_Linear_Representation_Models_for_Joint_Filtering_CVPR_2019_paper.pdf)

## Dependencies

- Python = 3.8
- PyTorch = 1.5
- TensorBoard
- numpy
- os
- cv2
- PIL
- glob
- logging

## Training
I trained and tested the model on a single NVIDIA RTX 2080Ti GPU, and this process took about 10 hours for 50w iterations. Except for learning rate update method, the ohter training strategies are the same as paper. We use the following manner for updating lr:
```bash
lr_ = opt.lr * (0.5 ** (epoch // opt.decay_step))
for param_group in optim.param_groups:
    param_group['lr'] = lr_
```
The author uses poly policy:
```bash
lr_ = opt.lr * (1 - float(curr_step) / opt.n_iters)**2
for param_group in optim.param_groups:
    param_group['lr'] = lr_
```

- Command

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
python test.py --upscaling_factor 8 --model weights/X8/model_10000_epoch.pth
```

## Results

- quantitative results (RMSE)

| depth image SR | SVLRM (paper) | Ours | 
| :----- | :-----: | :-----: | 
| x4 | 1.74 | -- |
| x8 | 5.59 | 5.001 | 
| x16 | 7.23 | -- | 

- visual results (X8 depth sr)

on the left is output of the model, on the right is the corresponding ground truth image
<img src="./results/X8/001065.png" width="400"/> <img src="./results/gt/001065.png" width="400"/>

img_001065  RMSE:3.2915470145958894 || PSNR:37.78280235489618 || SSIM:0.9761888256026665

<img src="./results/X8/001101.png" width="400"/> <img src="./results/gt/001101.png" width="400"/>

img_001101  RMSE:5.210280611855403 || PSNR:33.79358133117786 || SSIM:0.9628326188482944

<img src="./results/X8/001215.png" width="400"/> <img src="./results/gt/001215.png" width="400"/>

img_001215  RMSE:4.1548492643583925 || PSNR:35.75969815984982 || SSIM:0.9701610108526342

<img src="./results/X8/001320.png" width="400"/> <img src="./results/gt/001320.png" width="400"/>

img_001320  RMSE:6.202878047820046 || PSNR:32.278938753328994 || SSIM:0.9564319583736038

<img src="./results/X8/001436.png" width="400"/> <img src="./results/gt/001436.png" width="400"/>

img_001436  RMSE:4.407425071944234 || PSNR:35.24710485172616 || SSIM:0.9789862490518304

## Acknowledgements
- [SVLRM_matlab](https://www.dropbox.com/s/1z9ps20welw3c9a/CVPR19_SV_code.zip?dl=0)
- [SVLRM_Pytorch](https://github.com/curlyqian/SVLRM)
