import os.path
import logging
import utils

if __name__ == "__main__":
    utils.logger_info('calc-track', log_path='./results/x8_SR_rmse_psnr_ssim-track.log')
    logger = logging.getLogger('calc-track')

    # image path
    sr_path = './results/X8/'
    hr_path = './dataset/test_depth_sr/depth_gt/'

    sr_img = []
    hr_img = []

    for img in utils.get_image_paths(sr_path):
        img = utils.imread_uint(img, n_channels=1)
        sr_img.append(img)
    
    for img in utils.get_image_paths(hr_path):
        img = utils.imread_uint(img, n_channels=1)
        hr_img.append(img)

    if len(sr_img) != len(hr_img):
        print('ERROR: The number is not equal!')

    mean_rmse = 0
    mean_psnr = 0
    mean_ssim = 0
    for i in range(0, len(sr_img)):
        rmse, _ = utils.calc_rmse(sr_img[i], hr_img[i])
        psnr = utils.calc_psnr(sr_img[i], hr_img[i])
        ssim = utils.calc_ssim(sr_img[i], hr_img[i])

        logger.info('Image:{:03d} || RMSE:{} || PSNR:{} || SSIM:{}'.format(i+1, rmse, psnr, ssim))
        mean_rmse += rmse
        mean_psnr += psnr
        mean_ssim += ssim
    mean_rmse =  mean_rmse / len(sr_img)
    mean_psnr =  mean_psnr / len(sr_img)
    mean_ssim =  mean_ssim / len(sr_img)
    logger.info('AVG RMSE: {} || PSNR:{} || SSIM:{}'.format(mean_rmse, mean_psnr, mean_ssim))