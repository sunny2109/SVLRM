import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SVLRM(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_layer = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        feature_block = []
        for _ in range(1, 11):
            feature_block += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
                              nn.LeakyReLU(0.1)
                            ]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.data.normal_(0, math.sqrt(2.0 / n))
                nn.init.constant_(m.bias, 0)

    def forward(self, img_input, img_guide):
        input_tensor = torch.cat((img_input, img_guide), dim=1)
        param = F.leaky_relu(self.first_layer(input_tensor), 0.1)
        param = self.feature_block(param)
        param = self.final_layer(param)

        param_alpha, param_beta = param[:, :1, :, :], param[:, 1:, :, :]
        output = param_alpha * img_guide + param_beta

        return output, param_alpha, param_beta

# def weights_init(m):
#     """ initialize weights of the model """
#     class_name = m.__class__.__name__
#     if class_name.find('Conv') != -1:
#         nn.init.xavier_normal_(m.weight, 0.1)
#         if hasattr(m.bias, 'data'):
#             m.bias.data.fill_(0)
#     elif class_name.find('BatchNorm2d') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)



if __name__=='__main__':
    img_input = torch.randn(1, 1, 6, 6)
    img_guide = torch.ones(1, 1, 6, 6)

    network_s = SVLRM()
    # network_s = network_s.apply(weights_init)
    img_out, param_alpha, param_beta = network_s(img_input, img_guide)
    print(network_s)
    print("theta {0} '\n' beta{1}".format(param_alpha, param_beta))
    print(param_alpha.size(), param_beta.size())

    # img_out = param_alpha * img_guide + param_beta

    print("==========")
    print('img_input', img_input)
    print('img_guide', img_guide)
    print('img_out', img_out)

    diff = nn.L1Loss()
    diff = diff(img_out, img_guide)
    print(diff)