import torch
import torch.nn.functional as F
from utils.ops import conv
class PIAFusion():
    def CMDAF(self, F_vi, F_ir):
        sub_vi_ir = torch.sub(F_vi, F_ir)
        sub_w_vi_ir = torch.mean(sub_vi_ir, dim=[2, 3], keepdim=True)  # Global Average Pooling
        w_vi_ir = torch.sigmoid(sub_w_vi_ir)

        sub_ir_vi = torch.sub(F_ir, F_vi)
        sub_w_ir_vi = torch.mean(sub_ir_vi, dim=[2, 3], keepdim=True)  # Global Average Pooling
        w_ir_vi = torch.sigmoid(sub_w_ir_vi)

        F_dvi = torch.mul(w_vi_ir, sub_ir_vi) # 放大差分信号，此处是否应该调整为sub_ir_vi
        F_dir = torch.mul(w_ir_vi, sub_vi_ir)

        F_fvi = torch.add(F_vi, F_dir)
        F_fir = torch.add(F_ir, F_dvi)
        return F_fvi, F_fir

    def Encoder(self, vi_image, ir_image, reuse=False):
        channel = 16
        with torch.no_grad():
            x_ir = conv(ir_image, channel, kernel_size=1, stride=1, padding=0, scope='conv5x5_ir')
            x_ir = F.leaky_relu(x_ir)
            x_vi = conv(vi_image, channel, kernel_size=1, stride=1, padding=0, scope='conv5x5_vi')
            x_vi = F.leaky_relu(x_vi)
            block_num = 4
            for i in range(block_num):  # the number of resblocks in feature extractor is 3
                input_ir = x_ir
                input_vi = x_vi
                with torch.no_grad():
                        # conv1
                    x_ir = conv(input_ir, channel, kernel_size=3, stride=1, padding=1, scope='conv3x3')
                    x_ir = F.leaky_relu(x_ir)
                with torch.no_grad():
                    # conv1
                    x_vi = conv(input_vi, channel, kernel_size=3, stride=1, padding=1, scope='conv3x3')
                    x_vi = F.leaky_relu(x_vi)
                # # want to use one convolutional layer to extract features with consistent distribution from various sourece images  
                if i != block_num - 1:
                    channel = channel * 2
                    x_vi, x_ir = self.CMDAF(x_vi, x_ir)
            print('channel:',  channel)
            return x_vi, x_ir


    def Decoder(self, x, reuse=False):
        channel = x.size()[-1]
        print('channel:', channel)

        with torch.no_grad():
            block_num = 4
            for i in range(block_num):  # the number of resblocks in feature extractor is 3

                features = x
                x = conv(features, channel, kernel_size=3, stride=1, padding=1, scope='conv{}'.format(i + 1))
                x = F.leaky_relu(x)
                channel = channel / 2
            print('final channel:', channel)
            x = conv(x, 1, kernel_size=1, stride=1, padding=0, scope='conv1x1')
            x = torch.tanh(x) / 2 + 0.5
            return x

   
    def PIAFusion(self, vi_image, ir_image, reuse=False, Feature_out=True):
        vi_stream, ir_stream = self.Encoder(vi_image=vi_image, ir_image=ir_image, reuse=reuse)
        stream = torch.cat([vi_stream, ir_stream], dim=-1)
        fused_image = self.Decoder(stream, reuse=reuse)
        if Feature_out:
            return fused_image, vi_stream, ir_stream
        else:
            return fused_image
