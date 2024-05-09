import os
import torch.utils.data
from models.esa_net import ESA_net
from avcc import AVCC



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = ESA_net()
    model = model.cuda()


    scale_times=640.0
    resume_path='/home/houyi/Code/activevcc/weights/esa_net.pth.tar'

    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])


    avcc=AVCC()
    avcc.infer_test(model,scale_times)
