
import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetWrapper(nn.Module):
    def __init__(self):
        super(DenseNetWrapper, self).__init__()
        self.base_model = models.regnet_y_1_6gf(pretrained=True)
        self.stem=self.base_model.stem
        self.features=self.base_model.trunk_output

    def forward(self, x):
        layers = [0,1,2,3]  # 这里选择第4、第7、第14和第18个层 ## Here we select the 4th, 7th, 14th and 18th layers.
        outputs = []
        x=self.stem(x)
        for idx, module in enumerate(self.features):
            x = module(x)
            if idx in layers:
                outputs.append(x)
        return outputs  
    


if __name__ == '__main__':
    x1 = torch.randn(1,3,256,256)
    # x2 = torch.randn(1,3,256,256)
    model = DenseNetWrapper()
    y1, y2, y3, y4 = model(x1)
    print(y1.shape, y2.shape, y3.shape, y4.shape)  #([1, 48, 64, 64])  ([1, 120, 32, 32])  ([1, 336, 16, 16])  ([1, 888, 8, 8])