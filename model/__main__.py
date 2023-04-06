import torch
from model.pixelcnn import PixelCNN
from modules.df_conv2d import DfConv2d
from modules.df_module import check_pixel_parity

with torch.no_grad():

    model = PixelCNN(
        channels=3,
        hidden_channels=128,
        hidden_layers=4,
    )

    # model = DfConv2d(3, 64, 3)

    model = model.to("cuda").eval()
    
    check_pixel_parity(model, (3, 16, 16), (15, 15), batch_size=128)

    # x = model(img, pos=(0, 0))
    # x = model(img, pos=(0, 0))
    # x = model(img)
    # print(x.max())
    # print(x.argmax(-3).shape == img.shape)
    # print(x)
