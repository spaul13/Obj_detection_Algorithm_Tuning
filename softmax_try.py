import torch, os
x = torch.tensor([[-0.0116,  0.0104,  0.0417]])
#x = torch.tensor([[ 0.0120, -0.0227,  0.0310]])
print(torch.nn.functional.softshrink(x))
#print(torch.nn.functional.sigmoid(x))
#print(torch.nn.functional.logsigmoid(x))
#print(torch.nn.functional.tanhshrink(x))
print(torch.nn.functional.selu(x, inplace=True))
print(os.stat("H:\\drone_video_cp_3\\pic_1_org.png").st_size)
