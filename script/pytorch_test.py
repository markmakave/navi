import torch
import math

# create a fully connected layer (754 in, 10 out) and save it to a file as onnx
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

x = torch.randn(1, 754).cuda()
y = torch.randn(1, 10).cuda()
z = torch.randn(1, 10).cuda()

torch.onnx.export(
    torch.nn.functional.linear(x, y, z),
    (x, y, z),
    "linear.onnx",
    export_params=True
    
)

