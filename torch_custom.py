import torch
import timeit



# pytorch two layer network
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H) #1000
        self.linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

if torch.cuda.is_available():
    device = torch.device("cuda")          
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")
start = timeit.default_timer()
dtype = torch.float
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

#x = torch.randn(N, D_in)
#y = torch.randn(N, D_out)

model = TwoLayerNet(D_in, H, D_out)
model.cuda("cuda:0")
criterion = torch.nn.MSELoss(reduction='sum').cuda("cuda:0")
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

for t in range(10000):
    y_pred = model(x).cuda("cuda:0")
    #model.cuda("cuda:0")
    loss = criterion(y_pred, y).cuda("cuda:0")
    if t % 100 > 97:
        print(t, loss.item())
    optimizer.zero_grad
    loss.backward()
    optimizer.step()

stop = timeit.default_timer()
total_time = stop - start

print('Time:', total_time)