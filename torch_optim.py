import torch
import timeit
if torch.cuda.is_available():
    device = torch.device("cuda")          
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")

start = timeit.default_timer()

dtype = torch.float
N, D_in, H, D_out = 64, 1000, 100, 10

# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)
# model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
#                             torch.nn.ReLU(),
#                             torch.nn.Linear(H, D_out),
#                             ).cuda("cuda:0")

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, D_out),
                            )

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(10000):
    y_pred = model(x)
    #model.cuda("cuda:0")
    loss = loss_fn(y_pred, y)
    if t % 100 > 97:
        print(t, loss.item())
    optimizer.zero_grad
    loss.backward()
    optimizer.step()

stop = timeit.default_timer()
total_time = stop - start

print('Time:', total_time)