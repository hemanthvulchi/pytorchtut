from __future__ import print_function
import torch
x=torch.empty(5,3)
print(x)
x = torch.rand(5, 3)
print(x)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
x = torch.tensor([5.5,3])
print(x)
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
#print(x)
#x = torch.randn_like(x, dtype=torch.float)    # override dtype!
#print(x) 
#print(x.size())
y=x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(y)
print(torch.add(x,y))
result=torch.empty(5,3)
torch.add(x,y, out=result)
print(result)
y.add_(x)

print(y)
print(y[:, 1])

print("Resizing")
x=torch.randn(4,4)
y= x.view(16)
z=x.view(8,-1)
print(x.size(),y.size(),z.size())

# print("item")
# x=torch.randn(1)
# print(x)
# print(x.item())

print("numpy")
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

print("Cuda Code")
if torch.cuda.is_available():
    device = torch.device("cuda")          
    y = torch.ones_like(x, device=device)  
    x = x.to(device)                       
    z = x + y
    print(z)
    print(z.to(device, torch.double))     
    print(z.to("cpu", torch.double))     
