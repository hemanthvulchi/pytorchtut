import torch
import os
os.system('clear')
print("###############################")
print("###Requires Grad")
x=torch.ones(2,2,requires_grad=True)
print(x)

print("###tensor ops")
y=x+2
print(y)
print(y.grad_fn)

print("###other operations")
z=y*y*3
out=z.mean()
print(z)
print(out)

print("###Requires grad")
a=torch.randn(2,2)
print("value of a1:",a)
a=((a*3)/(a-1))
print("value of a2:",a)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
print("value of a3:",a)
b=(a*a).sum()
print("Value of b:",b)
print(b.grad_fn)
out.backward()
print(x.grad)