import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
import sympy as sm
#from scipy.special import erf
from sympy import symbols,diff

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * pow(x,3)))) #sqrt(2/pi) approximation is 0.797885
    # 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

x_values=np.linspace(-6,6,50)
y_values=gelu(x_values)

plt.plot(x_values,y_values,color='indigo',label="Gelu(x) in the preceding layers of the transformer", marker='o')
plt.ylabel('Output(Gelu(x)) at the layer 1 in the preceding layer')
plt.xlabel('Input(x)')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='red',linestyle='--')
plt.legend()
plt.grid(True)
print(y_values)
plt.show()

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * pow(x,3)))) #sqrt(2/pi) approximation is 0.797885
    # 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

x_values=np.linspace(-6,6,50)
y_values=gelu(x_values)
plt.figure(figsize=(8, 5))
plt.plot(x_values,y_values,color='indigo',label="Gelu(x) in the preceding layers of the transformer", marker='o')
plt.ylabel('Output(Gelu(x)) at the layer 1 in the preceding layer')
plt.xlabel('Input(x)')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='red',linestyle='--')
plt.legend()
plt.grid(True)
print(y_values)
plt.show()


#from scipy.special import erf

def sinh(x):
    return np.log(np.arcsin(x))#np.sinh(x)

x_values=np.linspace(-6,6,500)
y_values=sinh(x_values)

plt.plot(x_values,y_values,color='indigo',label="Gelu(x) in the preceding layers of the transformer", marker='o')
plt.ylabel('Output(Gelu(x)) at the layer 1 in the preceding layer')
plt.xlabel('Input(x)')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='blue',linestyle='--')
plt.legend()
plt.grid(True)
print(y_values)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# GELU function
# def gelu(x):
#     return x * 0.5 * (1 + np.tanh(1.0 + np.erf(x / np.sqrt(2))))

# # Generate values in the specified range
# x_values = np.linspace(-10, 10, 50)
# gelu_values = gelu(x_values)

# # Plot the GELU function
# plt.plot(x_values, gelu_values, label='GELU(x)', color='blue',marker='o')
# plt.title('GELU Function')
# plt.axhline(0,color='red',linestyle='--')
# plt.axvline(0,color='green',linestyle='--')
# plt.xlabel('x')
# plt.ylabel('Gelu(x)')
# plt.grid(True)
# plt.legend()
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
import sympy as sm
#from scipy.special import erf
from sympy import symbols,diff

# # Normal    Probability distribution of the inpur tokens in the stream of (4096 - Used Token size)
# mean=0
# variance=1
# std_dev=np.sqrt(variance)

# def gelu(x):
#     return 1/(np.sqrt(2 * np.pi* pow(variance,2))) * (np.exp(-(x - mean)**2) / (2 * variance**2))
# x_values=np.linspace(-6,6)
# y_values=gelu(x_values)

# plt.plot(x_values,y_values,color='indigo',label="Gelu(x) in the preceding layers of the transformer", marker='o')
# plt.ylabel('Output(Gelu(x)) at the layer 1 in the preceding layer')
# plt.xlabel('Input(x)')
# plt.axhline(0,color='red',linestyle='--')
# plt.axvline(0,color='orange',linestyle='--')
# plt.legend()
# plt.grid(True)
# print(y_values)
# plt.show()


#Computation in the case of Normal distribution
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.special as sp
import sympy as sm
#from scipy.special import erf
from sympy import symbols,diff
mean=0
variance=1
std_dev=np.sqrt(variance)

def Norm_Dist(x):
    return 1/(np.sqrt(2 * np.pi * pow(variance,2))) * ((-(x - pow(mean,2))) / (2 * pow(variance,2)))
x_values=np.linspace(-6,6,1000)
y_values=Norm_Dist(x_values)
plt.plot(x_values,y_values,color='indigo',label='Function at the 5th transformer ')
plt.xlabel('Input(x)')
plt.ylabel('Output(x) Normal distribution at the end of the 5th Transformer layer') 
plt.axhline(0,color='red', linestyle='--')
plt.axvline(0,color='green', linestyle='--')

def leaky_relu(x,alpha=0.1):
    return np.where(x > 0 , x ,alpha * x )

x_values=np.linspace(-8 , 8 ,100)
y_values=leaky_relu(x_values)

plt.plot(x_values,y_values,marker='o',color='maroon',label='Function at the output layers of the transformer block')
plt.ylabel('Output(Leaky_relu(x))')
plt.xlabel('Input(x)')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='blue',linestyle='--')
plt.grid(True)
plt.legend()
print(y_values)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the Leaky ReLU activation function
def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

# Generate a range of x values
x_values = np.linspace(-6, 6, 100)
y_values = leaky_relu(x_values)

# Plot the Leaky ReLU activation function
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label=r'Leaky ReLU ($\alpha=0.01$)', color='purple')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='blue',linestyle='--')
plt.title("Leaky ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("Leaky ReLU(x)")
plt.grid(True)
plt.legend()
plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


#Define the SELU Activation Function
# class SELU(nn.Module):
#     def __init__(self):
#         super(SELU, self).__init__()
#         self.alpha = 1.67326
#         self.scale = 1.0507

#     def forward(self, x):
#         return self.scale * torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

# # Custom Transformer Block with SELU
# class CustomTransformerLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
#         super(CustomTransformerLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, 8, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.selu = SELU()  # Using custom SELU activation
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, src):
#         # Self-attention part
#         src2 = self.self_attn(src,src,src)[0]
#         src = src + self.dropout(src2)
#         src = self.norm1(src)
        
#         # Feedforward part with SELU activation
#         src2 = self.linear2(self.selu(self.linear1(src)))
#         src = src + self.dropout(src2)
#         src = self.norm2(src)
#         return src
# d_model = pow(2,11)  #2048 
# num_heads = pow(2,13) #8192
# dim_feedforward = pow(2,15) #16384

# transformer_layer = CustomTransformerLayer(d_model, num_heads, dim_feedforward)
# x = torch.randn(10, 32, d_model)  
# output = transformer_layer(x)
# print(output.shape)  


import scipy.special as sp
import sympy as sm
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols,diff

x=sm.symbols('x')
f_x=sm.exp(sm.sinh(x)) 
d_f_x=sm.diff(f_x)
d_d_f_x=sm.diff(d_f_x)
print(f"Input Function in the Layer-1 of the transformer: {d_f_x}") #exp(sinh(x))*cosh(x)
print(f"Output Function in the Layer-1 of the transformer: {d_d_f_x}") #exp(sinh(x))*sinh(x) + exp(sinh(x))*cosh(x)**2

y=sm.symbols('y')
f_y=sm.log(sm.asin(y))
d_f_y=sm.diff(f_y)
d_d_f_y=sm.diff(d_f_y)
print(f"Input Function in the Layer-2 of the transformer: {d_f_y}")#1/(sqrt(1 - y**2)*asin(y))
print(f"Output Function in the Layer-2 of the transformer: {d_d_f_y}")#y/((1 - y**2)**(3/2)*asin(y)) - 1/((1 - y**2)*asin(y)**2)

s=sm.symbols('s')
f_s=sm.exp(sm.cos(s))
d_f_s=sm.diff(f_s)
d_d_f_s=sm.diff(d_f_s)
print(f"Input Function in the Layer-3 of the transformer: {d_f_s}")#-exp(cos(s))*sin(s)
print(f"Output Function in the Layer-3 of the transformer: {d_d_f_s}")#exp(cos(s))*sin(s)**2 - exp(cos(s))*cos(s)

p=sm.symbols('p')
f_p=sm.log(sm.sinh(p))
d_f_p=sm.diff(f_p)
d_d_f_p=sm.diff(d_f_p)
print(f"Input Function in the Layer-4 of the transformer: {d_f_p}")#cosh(p)/sinh(p)
print(f"Output Function in the Layer-4 of the transformer: {d_d_f_p}")#1 - cosh(p)**2/sinh(p)**2

m=sm.symbols('m')
f_m=(sm.atan(m)) 
d_f_m=sm.diff(f_m)
d_d_f_m=sm.diff(d_f_m)
print(f"Input Function in the Layer-5 of the transformer: {d_f_m}")# 1/(m**2 + 1)
print(f"Output Function in the Layer-5 of the transformer: {d_d_f_m}")# -2*m/(m**2 + 1)**2

u=sm.symbols('u')
f_u=sm.log(sm.acos(u)) 
d_f_u=sm.diff(f_u)
d_d_f_u=sm.diff(d_f_u)
print(f"Input Function in the Layer-6 of the transformer: {d_f_u}")# -1/(sqrt(1 - u**2)*acos(u))
print(f"Output Function in the Layer-6 of the transformer: {d_d_f_u}")# -u/((1 - u**2)**(3/2)*acos(u)) - 1/((1 - u**2)*acos(u)**2)

a=sm.symbols('a')                                               
f_a=sm.exp(sm.asin(a))
d_f_a=sm.diff(f_a)
d_d_f_a=sm.diff(d_f_a)
print(f"Input Function in the Layer-7 of the transformer: {d_f_a}")# exp(asin(a))/sqrt(1 - a**2)
print(f"Output Function in the Layer-7 of the transformer: {d_d_f_a}")#a*exp(asin(a))/(1 - a**2)**(3/2) + exp(asin(a))/(1 - a**2)

b=symbols('b')
f_b=sm.atan(b)
d_f_b=sm.diff(f_b)
d_d_f_b=sm.diff(d_f_b)
print(f"Input Function in the Layer-8 of the transformer: {d_f_b}")# 1/(b**2 + 1)
print(f"Output Function in the Layer-8 of the transformer: {d_d_f_b}")# -2*b/(b**2 + 1)**2



m=sm.symbols('m')                                               
f_m=(sm.atan(m)) 
d_f_m=sm.diff(f_m)
d_d_f_m=sm.diff(d_f_m)
print(f"Input Function in the Layer-9 of the transformer: {d_f_m}")# 1/(m**2 + 1)
print(f"Output Function in the Layer-9 of the transformer: {d_d_f_m}")# -2*m/(m**2 + 1)**2

n=symbols('n')
f_n=sm.acos(n) 
d_f_n=sm.diff(f_n)
d_d_f_n=sm.diff(d_f_n)
print(f"Input Function in the Layer-10 of the transformer: {d_f_n}")# -1/sqrt(1 - n**2)
print(f"Output Function in the Layer-10 of the transformer: {d_d_f_n}") # -n/(1 - n**2)**(3/2)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sp
import sympy as sm
from sympy import symbols,diff

#from scipy.special import erf

def Mixed_f(x):
    return np.arcsin(x) + np.arccos(x) + np.arctan(x) -(np.arccos(x))

x_values=np.linspace(-6,6,100)
y_values=Mixed_f(x_values)

plt.plot(x_values,y_values,color='orange',label='Mixed_Function at the level-2 transformer layer', marker='o')
plt.xlabel('Input(x)')
plt.ylabel('Mixed_f(x) at the Transformer level-2')
plt.legend()
plt.grid(True)
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='green',linestyle='--')
print(y_values)
plt.show()


# def erf(x):
#     return erf(x)

# x_values=np.linspace(-6,6,100)
# y_values=erf(x_values)

# plt.plot(x_values,y_values,color='orange',label='Accuracy function at the pushed transformer layer-2')
# plt.xlabel('Input(x)')
# plt.ylabel('Accuracy function Transformer level-2')
# plt.legend()
# plt.grid(True)
# plt.axhline(0,color='red',linestyle='--')
# plt.axvline(0,color='green',linestyle='--')
# print(y_values)
# plt.show()

# import scipy.special as sp
# import matplotlib.pyplot as plt
# from scipy.special import erf
# import numpy as np
# import sympy as sm
# x_values=range(-40, 40 ,1)
# erf_x=sp.erf(x_values)
# plt.plot(x_values,erf_x,color='red',label='Accuracy Function at the pushed transformer layer-2', marker='o')
# plt.xlabel('Input(x)')
# plt.ylabel('Accuracy function Transformer level-2')
# plt.legend()
# plt.grid(True)
# plt.axhline(0,color='blue',linestyle='--')
# plt.axvline(0,color='green',linestyle='--')
# print(erf_x)
# plt.show()


def leaky_relu(x,alpha=0.01):
    return np.where(x > 0, x , alpha * x)

x_values=np.linspace(-8,8,100)
y_values=leaky_relu(x_values)

plt.plot(x_values,y_values, color='magenta', label='Function at the final layer of the transformer', marker='o')
plt.xlabel('Input(x)')
plt.ylabel('Output at the end of the transformer layer')
plt.axhline(0,color='blue', linestyle='--')
plt.axvline(0,color='green', linestyle='--')
plt.legend()
plt.grid(True)
print(f'The augumented values at the end of the transformer layer are:{y_values}')

def sinh(x):
    return np.sinh(x)

x_values=np.linspace(-6,6,50)
y_values=np.sinh(x_values)

plt.plot(x_values,y_values, color='maroon' , label='Transformer output layer')
plt.xlabel('Input(x)')
plt.ylabel('Output at the end of the transformer layer( inverse of the sine function)')
plt.axhline(0,color='blue',linestyle='--')
plt.axvline(0,color='green',linestyle='--')
plt.grid(True)
plt.legend()
print(f"Values at the level 2 activation function are : {y_values}")
plt.show()

def selu(x,alpha=1.6732 ):
    return np.where(x > 0, 1.0507 * x, alpha * 1.0507 * (np.exp(x) - 1) )

x_values=np.linspace(-40, 40)
y_values=selu(x_values)

plt.plot(x_values,y_values, color='Maroon', label='Function at the end of the 4th Transformer layer', marker='o')
plt.xlabel('Input(x)')
plt.ylabel('Output Function selu(x) ath the 4th Transformer layer')
plt.grid(True)
plt.legend()
plt.axhline(0, color='Blue' , linestyle='--')
plt.axvline(0, color='Green' , linestyle='--')
print(f"The value of the selu function in the 4th transformer layer is: {y_values}")
plt.show()

import sympy as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
#from scipy.special import erf
from sympy import symbols,diff

x=sm.symbols('x')
g_x = 0.5 * x * (1 + sm.tanh(sm.sqrt(2/sm.pi)) * (x + 0.044715 ** 3))
d_g_x=sm.diff(g_x)
print(f'Output Function at the end of the domain: {d_g_x} ')

def d_g_x(x):
    return 0.5 * x * np.tanh(np.sqrt(2)/np.sqrt(np.pi)) + 0.5*(x + 8.9404567225875e-5)*np.tanh(np.sqrt(2)/np.sqrt(np.pi)) + 0.5

x_values=np.linspace(-8, 8, 50)
y_values=d_g_x(x_values)

plt.plot(x_values,y_values, color= 'green', label='Function used in the backpropagation')
plt.xlabel('Input(x)')
plt.ylabel('Output function during the backpropagation')
plt.grid(True)
plt.legend()
plt.axhline(0, color='Black' , linestyle='--')
plt.axvline(0, color='Red' , linestyle='--')
print(f'Function values at the augmemted domain: {y_values}')
plt.show()


def selu(x,alpha= 0.1 ):
    return np.where(x > 0, x, x*(1 + (np.sqrt(1 + alpha * pow(x,2)))))

x_values=np.linspace(-40, 40)
y_values=selu(x_values)

plt.plot(x_values,y_values, color='Maroon', label='Function at the end of the 4th Transformer layer', marker='o')
plt.xlabel('Input(x)')
plt.ylabel('Output Function selu(x) ath the 4th Transformer layer')
plt.grid(True)
plt.legend()
plt.axhline(0, color='Blue' , linestyle='--')
plt.axvline(0, color='Green' , linestyle='--')
print(f"The value of the selu function in the 4th transformer layer is: {y_values}")
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sp
#from scipy.special import erf
import sympy as sm
from sympy import diff,symbols

#Transformer functions in the 2nd Dimension for the earlier modelled 12 layers

x=sm.symbols('x')
f_x=sm.exp(sm.sinh(x)) + sm.tanh(x**2)
d_f_x=sm.diff(f_x)#2*x*(1 - tanh(x**2)**2) + exp(sinh(x))*cosh(x)
d_d_f_x=sm.diff(d_f_x)#-8*x**2*(1 - tanh(x**2)**2)*tanh(x**2) + exp(sinh(x))*sinh(x) + exp(sinh(x))*cosh(x)**2 - 2*tanh(x**2)**2 + 2
print(f'(2nd Dimension, Layer 1,Input Function: {d_f_x}')
print(f'(2nd Dimension, Layer 1,Output Function: {d_d_f_x}')


y=sm.symbols('y')
f_y=1/(sm.sqrt(1 - pow(y,2)) * sm.asin(y))#
d_f_y=sm.diff(f_y)#y/((1 - y**2)**(3/2)*asin(y)) - 1/((1 - y**2)*asin(y)**2)
d_d_f_y=sm.diff(d_f_y)#3*y**2/((1 - y**2)**(5/2)*asin(y)) - 3*y/((1 - y**2)**2*asin(y)**2) + 1/((1 - y**2)**(3/2)*asin(y)) + 2/((1 - y**2)**(3/2)*asin(y)**3)
print(f'(2nd Dimension, Layer 2,Input Function: {d_f_y}')
print(f'(2nd Dimension, Layer 2,Output Function: {d_d_f_y}')

z=sm.symbols('z')
f_z=sm.exp(sm.cos(z)) * sm.sin(z)
d_f_z=sm.diff(f_z)#-exp(cos(z))*sin(z)**2 + exp(cos(z))*cos(z)
d_d_f_z=sm.diff(d_f_z)# exp(cos(z))*sin(z)**3 - 3*exp(cos(z))*sin(z)*cos(z) - exp(cos(z))*sin(z)
print(f'(2nd Dimension, Layer 3,Input Function: {d_f_z}')
print(f'(2nd Dimension, Layer 3,Output Function: {d_d_f_z}')

p=sm.symbols('p')
f_p=sm.cosh(p)/sm.sinh(p)
d_f_p=sm.diff(f_p)# 1 - cosh(p)**2/sinh(p)**2
d_d_f_p=sm.diff(d_f_p)# -2*cosh(p)/sinh(p) + 2*cosh(p)**3/sinh(p)**3
print(f'(2nd Dimension, Layer 4,Input Function: {d_f_p}')
print(f'(2nd Dimension, Layer 4,Output Function: {d_d_f_p}')

r=sm.symbols('r')
f_r=sm.cosh(r)/sm.sinh(r)
d_f_r=sm.diff(f_r)# 1 - cosh(r)**2/sinh(r)**2
d_d_f_r=sm.diff(d_f_r)# -2*cosh(r)/sinh(r) + 2*cosh(r)**3/sinh(r)**3
print(f'(2nd Dimension, Layer 5,Input Function: {d_f_r}')
print(f'(2nd Dimension, Layer 5,Output Function: {d_d_f_r}')

s=sm.symbols('s')
f_s=1/(s**2 + 1)
d_f_s=sm.diff(f_s)# -2*s/(s**2 + 1)**2 
d_d_f_s=sm.diff(d_f_s)# 8*s**2/(s**2 + 1)**3 - 2/(s**2 + 1)**2
print(f'(2nd Dimension, Layer 6,Input Function: {d_f_s}')
print(f'(2nd Dimension, Layer 6,Output Function: {d_d_f_s}')

d=sm.symbols('d')
f_d=1/(d**2 + 1)
d_f_d=sm.diff(f_d)# -2*s/(s**2 + 1)**2
d_d_f_d=sm.diff(d_f_d)# 8*s**2/(s**2 + 1)**3 - 2/(s**2 + 1)**2
print(f'(2nd Dimension, Layer 7,Input Function: {d_f_s}')
print(f'(2nd Dimension, Layer 7,Output Function: {d_d_f_s}')

e=sm.symbols('e')
f_e= -2*e/(e**2 + 1)**2
d_f_e=sm.diff(f_e)# 8*e**2/(e**2 + 1)**3 - 2/(e**2 + 1)**2
d_d_f_e=sm.diff(d_f_e)# -48*e**3/(e**2 + 1)**4 + 24*e/(e**2 + 1)**3
print(f'(2nd Dimension, Layer 8,Input Function: {d_f_e}')
print(f'(2nd Dimension, Layer 8,Output Function: {d_d_f_e}')

f=sm.symbols('f')
f_f= -1/sm.sqrt(1 - f**2)
d_f_f=sm.diff(f_f)# -f/(1 - f**2)**(3/2)
d_d_f_f=sm.diff(d_f_f)# -3*f**2/(1 - f**2)**(5/2) - 1/(1 - f**2)**(3/2)
print(f'(2nd Dimension, Layer 9,Input Function: {d_f_f}')
print(f'(2nd Dimension, Layer 9,Output Function: {d_d_f_f}')

m=sm.symbols('m')
f_m= 1 - sm.cosh(m)**2/sm.sinh(m)**2
d_f_m=sm.diff(f_m)# -2*cosh(m)/sinh(m) + 2*cosh(m)**3/sinh(m)**3
d_d_f_m=sm.diff(d_f_m)# -2 + 8*cosh(m)**2/sinh(m)**2 - 6*cosh(m)**4/sinh(m)**4
print(f'(2nd Dimension, Layer 10,Input Function: {d_f_m}')
print(f'(2nd Dimension, Layer 10,Output Function: {d_d_f_m}')


n=sm.symbols('n')
f_n= 1 - sm.cosh(n)**2/sm.sinh(n)**2
d_f_n=sm.diff(f_n)# -2*cosh(n)/sinh(n) + 2*cosh(n)**3/sinh(n)**3
d_d_f_n=sm.diff(d_f_n)# -2 + 8*cosh(n)**2/sinh(n)**2 - 6*cosh(n)**4/sinh(n)**4
print(f'(2nd Dimension, Layer 11,Input Function: {d_f_n}')
print(f'(2nd Dimension, Layer 11,Output Function: {d_d_f_n}')

o=sm.symbols('o')
f_o= -o/((1 - o**2)**(3/2) * sm.acos(o)) - 1/((1 - o**2) * sm.acos(o)**2)
d_f_o=sm.diff(f_o)# -3.0*o**2/((1 - o**2)**2.5*acos(o)) - o/((1 - o**2)**2.0*acos(o)**2) - 2*o/((1 - o**2)**2*acos(o)**2) - 1/((1 - o**2)**1.5*acos(o)) - 2/((1 - o**2)**(3/2)*acos(o)**3) 
d_d_f_o=sm.diff(d_f_o)# -15.0*o**3/((1 - o**2)**3.5*acos(o)) - 7.0*o**2/((1 - o**2)**3.0*acos(o)**2) - 8*o**2/((1 - o**2)**3*acos(o)**2) - 9.0*o/((1 - o**2)**2.5*acos(o)) - 2*o/((1 - o**2)**2.5*acos(o)**3) - 10*o/((1 - o**2)**(5/2)*acos(o)**3) - 2/((1 - o**2)**2.0*acos(o)**2) - 2/((1 - o**2)**2*acos(o)**2) - 6/((1 - o**2)**2*acos(o)**4)
print(f'(2nd Dimension, Layer 12,Input Function: {d_f_o}')
print(f'(2nd Dimension, Layer 12,Output Function: {d_d_f_o}')

j=sm.symbols('j')
f_j= -j/((1 - j**2)**(3/2) * sm.acos(j)) - 1/((1 - j**2) * sm.acos(j)**2)
d_f_j=sm.diff(f_j)# -3.0*j**2/((1 - j**2)**2.5*acos(j)) - j/((1 - j**2)**2.0*acos(j)**2) - 2*j/((1 - j**2)**2*acos(j)**2) - 1/((1 - j**2)**1.5*acos(j)) - 2/((1 - j**2)**(3/2)*acos(j)**3) 
d_d_f_j=sm.diff(d_f_j)# -15.0*j**3/((1 - j**2)**3.5*acos(j)) - 7.0*j**2/((1 - j**2)**3.0*acos(j)**2) - 8*j**2/((1 - j**2)**3*acos(j)**2) - 9.0*j/((1 - j**2)**2.5*acos(j)) - 2*j/((1 - j**2)**2.5*acos(j)**3) - 10*j/((1 - j**2)**(5/2)*acos(j)**3) - 2/((1 - j**2)**2.0*acos(j)**2) - 2/((1 - j**2)**2*acos(j)**2) - 6/((1 - j**2)**2*acos(j)**4)
print(f'(2nd Dimension, Layer 13,Input Function: {d_f_j}')
print(f'(2nd Dimension, Layer 13,Output Function: {d_d_f_j}')

k=sm.symbols('k')
f_k= -k/((1 - k**2)**(3/2) * sm.acos(k)) - 1/((1 - k**2) * sm.acos(k)**2)
d_f_k=sm.diff(f_k)# -3.0*k**2/((1 - k**2)**2.5*acos(k)) - k/((1 - k**2)**2.0*acos(k)**2) - 2*k/((1 - k**2)**2*acos(k)**2) - 1/((1 - k**2)**1.5*acos(k)) - 2/((1 - k**2)**(3/2)*acos(k)**3)
d_d_f_k=sm.diff(d_f_k)# -15.0*k**3/((1 - k**2)**3.5*acos(k)) - 7.0*k**2/((1 - k**2)**3.0*acos(k)**2) - 8*k**2/((1 - k**2)**3*acos(k)**2) - 9.0*k/((1 - k**2)**2.5*acos(k)) - 2*k/((1 - k**2)**2.5*acos(k)**3) - 10*k/((1 - k**2)**(5/2)*acos(k)**3) - 2/((1 - k**2)**2.0*acos(k)**2) - 2/((1 - k**2)**2*acos(k)**2) - 6/((1 - k**2)**2*acos(k)**4)
print(f'(2nd Dimension, Layer 15,Input Function: {d_f_k}')
print(f'(2nd Dimension, Layer 15,Output Function: {d_d_f_k}')

l=sm.symbols('l')
f_l= l * sm.exp(sm.asin(l))/(1 - l**2)**(3/2) + sm.exp(sm.asin(l))/(1 - l**2)
d_f_l=sm.diff(f_l)# 3.0*l**2*exp(asin(l))/(1 - l**2)**2.5 + l*exp(asin(l))/(1 - l**2)**2.0 + 2*l*exp(asin(l))/(1 - l**2)**2 + exp(asin(l))/(1 - l**2)**1.5 + exp(asin(l))/(1 - l**2)**(3/2)
d_d_f_l=sm.diff(d_f_l)# 15.0*l**3*exp(asin(l))/(1 - l**2)**3.5 + 7.0*l**2*exp(asin(l))/(1 - l**2)**3.0 + 8*l**2*exp(asin(l))/(1 - l**2)**3 + 10.0*l*exp(asin(l))/(1 - l**2)**2.5 + 5*l*exp(asin(l))/(1 - l**2)**(5/2) + 2*exp(asin(l))/(1 - l**2)**2.0 + 3*exp(asin(l))/(1 - l**2)**2
print(f'(2nd Dimension, Layer 16,Input Function: {d_f_l}')
print(f'(2nd Dimension, Layer 16,Output Function: {d_d_f_l}')


import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sp
#from scipy.special import erf
import sympy as sm
from sympy import diff,symbols

#Transformer functions in the 2nd Dimension for the earlier modelled 12 layers

x=sm.symbols('x')
f_x=sm.exp(sm.sinh(x)) + sm.tanh(x**2)
d_f_x=sm.diff(f_x)#2*x*(1 - tanh(x**2)**2) + exp(sinh(x))*cosh(x)
d_d_f_x=sm.diff(d_f_x)#-8*x**2*(1 - tanh(x**2)**2)*tanh(x**2) + exp(sinh(x))*sinh(x) + exp(sinh(x))*cosh(x)**2 - 2*tanh(x**2)**2 + 2
print(f'(2nd Dimension, Layer 1,Input Function: {d_f_x}')
print(f'(2nd Dimension, Layer 1,Output Function: {d_d_f_x}')


y=sm.symbols('y')
f_y=1/(sm.sqrt(1 - pow(y,2)) * sm.asin(y))#
d_f_y=sm.diff(f_y)#y/((1 - y**2)**(3/2)*asin(y)) - 1/((1 - y**2)*asin(y)**2)
d_d_f_y=sm.diff(d_f_y)#3*y**2/((1 - y**2)**(5/2)*asin(y)) - 3*y/((1 - y**2)**2*asin(y)**2) + 1/((1 - y**2)**(3/2)*asin(y)) + 2/((1 - y**2)**(3/2)*asin(y)**3)
print(f'(2nd Dimension, Layer 2,Input Function: {d_f_y}')
print(f'(2nd Dimension, Layer 2,Output Function: {d_d_f_y}')

z=sm.symbols('z')
f_z=sm.exp(sm.cos(z)) * sm.sin(z)
d_f_z=sm.diff(f_z)#-exp(cos(z))*sin(z)**2 + exp(cos(z))*cos(z)
d_d_f_z=sm.diff(d_f_z)# exp(cos(z))*sin(z)**3 - 3*exp(cos(z))*sin(z)*cos(z) - exp(cos(z))*sin(z)
print(f'(2nd Dimension, Layer 3,Input Function: {d_f_z}')
print(f'(2nd Dimension, Layer 3,Output Function: {d_d_f_z}')

p=sm.symbols('p')
f_p=sm.cosh(p)/sm.sinh(p)
d_f_p=sm.diff(f_p)# 1 - cosh(p)**2/sinh(p)**2
d_d_f_p=sm.diff(d_f_p)# -2*cosh(p)/sinh(p) + 2*cosh(p)**3/sinh(p)**3
print(f'(2nd Dimension, Layer 4,Input Function: {d_f_p}')
print(f'(2nd Dimension, Layer 4,Output Function: {d_d_f_p}')

r=sm.symbols('r')
f_r=sm.cosh(r)/sm.sinh(r)
d_f_r=sm.diff(f_r)# 1 - cosh(r)**2/sinh(r)**2
d_d_f_r=sm.diff(d_f_r)# -2*cosh(r)/sinh(r) + 2*cosh(r)**3/sinh(r)**3
print(f'(2nd Dimension, Layer 5,Input Function: {d_f_r}')
print(f'(2nd Dimension, Layer 5,Output Function: {d_d_f_r}')

s=sm.symbols('s')
f_s=1/(s**2 + 1)
d_f_s=sm.diff(f_s)# -2*s/(s**2 + 1)**2 
d_d_f_s=sm.diff(d_f_s)# 8*s**2/(s**2 + 1)**3 - 2/(s**2 + 1)**2
print(f'(2nd Dimension, Layer 6,Input Function: {d_f_s}')
print(f'(2nd Dimension, Layer 6,Output Function: {d_d_f_s}')

d=sm.symbols('d')
f_d=1/(d**2 + 1)
d_f_d=sm.diff(f_d)# -2*s/(s**2 + 1)**2
d_d_f_d=sm.diff(d_f_d)# 8*s**2/(s**2 + 1)**3 - 2/(s**2 + 1)**2
print(f'(2nd Dimension, Layer 7,Input Function: {d_f_d}')
print(f'(2nd Dimension, Layer 7,Output Function: {d_d_f_d}')

e=sm.symbols('e')
f_e= -2*e/(e**2 + 1)**2
d_f_e=sm.diff(f_e)# 8*e**2/(e**2 + 1)**3 - 2/(e**2 + 1)**2
d_d_f_e=sm.diff(d_f_e)# -48*e**3/(e**2 + 1)**4 + 24*e/(e**2 + 1)**3
print(f'(2nd Dimension, Layer 8,Input Function: {d_f_e}')
print(f'(2nd Dimension, Layer 8,Output Function: {d_d_f_e}')

f=sm.symbols('f')
f_f= -1/sm.sqrt(1 - f**2)
d_f_f=sm.diff(f_f)# -f/(1 - f**2)**(3/2)
d_d_f_f=sm.diff(d_f_f)# -3*f**2/(1 - f**2)**(5/2) - 1/(1 - f**2)**(3/2)
print(f'(2nd Dimension, Layer 9,Input Function: {d_f_f}')
print(f'(2nd Dimension, Layer 9,Output Function: {d_d_f_f}')

m=sm.symbols('m')
f_m= 1 - sm.cosh(m)**2/sm.sinh(m)**2
d_f_m=sm.diff(f_m)# -2*cosh(m)/sinh(m) + 2*cosh(m)**3/sinh(m)**3
d_d_f_m=sm.diff(d_f_m)# -2 + 8*cosh(m)**2/sinh(m)**2 - 6*cosh(m)**4/sinh(m)**4
print(f'(2nd Dimension, Layer 10,Input Function: {d_f_m}')
print(f'(2nd Dimension, Layer 10,Output Function: {d_d_f_m}')


n=sm.symbols('n')
f_n= 1 - sm.cosh(n)**2/sm.sinh(n)**2
d_f_n=sm.diff(f_n)# -2*cosh(n)/sinh(n) + 2*cosh(n)**3/sinh(n)**3
d_d_f_n=sm.diff(d_f_n)# -2 + 8*cosh(n)**2/sinh(n)**2 - 6*cosh(n)**4/sinh(n)**4
print(f'(2nd Dimension, Layer 11,Input Function: {d_f_n}')
print(f'(2nd Dimension, Layer 11,Output Function: {d_d_f_n}')

o=sm.symbols('o')
f_o= -o/((1 - o**2)**(3/2) * sm.acos(o)) - 1/((1 - o**2) * sm.acos(o)**2)
d_f_o=sm.diff(f_o)# -3.0*o**2/((1 - o**2)**2.5*acos(o)) - o/((1 - o**2)**2.0*acos(o)**2) - 2*o/((1 - o**2)**2*acos(o)**2) - 1/((1 - o**2)**1.5*acos(o)) - 2/((1 - o**2)**(3/2)*acos(o)**3) 
d_d_f_o=sm.diff(d_f_o)# -15.0*o**3/((1 - o**2)**3.5*acos(o)) - 7.0*o**2/((1 - o**2)**3.0*acos(o)**2) - 8*o**2/((1 - o**2)**3*acos(o)**2) - 9.0*o/((1 - o**2)**2.5*acos(o)) - 2*o/((1 - o**2)**2.5*acos(o)**3) - 10*o/((1 - o**2)**(5/2)*acos(o)**3) - 2/((1 - o**2)**2.0*acos(o)**2) - 2/((1 - o**2)**2*acos(o)**2) - 6/((1 - o**2)**2*acos(o)**4)
print(f'(2nd Dimension, Layer 12,Input Function: {d_f_o}')
print(f'(2nd Dimension, Layer 12,Output Function: {d_d_f_o}')

j=sm.symbols('j')
f_j= -j/((1 - j**2)**(3/2) * sm.acos(j)) - 1/((1 - j**2) * sm.acos(j)**2)
d_f_j=sm.diff(f_j)# -3.0*j**2/((1 - j**2)**2.5*acos(j)) - j/((1 - j**2)**2.0*acos(j)**2) - 2*j/((1 - j**2)**2*acos(j)**2) - 1/((1 - j**2)**1.5*acos(j)) - 2/((1 - j**2)**(3/2)*acos(j)**3) 
d_d_f_j=sm.diff(d_f_j)# -15.0*j**3/((1 - j**2)**3.5*acos(j)) - 7.0*j**2/((1 - j**2)**3.0*acos(j)**2) - 8*j**2/((1 - j**2)**3*acos(j)**2) - 9.0*j/((1 - j**2)**2.5*acos(j)) - 2*j/((1 - j**2)**2.5*acos(j)**3) - 10*j/((1 - j**2)**(5/2)*acos(j)**3) - 2/((1 - j**2)**2.0*acos(j)**2) - 2/((1 - j**2)**2*acos(j)**2) - 6/((1 - j**2)**2*acos(j)**4)
print(f'(2nd Dimension, Layer 13,Input Function: {d_f_j}')
print(f'(2nd Dimension, Layer 13,Output Function: {d_d_f_j}')

k=sm.symbols('k')
f_k= -k/((1 - k**2)**(3/2) * sm.acos(k)) - 1/((1 - k**2) * sm.acos(k)**2)
d_f_k=sm.diff(f_k)# -3.0*k**2/((1 - k**2)**2.5*acos(k)) - k/((1 - k**2)**2.0*acos(k)**2) - 2*k/((1 - k**2)**2*acos(k)**2) - 1/((1 - k**2)**1.5*acos(k)) - 2/((1 - k**2)**(3/2)*acos(k)**3)
d_d_f_k=sm.diff(d_f_k)# -15.0*k**3/((1 - k**2)**3.5*acos(k)) - 7.0*k**2/((1 - k**2)**3.0*acos(k)**2) - 8*k**2/((1 - k**2)**3*acos(k)**2) - 9.0*k/((1 - k**2)**2.5*acos(k)) - 2*k/((1 - k**2)**2.5*acos(k)**3) - 10*k/((1 - k**2)**(5/2)*acos(k)**3) - 2/((1 - k**2)**2.0*acos(k)**2) - 2/((1 - k**2)**2*acos(k)**2) - 6/((1 - k**2)**2*acos(k)**4)
print(f'(2nd Dimension, Layer 14,Input Function: {d_f_k}')
print(f'(2nd Dimension, Layer 14,Output Function: {d_d_f_k}')

l=sm.symbols('l')
f_l= l * sm.exp(sm.asin(l))/(1 - l**2)**(3/2) + sm.exp(sm.asin(l))/(1 - l**2)
d_f_l=sm.diff(f_l)# 3.0*l**2*exp(asin(l))/(1 - l**2)**2.5 + l*exp(asin(l))/(1 - l**2)**2.0 + 2*l*exp(asin(l))/(1 - l**2)**2 + exp(asin(l))/(1 - l**2)**1.5 + exp(asin(l))/(1 - l**2)**(3/2)
d_d_f_l=sm.diff(d_f_l)# /(1 - l**2)**3 + 10.0*l*exp(asin(l))/(1 - l**2)**2.5 + 5*l*exp(asin(l))/(1 - l**2)**(5/2) + 2*exp(asin(l))/(1 - l**2)**2.0 + 3*exp(asin(l))/(1 - l**2)**2
print(f'(2nd Dimension, Layer 15,Input Function: {d_f_l}')
print(f'(2nd Dimension, Layer 15,Output Function: {d_d_f_l}')


import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sp
#from scipy.special import erf
import sympy as sm
from sympy import diff,symbols

#Transformer functions in the 2nd Dimension for the earlier modelled 12 layers

x=sm.symbols('x')
f_x=sm.exp(sm.sinh(x)) + sm.tanh(x**2)
d_f_x=sm.diff(f_x)#2*x*(1 - tanh(x**2)**2) + exp(sinh(x))*cosh(x)
d_d_f_x=sm.diff(d_f_x)#-8*x**2*(1 - tanh(x**2)**2)*tanh(x**2) + exp(sinh(x))*sinh(x) + exp(sinh(x))*cosh(x)**2 - 2*tanh(x**2)**2 + 2
print(f'(2nd Dimension, Layer 1,Input Function: {d_f_x}')
print(f'(2nd Dimension, Layer 1,Output Function: {d_d_f_x}')


y=sm.symbols('y')
f_y=1/(sm.sqrt(1 - pow(y,2)) * sm.asin(y))#
d_f_y=sm.diff(f_y)#y/((1 - y**2)**(3/2)*asin(y)) - 1/((1 - y**2)*asin(y)**2)
d_d_f_y=sm.diff(d_f_y)#3*y**2/((1 - y**2)**(5/2)*asin(y)) - 3*y/((1 - y**2)**2*asin(y)**2) + 1/((1 - y**2)**(3/2)*asin(y)) + 2/((1 - y**2)**(3/2)*asin(y)**3)
print(f'(2nd Dimension, Layer 2,Input Function: {d_f_y}')
print(f'(2nd Dimension, Layer 2,Output Function: {d_d_f_y}')

z=sm.symbols('z')
f_z=sm.exp(sm.cos(z)) * sm.sin(z)
d_f_z=sm.diff(f_z)#-exp(cos(z))*sin(z)**2 + exp(cos(z))*cos(z)
d_d_f_z=sm.diff(d_f_z)# exp(cos(z))*sin(z)**3 - 3*exp(cos(z))*sin(z)*cos(z) - exp(cos(z))*sin(z)
print(f'(2nd Dimension, Layer 3,Input Function: {d_f_z}')
print(f'(2nd Dimension, Layer 3,Output Function: {d_d_f_z}')

p=sm.symbols('p')
f_p=sm.cosh(p)/sm.sinh(p)
d_f_p=sm.diff(f_p)# 1 - cosh(p)**2/sinh(p)**2
d_d_f_p=sm.diff(d_f_p)# -2*cosh(p)/sinh(p) + 2*cosh(p)**3/sinh(p)**3
print(f'(2nd Dimension, Layer 4,Input Function: {d_f_p}')
print(f'(2nd Dimension, Layer 4,Output Function: {d_d_f_p}')

r=sm.symbols('r')
f_r=sm.cosh(r)/sm.sinh(r)
d_f_r=sm.diff(f_r)# 1 - cosh(r)**2/sinh(r)**2
d_d_f_r=sm.diff(d_f_r)# -2*cosh(r)/sinh(r) + 2*cosh(r)**3/sinh(r)**3
print(f'(2nd Dimension, Layer 5,Input Function: {d_f_r}')
print(f'(2nd Dimension, Layer 5,Output Function: {d_d_f_r}')

s=sm.symbols('s')
f_s=1/(s**2 + 1)
d_f_s=sm.diff(f_s)# -2*s/(s**2 + 1)**2 
d_d_f_s=sm.diff(d_f_s)# 8*s**2/(s**2 + 1)**3 - 2/(s**2 + 1)**2
print(f'(2nd Dimension, Layer 6,Input Function: {d_f_s}')
print(f'(2nd Dimension, Layer 6,Output Function: {d_d_f_s}')

d=sm.symbols('d')
f_d=1/(d**2 + 1)
d_f_d=sm.diff(f_d)# -2*s/(s**2 + 1)**2
d_d_f_d=sm.diff(d_f_d)# 8*s**2/(s**2 + 1)**3 - 2/(s**2 + 1)**2
print(f'(2nd Dimension, Layer 7,Input Function: {d_f_d}')
print(f'(2nd Dimension, Layer 7,Output Function: {d_d_f_d}')

e=sm.symbols('e')
f_e= -2*e/(e**2 + 1)**2
d_f_e=sm.diff(f_e)# 8*e**2/(e**2 + 1)**3 - 2/(e**2 + 1)**2
d_d_f_e=sm.diff(d_f_e)# -48*e**3/(e**2 + 1)**4 + 24*e/(e**2 + 1)**3
print(f'(2nd Dimension, Layer 8,Input Function: {d_f_e}')
print(f'(2nd Dimension, Layer 8,Output Function: {d_d_f_e}')

f=sm.symbols('f')
f_f= -1/sm.sqrt(1 - f**2)
d_f_f=sm.diff(f_f)# -f/(1 - f**2)**(3/2)
d_d_f_f=sm.diff(d_f_f)# -3*f**2/(1 - f**2)**(5/2) - 1/(1 - f**2)**(3/2)
print(f'(2nd Dimension, Layer 9,Input Function: {d_f_f}')
print(f'(2nd Dimension, Layer 9,Output Function: {d_d_f_f}')

m=sm.symbols('m')
f_m= 1 - sm.cosh(m)**2/sm.sinh(m)**2
d_f_m=sm.diff(f_m)# -2*cosh(m)/sinh(m) + 2*cosh(m)**3/sinh(m)**3
d_d_f_m=sm.diff(d_f_m)# -2 + 8*cosh(m)**2/sinh(m)**2 - 6*cosh(m)**4/sinh(m)**4
print(f'(2nd Dimension, Layer 10,Input Function: {d_f_m}')
print(f'(2nd Dimension, Layer 10,Output Function: {d_d_f_m}')


n=sm.symbols('n')
f_n= 1 - sm.cosh(n)**2/sm.sinh(n)**2
d_f_n=sm.diff(f_n)# -2*cosh(n)/sinh(n) + 2*cosh(n)**3/sinh(n)**3
d_d_f_n=sm.diff(d_f_n)# -2 + 8*cosh(n)**2/sinh(n)**2 - 6*cosh(n)**4/sinh(n)**4
print(f'(2nd Dimension, Layer 11,Input Function: {d_f_n}')
print(f'(2nd Dimension, Layer 11,Output Function: {d_d_f_n}')

o=sm.symbols('o')
f_o= -o/((1 - o**2)**(3/2) * sm.acos(o)) - 1/((1 - o**2) * sm.acos(o)**2)
d_f_o=sm.diff(f_o)# -3.0*o**2/((1 - o**2)**2.5*acos(o)) - o/((1 - o**2)**2.0*acos(o)**2) - 2*o/((1 - o**2)**2*acos(o)**2) - 1/((1 - o**2)**1.5*acos(o)) - 2/((1 - o**2)**(3/2)*acos(o)**3) 
d_d_f_o=sm.diff(d_f_o)# -15.0*o**3/((1 - o**2)**3.5*acos(o)) - 7.0*o**2/((1 - o**2)**3.0*acos(o)**2) - 8*o**2/((1 - o**2)**3*acos(o)**2) - 9.0*o/((1 - o**2)**2.5*acos(o)) - 2*o/((1 - o**2)**2.5*acos(o)**3) - 10*o/((1 - o**2)**(5/2)*acos(o)**3) - 2/((1 - o**2)**2.0*acos(o)**2) - 2/((1 - o**2)**2*acos(o)**2) - 6/((1 - o**2)**2*acos(o)**4)
print(f'(2nd Dimension, Layer 12,Input Function: {d_f_o}')
print(f'(2nd Dimension, Layer 12,Output Function: {d_d_f_o}')

j=sm.symbols('j')
f_j= -j/((1 - j**2)**(3/2) * sm.acos(j)) - 1/((1 - j**2) * sm.acos(j)**2)
d_f_j=sm.diff(f_j)# -3.0*j**2/((1 - j**2)**2.5*acos(j)) - j/((1 - j**2)**2.0*acos(j)**2) - 2*j/((1 - j**2)**2*acos(j)**2) - 1/((1 - j**2)**1.5*acos(j)) - 2/((1 - j**2)**(3/2)*acos(j)**3) 
d_d_f_j=sm.diff(d_f_j)# -15.0*j**3/((1 - j**2)**3.5*acos(j)) - 7.0*j**2/((1 - j**2)**3.0*acos(j)**2) - 8*j**2/((1 - j**2)**3*acos(j)**2) - 9.0*j/((1 - j**2)**2.5*acos(j)) - 2*j/((1 - j**2)**2.5*acos(j)**3) - 10*j/((1 - j**2)**(5/2)*acos(j)**3) - 2/((1 - j**2)**2.0*acos(j)**2) - 2/((1 - j**2)**2*acos(j)**2) - 6/((1 - j**2)**2*acos(j)**4)
print(f'(2nd Dimension, Layer 13,Input Function: {d_f_j}')
print(f'(2nd Dimension, Layer 13,Output Function: {d_d_f_j}')

k=sm.symbols('k')
f_k= -k/((1 - k**2)**(3/2) * sm.acos(k)) - 1/((1 - k**2) * sm.acos(k)**2)
d_f_k=sm.diff(f_k)# -3.0*k**2/((1 - k**2)**2.5*acos(k)) - k/((1 - k**2)**2.0*acos(k)**2) - 2*k/((1 - k**2)**2*acos(k)**2) - 1/((1 - k**2)**1.5*acos(k)) - 2/((1 - k**2)**(3/2)*acos(k)**3)
d_d_f_k=sm.diff(d_f_k)# -15.0*k**3/((1 - k**2)**3.5*acos(k)) - 7.0*k**2/((1 - k**2)**3.0*acos(k)**2) - 8*k**2/((1 - k**2)**3*acos(k)**2) - 9.0*k/((1 - k**2)**2.5*acos(k)) - 2*k/((1 - k**2)**2.5*acos(k)**3) - 10*k/((1 - k**2)**(5/2)*acos(k)**3) - 2/((1 - k**2)**2.0*acos(k)**2) - 2/((1 - k**2)**2*acos(k)**2) - 6/((1 - k**2)**2*acos(k)**4)
print(f'(2nd Dimension, Layer 14,Input Function: {d_f_k}')
print(f'(2nd Dimension, Layer 14,Output Function: {d_d_f_k}')

l=sm.symbols('l')
f_l= l * sm.exp(sm.asin(l))/(1 - l**2)**(3/2) + sm.exp(sm.asin(l))/(1 - l**2)
d_f_l=sm.diff(f_l)# 3.0*l**2*exp(asin(l))/(1 - l**2)**2.5 + l*exp(asin(l))/(1 - l**2)**2.0 + 2*l*exp(asin(l))/(1 - l**2)**2 + exp(asin(l))/(1 - l**2)**1.5 + exp(asin(l))/(1 - l**2)**(3/2)
d_d_f_l=sm.diff(d_f_l)# /(1 - l**2)**3 + 10.0*l*exp(asin(l))/(1 - l**2)**2.5 + 5*l*exp(asin(l))/(1 - l**2)**(5/2) + 2*exp(asin(l))/(1 - l**2)**2.0 + 3*exp(asin(l))/(1 - l**2)**2
print(f'(2nd Dimension, Layer 15,Input Function: {d_f_l}')
print(f'(2nd Dimension, Layer 15,Output Function: {d_d_f_l}')

a=sm.symbols('a')
f_a= -2*a/(a**2 + 1)**2
d_f_a=sm.diff(f_a)# 8*a**2/(a**2 + 1)**3 - 2/(a**2 + 1)**2
d_d_f_a=sm.diff(d_f_a)# -48*a**3/(a**2 + 1)**4 + 24*a/(a**2 + 1)**3
print(f'(2nd Dimension, Layer 16,Input Function: {d_f_a}')
print(f'(2nd Dimension, Layer 16,Output Function: {d_d_f_a}')

b=sm.symbols('b')
f_b= -b/(1 - b**2)**(3/2)
d_f_b=sm.diff(f_b)# -3.0*b**2/(1 - b**2)**2.5 - 1/(1 - b**2)**1.5
d_d_f_b=sm.diff(d_f_b)# -15.0*b**3/(1 - b**2)**3.5 - 9.0*b/(1 - b**2)**2.5
print(f'(2nd Dimension, Layer 17,Input Function: {d_f_b}')
print(f'(2nd Dimension, Layer 17,Output Function: {d_d_f_b}')


c=sm.symbols('c')
f_c= sm.exp(sm.sinh(c)) + sm.tanh(c**2)
d_f_c=sm.diff(f_c)# 2*c*(1 - tanh(c**2)**2) + exp(sinh(c))*cosh(c)
d_d_f_c=sm.diff(d_f_c)# -8*c**2*(1 - tanh(c**2)**2)*tanh(c**2) + exp(sinh(c))*sinh(c) + exp(sinh(c))*cosh(c)**2 - 2*tanh(c**2)**2 + 2
print(f'(2nd Dimension, Layer 18,Input Function: {d_f_c}')
print(f'(2nd Dimension, Layer 18,Output Function: {d_d_f_c}')

g=sm.symbols('g')
f_g= 1/(sm.sqrt(1 - pow(g,2)) * sm.asin(g))
d_f_g=sm.diff(f_g)# g/((1 - g**2)**(3/2)*asin(g)) - 1/((1 - g**2)*asin(g)**2)
d_d_f_g=sm.diff(d_f_g)# 3*g**2/((1 - g**2)**(5/2)*asin(g)) - 3*g/((1 - g**2)**2*asin(g)**2) + 1/((1 - g**2)**(3/2)*asin(g)) + 2/((1 - g**2)**(3/2)*asin(g)**3)
print(f'(2nd Dimension, Layer 19,Input Function: {d_f_g}')
print(f'(2nd Dimension, Layer 19,Output Function: {d_d_f_g}')

h=sm.symbols('h')
f_h= -sm.exp(sm.cos(h))*sm.sinh(h)**2 + sm.exp(sm.cos(h)) * sm.cos(h)
d_f_h=sm.diff(f_h)# - exp(cos(z))*cos(z) exp(cos(z))*sin(z)**3 - 3*exp(cos(z))*sin(z)*cos(z) - exp(cos(z))*sin(z)
d_d_f_h=sm.diff(d_f_h)# -exp(cos(z))*sin(z)**4 + 6*exp(cos(z))*sin(z)**2*cos(z) + 4*exp(cos(z))*sin(z)**2 - 3*exp(cos(z))*cos(z)**2
print(f'(2nd Dimension, Layer 20,Input Function: {d_f_h}')
print(f'(2nd Dimension, Layer 20,Output Function: {d_d_f_h}')

i=sm.symbols('i')
f_i= -sm.exp(sm.cos(i))*sm.sinh(i)**2 + sm.exp(sm.cos(i)) * sm.cos(i)
d_f_i=sm.diff(f_i)# -exp(cos(z))*sin(z)**4 + 6*exp(cos(z))*sin(z)**2*cos(z) + 4*exp(cos(z))*sin(z)**2 - 3*exp(cos(z))*cos(z)**2
d_d_f_i=sm.diff(d_f_i)# exp(cos(i))*sin(i)**2*cos(i) - exp(cos(i))*sin(i)**2*sinh(i)**2 + 2*exp(cos(i))*sin(i)**2 + 4*exp(cos(i))*sin(i)*sinh(i)*cosh(i) - exp(cos(i))*cos(i)**2 + exp(cos(i))*cos(i)*sinh(i)**2 - exp(cos(i))*cos(i) - 2*exp(cos(i))*sinh(i)**2 - 2*exp(cos(i))*cosh(i)**2
print(f'(2nd Dimension, Layer 21,Input Function: {d_f_i}')
print(f'(2nd Dimension, Layer 21,Output Function: {d_d_f_i}')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
import sympy as sm
#from scipy.special import erf
from sympy import symbols,diff

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * pow(x,3)))) #sqrt(2/pi) approximation is 0.797885
    # 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

x_values=np.linspace(-6,6,50)
y_values=gelu(x_values)

plt.plot(x_values,y_values,color='indigo',label="Gelu(x) in the preceding layers of the transformer", marker='o')
plt.ylabel('Output(Gelu(x)) at the layer 1 in the preceding layer')
plt.xlabel('Input(x)')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='red',linestyle='--')
plt.legend()
plt.grid(True)
print(y_values)
plt.show()

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * pow(x,3)))) #sqrt(2/pi) approximation is 0.797885
    # 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

x_values=np.linspace(-6,6,50)
y_values=gelu(x_values)
plt.figure(figsize=(8, 5))
plt.plot(x_values,y_values,color='indigo',label="Gelu(x) in the preceding layers of the transformer", marker='o')
plt.ylabel('Output(Gelu(x)) at the layer 1 in the preceding layer')
plt.xlabel('Input(x)')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='red',linestyle='--')
plt.legend()
plt.grid(True)
print(y_values)
plt.show()

#from scipy.special import erf

def sinh(x):
    return np.log(np.arcsin(x))                                   #np.sinh(x)

x_values=np.linspace(-6,6,500)
y_values=sinh(x_values)

plt.plot(x_values,y_values,color='indigo',label="Gelu(x) in the preceding layers of the transformer", marker='o')
plt.ylabel('Output(Gelu(x)) at the layer 1 in the preceding layer')
plt.xlabel('Input(x)')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='blue',linestyle='--')
plt.legend()
plt.grid(True)
print(y_values)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
#from scipy.special import erf

# GELU function
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * pow(x,3))))

# Generate values in the specified range
x_values = np.linspace(-10, 10, 50)
gelu_values = gelu(x_values)

# Plot the GELU function
plt.plot(x_values, gelu_values, label='GELU(x)', color='blue',marker='o')
plt.title('GELU Function')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='green',linestyle='--')
plt.xlabel('x')
plt.ylabel('Gelu(x)')
plt.grid(True)
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
import sympy as sm
#from scipy.special import erf
from sympy import symbols,diff

# Normal    Probability distribution of the inpur tokens in the stream of (4096 - Used Token size)
mean=0
variance=1
std_dev=np.sqrt(variance)

def gelu(x):
    return 1/(np.sqrt(2 * np.pi* pow(variance,2))) * (np.exp(-(x - mean)**2) / (2 * variance**2))
x_values=np.linspace(-6,6)
y_values=gelu(x_values)

plt.plot(x_values,y_values,color='indigo',label="Gelu(x) in the preceding layers of the transformer", marker='o')
plt.ylabel('Output(Gelu(x)) at the layer 1 in the preceding layer')
plt.xlabel('Input(x)')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='orange',linestyle='--')
plt.legend()
plt.grid(True)
print(y_values)
plt.show()


#Computation in the case of Normal distribution
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.special as sp
import sympy as sm
#from scipy.special import erf
from sympy import symbols,diff
mean=0
variance=1
std_dev=np.sqrt(variance)

def Norm_Dist(x):
    return 1/(np.sqrt(2 * np.pi * pow(variance,2))) * ((-(x - pow(mean,2))) / (2 * pow(variance,2)))
x_values=np.linspace(-6,6,1000)
y_values=Norm_Dist(x_values)
plt.plot(x_values,y_values,color='indigo',label='Function at the 5th transformer ')
plt.xlabel('Input(x)')
plt.ylabel('Output(x) Normal distribution at the end of the 5th Transformer layer') 
plt.axhline(0,color='red', linestyle='--')
plt.axvline(0,color='green', linestyle='--')



def leaky_relu(x,alpha=0.1):
    return np.where(x > 0 , x ,alpha * x )

x_values=np.linspace(-8 , 8 ,100)
y_values=leaky_relu(x_values)

plt.plot(x_values,y_values,marker='o',color='maroon',label='Function at the output layers of the transformer block')
plt.ylabel('Output(Leaky_relu(x))')
plt.xlabel('Input(x)')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='blue',linestyle='--')
plt.grid(True)
plt.legend()
print(y_values)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the Leaky ReLU activation function
def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

# Generate a range of x values
x_values = np.linspace(-6, 6, 100)
y_values = leaky_relu(x_values)

# Plot the Leaky ReLU activation function
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label=r'Leaky ReLU ($\alpha=0.01$)', color='purple')
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='blue',linestyle='--')
plt.title("Leaky ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("Leaky ReLU(x)")
plt.grid(True)
plt.legend()
plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


# # Define the SELU Activation Function
# class SELU(nn.Module):
#     def __init__(self):
#         super(SELU, self).__init__()
#         self.alpha = 1.67326
#         self.scale = 1.0507

#     def forward(self, x):
#         return self.scale * torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

# # Custom Transformer Block with SELU
# class CustomTransformerLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
#         super(CustomTransformerLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.selu = SELU()  # Using custom SELU activation
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, src):
#         # Self-attention part
#         src2 = self.self_attn(src,src,src)[0]
#         src = src + self.dropout(src2)
#         src = self.norm1(src)
        
#         # Feedforward part with SELU activation
#         src2 = self.linear2(self.selu(self.linear1(src)))
#         src = src + self.dropout(src2)
#         src = self.norm2(src)
#         return src
# d_model = pow(2,13)  #2048 
# num_heads = pow(2,14) #8192
# dim_feedforward = pow(2,16) #16384

# transformer_layer = CustomTransformerLayer(d_model, num_heads, dim_feedforward)
# x = torch.randn(10, 32, d_model)  
# output = transformer_layer(x)
# print(output.shape)  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sp
import sympy as sm
from sympy import symbols,diff
#from scipy.special import erf

def Mixed_f(x):
    return np.arcsin(x) + np.arccos(x) + np.arctan(x) -(np.arccos(x))

x_values=np.linspace(-6,6,100)
y_values=Mixed_f(x_values)

plt.plot(x_values,y_values,color='orange',label='Mixed_Function at the level-2 transformer layer', marker='o')
plt.xlabel('Input(x)')
plt.ylabel('Mixed_f(x) at the Transformer level-2')
plt.legend()
plt.grid(True)
plt.axhline(0,color='red',linestyle='--')
plt.axvline(0,color='green',linestyle='--')
print(y_values)
plt.show()


# def erf(x):
#     return erf(x)

# x_values=np.linspace(-6,6,100)
# y_values=erf(x_values)

# plt.plot(x_values,y_values,color='orange',label='Accuracy function at the pushed transformer layer-2')
# plt.xlabel('Input(x)')
# plt.ylabel('Accuracy function Transformer level-2')
# plt.legend()
# plt.grid(True)
# plt.axhline(0,color='red',linestyle='--')
# plt.axvline(0,color='green',linestyle='--')
# print(y_values)
# plt.show()


def selu(x,alpha=1.6732 ):
    return np.where(x > 0, 1.0507 * x, alpha * 1.0507 * (np.exp(x) - 1) )

x_values=np.linspace(-40, 40)
y_values=selu(x_values)

plt.plot(x_values,y_values, color='Maroon', label='Function at the end of the 4th Transformer layer', marker='o')
plt.xlabel('Input(x)')
plt.ylabel('Output Function selu(x) ath the 4th Transformer layer')
plt.grid(True)
plt.legend()
plt.axhline(0, color='Blue' , linestyle='--')
plt.axvline(0, color='Green' , linestyle='--')
print(f"The value of the selu function in the 4th transformer layer is: {y_values}")
plt.show


import sympy as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
#from scipy.special import erf
from sympy import symbols,diff

x1=sm.symbols('x1')
f_x1= sm.atan(2 * x1) + sm.cos(x1**2) 
d_f_x1=sm.diff(f_x1)
d_d_f_x1=sm.diff(d_f_x1)
print(f'(3rd Dimension, Layer 1,Input Function: {d_f_x1}')
print(f'(3rd Dimension, Layer 1,Output Function: {d_d_f_x1}')

x2=sm.symbols('x2')
f_x2=sm.exp(3* x) + 4*(x**3)
d_f_x2=sm.diff(f_x2)#12*x**2 + 3*exp(3*x)
d_d_f_x2=sm.diff(d_f_x2)# 24*x + 9*exp(3*x)
print(f'The Input function at the 2nd Block Layer 2: {d_f_x2}')
print(f'The Output function at the 2nd Block Layer 2: {d_d_f_x2}')


x3=sm.symbols('x3')
f_x3=sm.sin(3 * (x3**4)) +sm.log(2*(x3**4)) 
d_f_x3=sm.diff(f_x3) # 12*x3**3*cos(3*x3**4) + 4/x3
d_d_f_x3=sm.diff(d_f_x3)#-144*x3**6*sin(3*x3**4) + 36*x3**2*cos(3*x3**4) - 4/x3**2
print(f'The Input function at the 2nd Block Layer 3: {d_f_x3}')
print(f'The Output function at the 2nd Block Layer 3: {d_d_f_x3}')

x4=sm.symbols('x4')
f_x4=sm.cosh(2*x4**3) +sm.tanh(8*x4**5)
d_f_x4=sm.diff(f_x4)#40*x4**4*(1 - tanh(8*x4**5)**2) + 6*x4**2*sinh(2*x4**3)
d_d_f_x4=sm.diff(d_f_x4)# 
print(f'The Input function at the 2nd Block Layer 4: {d_f_x4}')
print(f'The Output function at the 2nd Block Layer 4: {d_d_f_x4}')

x5=sm.symbols('x5')
f_x5=sm.exp(4*x**4) + sm.log(3*x**3)
d_f_x5=sm.diff(f_x5)# 16*x**3*exp(4*x**4) + 3/x
d_d_f_x5=sm.diff(d_f_x5)# 256*x**6*exp(4*x**4) + 48*x**2*exp(4*x**4) - 3/x**2
print(f'The Input function at the 2nd Block Layer 5: {d_f_x5}')
print(f'The Output function at the 2nd Block Layer 5: {d_d_f_x5}')

x6=sm.symbols('x6')
f_x6=sm.sin(3*x6**2) +sm.tanh(4*x6**5)
d_f_x6=sm.diff(f_x6)# 20*x6**4*(1 - tanh(4*x6**5)**2) + 6*x6*cos(3*x6**2)
d_d_f_x6=sm.diff(d_f_x6)# -800*x6**8*(1 - tanh(4*x6**5)**2)*tanh(4*x6**5) + 80*x6**3*(1 - tanh(4*x6**5)**2) - 36*x6**2*sin(3*x6**2) + 6*cos(3*x6**2)
print(f'The Input function at the 2nd Block Layer 6: {d_f_x6}')
print(f'The Output function at the 2nd Block Layer 6: {d_d_f_x6}')

x7=sm.symbols('x7')
f_x7=sm.exp(x7**4) +sm.cosh(2*x7**5) + sm.log(4*x7**5)
d_f_x7=sm.diff(f_x7)# 10*x7**4*sinh(2*x7**5) + 4*x7**3*exp(x7**4) + 5/x7
d_d_f_x7=sm.diff(d_f_x7)# 10*x7**4*sinh(2*x7**5) + 4*x7**3*exp(x7**4) + 5/x7
print(f'The Input function at the 2nd Block Layer 7: {d_f_x7}')
print(f'The Output function at the 2nd Block Layer 7: {d_d_f_x7}')

x8=sm.symbols('x8')
f_x8=sm.tan(x8**2) +sm.sinh(2*x8)
d_f_x8=sm.diff(f_x8)# 2*x8*(tan(x8**2)**2 + 1) + 2*cosh(2*x8)
d_d_f_x8=sm.diff(d_f_x8)#  8*x8**2*(tan(x8**2)**2 + 1)*tan(x8**2) + 2*tan(x8**2)**2 + 4*sinh(2*x8) + 2
print(f'The Input function at the 2nd Block Layer 8: {d_f_x8}')
print(f'The Output function at the 2nd Block Layer 8: {d_d_f_x8}')

x9=sm.symbols('x9')
f_x9=sm.sin(x9**2) +sm.cosh(2*x9)
d_f_x9=sm.diff(f_x9)# 2*x9*cos(x9**2) + 2*sinh(2*x9)
d_d_f_x9=sm.diff(d_f_x9)# -4*x9**2*sin(x9**2) + 2*cos(x9**2) + 4*cosh(2*x9)
print(f'The Input function at the 2nd Block Layer 9: {d_f_x9}')
print(f'The Output function at the 2nd Block Layer 9: {d_d_f_x9}')

x9=sm.symbols('x9')
f_x9=sm.sin(x9**2) +sm.cosh(2*x9)
d_f_x9=sm.diff(f_x9)# 2*x9*cos(x9**2) + 2*sinh(2*x9)
d_d_f_x9=sm.diff(d_f_x9)# -4*x9**2*sin(x9**2) + 2*cos(x9**2) + 4*cosh(2*x9)
print(f'The Input function at the 2nd Block Layer 9: {d_f_x9}')
print(f'The Output function at the 2nd Block Layer 9: {d_d_f_x9}')

x11=sm.symbols('x11')
f_x11=12*x11**2 + 3*sm.exp(3*x11)
d_f_x11=sm.diff(f_x11)# 24*x11 + 9*exp(3*x11)
d_d_f_x11=sm.diff(d_f_x11)# 27*exp(3*x11) + 24
print(f'The Input function at the 2nd Block Layer 11: {d_f_x11}')
print(f'The Output function at the 2nd Block Layer 11: {d_d_f_x11}')

x12=sm.symbols('x12')
f_x12=sm.sin(x12**2) +sm.cosh(2*x12)
d_f_x12=sm.diff(f_x12)# 2*x12*cos(x12**2) + 2*sinh(2*x12)
d_d_f_x12=sm.diff(d_f_x12)# -4*x12**2*sin(x12**2) + 2*cos(x12**2) + 4*cosh(2*x12)
print(f'The Input function at the 2nd Block Layer 12: {d_f_x12}')
print(f'The Output function at the 2nd Block Layer 12: {d_d_f_x12}')

x13=sm.symbols('x13')
f_x13=12*x3**3*sm.cos(3*x3**4) + 4/x3
d_f_x13=sm.diff(f_x13)# -144*x3**6*sin(3*x3**4) + 36*x3**2*cos(3*x3**4) - 4/x3**2
d_d_f_x13=sm.diff(d_f_x13)# -1728*x3**9*cos(3*x3**4) - 1296*x3**5*sin(3*x3**4) + 72*x3*cos(3*x3**4) + 8/x3**3
print(f'The Input function at the 2nd Block Layer 13: {d_f_x13}')
print(f'The Output function at the 2nd Block Layer 13: {d_d_f_x13}')

x14=sm.symbols('x14')
f_x14=sm.tanh(x14**2 + x14**3) +sm.cosh(2*x14**2 + sm.tan(x14 **2))
d_f_x14=sm.diff(f_x14)# (1 - tanh(x14**3 + x14**2)**2)*(3*x14**2 + 2*x14) + (2*x14*(tan(x14**2)**2 + 1) + 4*x14)*sinh(2*x14**2 + tan(x14**2))
d_d_f_x14=sm.diff(d_f_x14)# (1 - tanh(x14**3 + x14**2)**2)*(6*x14 + 2) - 2*(1 - tanh(x14**3 + x14**2)**2)*(3*x14**2 + 2*x14)**2*tanh(x14**3 + x14**2) + (2*x14*(tan(x14**2)**2 + 1) + 4*x14)**2*cosh(2*x14**2 + tan(x14**2)) + (8*x14**2*(tan(x14**2)**2 + 1)*tan(x14**2) + 2*tan(x14**2)**2 + 6)*sinh(2*x14**2 + tan(x14**2))
print(f'The Input function at the 2nd Block Layer 14: {d_f_x14}')
print(f'The Output function at the 2nd Block Layer 14: {d_d_f_x14}')

x15=sm.symbols('x15')
f_x15=sm.exp(sm.sin(x15**5)) +sm.cosh(2*x15**4) + sm.log(x15**4)
d_f_x15=sm.diff(f_x15)# 15**4*exp(sin(x15**5))*cos(x15**5) + 8*x15**3*sinh(2*x15**4) + 4/x15
d_d_f_x15=sm.diff(d_f_x15)# -25*x15**8*exp(sin(x15**5))*sin(x15**5) + 25*x15**8*exp(sin(x15**5))*cos(x15**5)**2 + 64*x15**6*cosh(2*x15**4) + 20*x15**3*exp(sin(x15**5))*cos(x15**5) + 24*x15**2*sinh(2*x15**4) - 4/x15**2
print(f'The Input function at the 2nd Block Layer 15: {d_f_x15}')
print(f'The Output function at the 2nd Block Layer 15: {d_d_f_x15}')

x16=sm.symbols('x16')
f_x16=10*x16**4*sm.sinh(2*x16**5) + 4*x16**3*sm.exp(x16**4) + 5/x16
d_f_x16=sm.diff(f_x16)# 100*x16**8*cosh(2*x16**5) + 16*x16**6*exp(x16**4) + 40*x16**3*sinh(2*x16**5) + 12*x16**2*exp(x16**4) - 5/x16**2
d_d_f_x16=sm.diff(d_f_x16)# 1000*x16**12*sinh(2*x16**5) + 64*x16**9*exp(x16**4) + 1200*x16**7*cosh(2*x16**5) + 144*x16**5*exp(x16**4) + 120*x16**2*sinh(2*x16**5) + 24*x16*exp(x16**4) + 10/x16**3 
print(f'The Input function at the 2nd Block Layer 16: {d_f_x16}')
print(f'The Output function at the 2nd Block Layer 16: {d_d_f_x16}')

x17=sm.symbols('x17')
f_x17=sm.sinh(2*x17**5) + 4*x17**3*sm.exp(x17**4) + 5/x17
d_f_x17=sm.diff(f_x17)# 16*x17**6*exp(x17**4) + 10*x17**4*cosh(2*x17**5) + 12*x17**2*exp(x17**4) - 5/x17**2
d_d_f_x17=sm.diff(d_f_x17)# 64*x17**9*exp(x17**4) + 100*x17**8*sinh(2*x17**5) + 144*x17**5*exp(x17**4) + 40*x17**3*cosh(2*x17**5) + 24*x17*exp(x17**4) + 10/x17**3
print(f'The Input function at the 2nd Block Layer 17: {d_f_x17}')
print(f'The Output function at the 2nd Block Layer 17: {d_d_f_x17}')

x18=sm.symbols('x18')
f_x18=2*x18*(sm.tan(x18**2)**2 + 1) + 2*sm.cosh(2*x18)
d_f_x18=sm.diff(f_x18)# 8*x18**2*(tan(x18**2)**2 + 1)*tan(x18**2) + 2*tan(x18**2)**2 + 4*sinh(2*x18) + 2
d_d_f_x18=sm.diff(d_f_x18)# 16*x18**3*(tan(x18**2)**2 + 1)**2 + 32*x18**3*(tan(x18**2)**2 + 1)*tan(x18**2)**2 + 24*x18*(tan(x18**2)**2 + 1)*tan(x18**2) + 8*cosh(2*x18)
print(f'The Input function at the 2nd Block Layer 18: {d_f_x18}')
print(f'The Output function at the 2nd Block Layer 18: {d_d_f_x18}')

x19=sm.symbols('x19')
f_x19=2*x19*sm.cos(x19**2) + 2*sm.sinh(2*x19)
d_f_x19=sm.diff(f_x19)# -4*x19**2*sin(x19**2) + 2*cos(x19**2) + 4*cosh(2*x19)
d_d_f_x19=sm.diff(d_f_x19)# -8*x19**3*cos(x19**2) - 12*x19*sin(x19**2) + 8*sinh(2*x19)
print(f'The Input function at the 2nd Block Layer 19: {d_f_x19}')
print(f'The Output function at the 2nd Block Layer 19: {d_d_f_x19}')

x20=sm.symbols('x20')
f_x20=2*x20* sm.cos(x20**2) + 2*sm.sinh(2*20)
d_f_x20=sm.diff(f_x20)# -4*x20**2*sin(x20**2) + 2*cos(x20**2)
d_d_f_x20=sm.diff(d_f_x20)# -8*x20**3*cos(x20**2) - 12*x20*sin(x20**2)
print(f'The Input function at the 2nd Block Layer 20: {d_f_x20}')
print(f'The Output function at the 2nd Block Layer 20: {d_d_f_x20}')

x21=sm.symbols('x21')
f_x21=12*x21**2 + 3*sm.exp(3*x21)
d_f_x21=sm.diff(f_x21)# 24*x21 + 9*exp(3*x21)
d_d_f_x21=sm.diff(d_f_x21)# 27*exp(3*x21) + 24
print(f'The Input function at the 2nd Block Layer 21: {d_f_x21}')
print(f'The Output function at the 2nd Block Layer 21: {d_d_f_x21}')

x22=sm.symbols('x22')
f_x22=2*x22*sm.cos(x22**2) + 2*sm.sinh(2*x22)
d_f_x22=sm.diff(f_x22)# -4*x22**2*sin(x22**2) + 2*cos(x22**2) + 4*cosh(2*x22)
d_d_f_x22=sm.diff(d_f_x22)# -8*x22**3*cos(x22**2) - 12*x22*sin(x22**2) + 8*sinh(2*x22)
print(f'The Input function at the 2nd Block Layer 22: {d_f_x22}')
print(f'The Output function at the 2nd Block Layer 22: {d_d_f_x22}')

x23=sm.symbols('x23')
f_x22=-144*x23**6*sm.sin(3*x23**4) + 36*x23**2*sm.cos(3*x23**4) - 4/x23**2
d_f_x22=sm.diff(f_x22)# -1728*x23**9*cos(3*x23**4) - 1296*x23**5*sin(3*x23**4) + 72*x23*cos(3*x23**4) + 8/x23**3
d_d_f_x22=sm.diff(d_f_x22)# 20736*x23**12*sin(3*x23**4) - 31104*x23**8*cos(3*x23**4) - 7344*x23**4*sin(3*x23**4) + 72*cos(3*x23**4) - 24/x23**4
print(f'The Input function at the 2nd Block Layer 2: {d_f_x22}')
print(f'The Output function at the 2nd Block Layer 2: {d_d_f_x22}')

x24=sm.symbols('x24')
f_x24=(1 - sm.tanh(x24**3 + x24**2)**2)*(3*x24**2 + 2*x24) + (2*x24*(sm.tan(x24**2)**2 + 1) + 4*x24)*sm.sinh(2*x24**2 + sm.tan(x24**2))
d_f_x24=sm.diff(f_x24)# (1 - tanh(x24**3 + x24**2)**2)*(6*x24 + 2) - 2*(1 - tanh(x24**3 + x24**2)**2)*(3*x24**2 + 2*x24)**2*tanh(x24**3 + x24**2) + (2*x24*(tan(x24**2)**2 + 1) + 4*x24)**2*cosh(2*x24**2 + tan(x24**2)) + (8*x24**2*(tan(x24**2)**2 + 1)*tan(x24**2) + 2*tan(x24**2)**2 + 6)*sinh(2*x24**2 + tan(x24**2))
d_d_f_x24=sm.diff(d_f_x24)# -2*(1 - tanh(x24**3 + x24**2)**2)**2*(3*x24**2 + 2*x24)**3 - 2*(1 - tanh(x24**3 + x24**2)**2)*(6*x24 + 2)*(3*x24**2 + 2*x24)*tanh(x24**3 + x24**2) - 2*(1 - tanh(x24**3 + x24**2)**2)*(12*x24 + 4)*(3*x24**2 + 2*x24)*tanh(x24**3 + x24**2) + 4*(1 - tanh(x24**3 + x24**2)**2)*(3*x24**2 + 2*x24)**3*tanh(x24**3 + x24**2)**2 + (2*x24*(tan(x24**2)**2 + 1) + 4*x24)**3*sinh(2*x24**2 + tan(x24**2)) + (2*x24*(tan(x24**2)**2 + 1) + 4*x24)*(8*x24**2*(tan(x24**2)**2 + 1)*tan(x24**2) + 2*tan(x24**2)**2 + 6)*cosh(2*x24**2 + tan(x24**2)) + (2*x24*(tan(x24**2)**2 + 1) + 4*x24)*(16*x24**2*(tan(x24**2)**2 + 1)*tan(x24**2) + 4*tan(x24**2)**2 + 12)*cosh(2*x24**2 + tan(x24**2)) + (16*x24**3*(tan(x24**2)**2 + 1)**2 + 32*x24**3*(tan(x24**2)**2 + 1)*tan(x24**2)**2 + 24*x24*(tan(x24**2)**2 + 1)*tan(x24**2))*sinh(2*x24**2 + tan(x24**2)) - 6*tanh(x24**3 + x24**2)**2 + 6
print(f'The Input function at the 2nd Block Layer 2: {d_f_x24}')
print(f'The Output function at the 2nd Block Layer 2: {d_d_f_x24}')

x25=sm.symbols('x25')
f_x25=15**4*sm.exp(sm.sin(x25**5))*sm.cos(x25**5) + 8*x25**3*sm.sinh(2*x25**4) + 4/x25
d_f_x25=sm.diff(f_x25)# 64*x25**6*cosh(2*x25**4) - 253125*x25**4*exp(sin(x25**5))*sin(x25**5) + 253125*x25**4*exp(sin(x25**5))*cos(x25**5)**2 + 24*x25**2*sinh(2*x25**4) - 4/x25**2
d_d_f_x25=sm.diff(d_f_x25)# 512*x25**9*sinh(2*x25**4) - 3796875*x25**8*exp(sin(x25**5))*sin(x25**5)*cos(x25**5) + 1265625*x25**8*exp(sin(x25**5))*cos(x25**5)**3 - 1265625*x25**8*exp(sin(x25**5))*cos(x25**5) + 576*x25**5*cosh(2*x25**4) - 1012500*x25**3*exp(sin(x25**5))*sin(x25**5) + 1012500*x25**3*exp(sin(x25**5))*cos(x25**5)**2 + 48*x25*sinh(2*x25**4) + 8/x25**3
print(f'The Input function at the 2nd Block Layer 2: {d_f_x25}')
print(f'The Output function at the 2nd Block Layer 2: {d_d_f_x25}')


import sympy as sm  
import scipy.special as sp
from sympy import symbols, diff
from scipy.special import erf, erf_zeros
z1=sm.symbols('z1')
f_z1= 0.5 * z1 * (1 + sm.tanh(sm.sqrt(2 / sm.pi) * (z1 + 0.044715 * z1**3)))
d_f_z1=sm.diff(f_z1)# 0.5*sqrt(2)*z1*(1 - tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi))**2)*(0.134145*z1**2 + 1)/sqrt(pi) + 0.5*tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi)) + 0.5 
d_d_f_z1=sm.diff(d_f_z1)# 0.134145*sqrt(2)*z1**2*(1 - tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi))**2)/sqrt(pi) - 2.0*z1*(1 - tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi))**2)*(0.134145*z1**2 + 1)**2*tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi))/pi + 1.0*sqrt(2)*(1 - tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi))**2)*(0.134145*z1**2 + 1)/sqrt(pi) 
print(f"Input Function in the Block 2, 3rd Dimension, Layer 1: {d_f_z1} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 1: {d_d_f_z1} ")

z2=sm.symbols('z2')
f_z2= 0.5 * z2 * (1 + sm.tanh(pow(z2,1))) 
d_f_z2=sm.diff(f_z2)# 1.5*z2**3*(1 - tanh(z2**3)**2) + 0.5*tanh(z2**3) + 0.5 
d_d_f_z2=sm.diff(d_f_z2)# -9.0*z2**5*(1 - tanh(z2**3)**2)*tanh(z2**3) + 6.0*z2**2*(1 - tanh(z2**3)**2) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 2: {d_f_z2} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 2: {d_d_f_z2} ")


z3=sm.symbols('z3')
f_z3= 0.5 * z3 * (z3 + 0.044715 * z3**3)
d_f_z3=sm.diff(f_z3)# 0.0223575*z3**3 + 0.5*z3*(0.134145*z3**2 + 1) + 0.5*z3 
d_d_f_z3=sm.diff(d_f_z3)# 0.26829*z3**2 + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 3: {d_f_z3} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer : {d_d_f_z3} ")

z4=sm.symbols('z4')
f_z4= sm.exp(sm.tanh) +  pow(sm.sinh(z4),2)
d_f_z4=sm.diff(f_z4)# 2*sinh(z4)*cosh(z4) 
d_d_f_z4=sm.diff(d_f_z4)# 2*sinh(z4)**2 + 2*cosh(z4)**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 4: {d_f_z4} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 4: {d_d_f_z4} ")

z5=sm.symbols('z5')
f_z5= 0.5 * z5 * (z5 + 0.044715 * z5**3) + sm.cosh(pow(z5,4))
d_f_z5=sm.diff(f_z5)# 4*z5**3*sinh(z5**4) + 0.0223575*z5**3 + 0.5*z5*(0.134145*z5**2 + 1) + 0.5*z5 
d_d_f_z5=sm.diff(d_f_z5)# 16*z5**6*cosh(z5**4) + 12*z5**2*sinh(z5**4) + 0.26829*z5**2 + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 5: {d_f_z5} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 5: {d_d_f_z5} ")

z6=sm.symbols('z6')
f_z6= sm.exp(z6) *sm.cos(z6**2) * (z6 + 0.044715 * z6**3)
d_f_z6=sm.diff(f_z6)# -2*z6*(0.044715*z6**3 + z6)*exp(z6)*sin(z6**2) + (0.134145*z6**2 + 1)*exp(z6)*cos(z6**2) + (0.044715*z6**3 + z6)*exp(z6)*cos(z6**2) 
d_d_f_z6=sm.diff(d_f_z6)# -4*z6**2*(0.044715*z6**3 + z6)*exp(z6)*cos(z6**2) - 4*z6*(0.134145*z6**2 + 1)*exp(z6)*sin(z6**2) - 4*z6*(0.044715*z6**3 + z6)*exp(z6)*sin(z6**2) + 0.26829*z6*exp(z6)*cos(z6**2) + 2*(0.134145*z6**2 + 1)*exp(z6)*cos(z6**2) - 2*(0.044715*z6**3 + z6)*exp(z6)*sin(z6**2) + (0.044715*z6**3 + z6)*exp(z6)*cos(z6**2
print(f"Input Function in the Block 2, 3rd Dimension,Layer 6: {d_f_z6} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 6: {d_d_f_z6} ")

z7=sm.symbols('z7')
f_z7= 0.5 * z7 * (z7 + 0.044715 * z7**3)
d_f_z7=sm.diff(f_z7)# 0.0223575*z7**3 + 0.5*z7*(0.134145*z7**2 + 1) + 0.5*z7 
d_d_f_z7=sm.diff(d_f_z7)# 0.26829*z7**2 + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 7: {d_f_z7} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 7: {d_d_f_z7} ")

z8=sm.symbols('z8')
f_z8=sm.log(z8**4) + sm.tanh(z8 + 0.04475 * pow(z8,3))
d_f_z8=sm.diff(f_z8)# (1 - tanh(0.04475*z8**3 + z8)**2)*(0.13425*z8**2 + 1) + 4/z8 
d_d_f_z8=sm.diff(d_f_z8)# 0.2685*z8*(1 - tanh(0.04475*z8**3 + z8)**2) - 2*(1 - tanh(0.04475*z8**3 + z8)**2)*(0.13425*z8**2 + 1)**2*tanh(0.04475*z8**3 + z8) - 4/z8**2
print(f"Input Function in the Block 2, 3rd Dimension,Layer 8: {d_f_z8} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 8: {d_d_f_z8} ")

z9=sm.symbols('z9')
f_z9= 0.5 * z9 * (z9 + sm.sinh(0.044715 * z9**3))
d_f_z9=sm.diff(f_z9)# 0.5*z9*(0.134145*z9**2*cosh(0.044715*z9**3) + 1) + 0.5*z9 + 0.5*sinh(0.044715*z9**3) 
d_d_f_z9=sm.diff(d_f_z9)# 0.134145*z9**2*cosh(0.044715*z9**3) + 0.5*z9*(0.017994881025*z9**4*sinh(0.044715*z9**3) + 0.26829*z9*cosh(0.044715*z9**3)) + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 9: {d_f_z9} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 9: {d_d_f_z9} ")

z10=sm.symbols('z10')
f_z10= sm.exp(sm.sinh(z10**2)) + sm.tanh(z10**3)
d_f_z10=sm.diff(f_z10)# 3*z10**2*(1 - tanh(z10**3)**2) + 2*z10*exp(sinh(z10**2))*cosh(z10**2) 
d_d_f_z10=sm.diff(d_f_z10)# -18*z10**4*(1 - tanh(z10**3)**2)*tanh(z10**3) + 4*z10**2*exp(sinh(z10**2))*sinh(z10**2) + 4*z10**2*exp(sinh(z10**2))*cosh(z10**2)**2 + 6*z10*(1 - tanh(z10**3)**2) + 2*exp(sinh(z10**2))*cosh(z10**2) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 10: {d_f_z10} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 10: {d_d_f_z10} ")

import pandas as pd
import numpy as np
import sympy as sm
import scipy.special as sp
from scipy.special import erf
from sympy import diff, symbols

z11=sm.symbols('z11')
f_z11=sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z11=sm.diff(f_z11)# 3*z11**2*(1 - tanh(z11**3)**2) + 2*z11*exp(sinh(z11**2))*cosh(z11**2) 
d_d_f_z11=sm.diff(d_f_z11)# -18*z11**4*(1 - tanh(z11**3)**2)*tanh(z11**3) + 4*z11**2*exp(sinh(z11**2))*sinh(z11**2) + 4*z11**2*exp(sinh(z11**2))*cosh(z11**2)**2 + 6*z11*(1 - tanh(z11**3)**2) + 2*exp(sinh(z11**2))*cosh(z11**2)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 11: {d_f_z11} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 11: {d_d_f_z11} ")

z12=symbols('z12')
f_z12=sm.exp(sm.cos(z12**2)) + sm.tanh(z12**3) + sm.exp(sm.log(z12))
d_f_z12=sm.diff(f_z12)# 3*z12**2*(1 - tanh(z12**3)**2) - 2*z12*exp(cos(z12**2))*sin(z12**2) + 1 
d_d_f_z12=sm.diff(d_f_z12)# -18*z12**4*(1 - tanh(z12**3)**2)*tanh(z12**3) + 4*z12**2*exp(cos(z12**2))*sin(z12**2)**2 - 4*z12**2*exp(cos(z12**2))*cos(z12**2) + 6*z12*(1 - tanh(z12**3)**2) - 2*exp(cos(z12**2))*sin(z12**2)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 12: {d_f_z12} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 12: {d_d_f_z12} ")

z13=sm.symbols('z13')
f_z13=2*z13*sm.cos(z13**2) + 2*sm.sinh(2*z13) #sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z13=sm.diff(f_z13)# -4*z13**2*sin(z13**2) + 2*cos(z13**2) + 4*cosh(2*z13) 
d_d_f_z13=sm.diff(d_f_z13)# -8*z13**3*cos(z13**2) - 12*z13*sin(z13**2) + 8*sinh(2*z13) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 13: {d_f_z13} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 13: {d_d_f_z13} ")

z14=sm.symbols('z14')
f_z14=12*z14**2 + 3*sm.exp(3*z14) #sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z14=sm.diff(f_z14)# 24*z14 + 9*exp(3*z14) 
d_d_f_z14=sm.diff(d_f_z14)# 27*exp(3*z14) + 24 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 14: {d_f_z14} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 14: {d_d_f_z14} ")

z15=sm.symbols('z15')
f_z15=20*z15**4*(1 - sm.tanh(4*z15**5)**2) + 6*z15*sm.cos(3*z15**2)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z15=sm.diff(f_z15)# -800*z15**8*(1 - tanh(4*z15**5)**2)*tanh(4*z15**5) + 80*z15**3*(1 - tanh(4*z15**5)**2) - 36*z15**2*sin(3*z15**2) + 6*cos(3*z15**2) 
d_d_f_z15=sm.diff(d_f_z15)# -16000*z15**12*(1 - tanh(4*z15**5)**2)**2 + 32000*z15**12*(1 - tanh(4*z15**5)**2)*tanh(4*z15**5)**2 - 9600*z15**7*(1 - tanh(4*z15**5)**2)*tanh(4*z15**5) - 216*z15**3*cos(3*z15**2) + 240*z15**2*(1 - tanh(4*z15**5)**2) - 108*z15*sin(3*z15**2)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 15: {d_f_z15} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 15: {d_d_f_z15} ")

z16=sm.symbols('z16')
f_z16=10*z16**4*sm.sinh(2*z16**5) + 4*z16**3*sm.exp(z16**4) + 5/z16#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z16=sm.diff(f_z16)# 100*z16**8*cosh(2*z16**5) + 16*z16**6*exp(z16**4) + 40*z16**3*sinh(2*z16**5) + 12*z16**2*exp(z16**4) - 5/z16**2 
d_d_f_z16=sm.diff(d_f_z16)# 1000*z16**12*sinh(2*z16**5) + 64*z16**9*exp(z16**4) + 1200*z16**7*cosh(2*z16**5) + 144*z16**5*exp(z16**4) + 120*z16**2*sinh(2*z16**5) + 24*z16*exp(z16**4) + 10/z16**3
print(f"Input Function in the Block 2, 3rd Dimension,Layer 16: {d_f_z16} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 16: {d_d_f_z16} ")

z17=sm.symbols('z17')
f_z17=2*z17*(sm.tan(z17**2)**2 + 1) + 2*sm.cosh(2*z17)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z17=sm.diff(f_z17)# 8*z17**2*(tan(z17**2)**2 + 1)*tan(z17**2) + 2*tan(z17**2)**2 + 4*sinh(2*z17) + 2 
d_d_f_z17=sm.diff(d_f_z17)# 16*z17**3*(tan(z17**2)**2 + 1)**2 + 32*z17**3*(tan(z17**2)**2 + 1)*tan(z17**2)**2 + 24*z17*(tan(z17**2)**2 + 1)*tan(z17**2) + 8*cosh(2*z17)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 17: {d_f_z17} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 17: {d_d_f_z17} ")

z18=sm.symbols('z18')
f_z18=2*z18*sm.cos(z18**2) + 2*sm.sinh(2*z18)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z18=sm.diff(f_z18)# -4*z18**2*sin(z18**2) + 2*cos(z18**2) + 4*cosh(2*z18) 
d_d_f_z18=sm.diff(d_f_z18)# -8*z18**3*cos(z18**2) - 12*z18*sin(z18**2) + 8*sinh(2*z18) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 18: {d_f_z18} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 18: {d_d_f_z18} ")

z19=sm.symbols('z19')
f_z19=-4*z19**2*sm.sin(z19**2) + 2*sm.cos(z19**2) + 4*sm.cosh(2*z19)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z19=sm.diff(f_z19)# -8*z19**3*cos(z19**2) - 12*z19*sin(z19**2) + 8*sinh(2*z19) 
d_d_f_z19=sm.diff(d_f_z19)# 16*z19**4*sin(z19**2) - 48*z19**2*cos(z19**2) - 12*sin(z19**2) + 16*cosh(2*z19) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 19: {d_f_z19} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 19: {d_d_f_z19} ")

z20=sm.symbols('z20')
f_z20=2*z20*sm.cos(z20**2) + 2*sm.log(2*z20)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z20=sm.diff(f_z20)# -4*z20**2*sin(z20**2) + 2*cos(z20**2) + 2/z20
d_d_f_z20=sm.diff(d_f_z20)# -8*z20**3*cos(z20**2) - 12*z20*sin(z20**2) - 2/z20**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 20: {d_f_z20} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 20: {d_d_f_z20} ")

z21=sm.symbols('z21')
f_z21= sm.exp(sm.sin(z21**5)) + sm.cosh(2*z21**4) + sm.log(z21**4)  #2*z20*sm.cos(z20**2) + 2*sm.log(2*z20)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z21=sm.diff(f_z21)# 5*z21**4*exp(sin(z21**5))*cos(z21**5) + 8*z21**3*sinh(2*z21**4) + 4/z21 
d_d_f_z21=sm.diff(d_f_z21)# -25*z21**8*exp(sin(z21**5))*sin(z21**5) + 25*z21**8*exp(sin(z21**5))*cos(z21**5)**2 + 64*z21**6*cosh(2*z21**4) + 20*z21**3*exp(sin(z21**5))*cos(z21**5) + 24*z21**2*sinh(2*z21**4) - 4/z21**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 21: {d_f_z21} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 21: {d_d_f_z21} ")

z22=sm.symbols('z22')
f_z22= sm.tanh(z22**2 + z22**3) + sm.cosh(2*z22**2 + sm.tan(z22 **2))  #2*z20*sm.cos(z20**2) + 2*sm.log(2*z20)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z22=sm.diff(f_z22)# (1 - tanh(z22**3 + z22**2)**2)*(3*z22**2 + 2*z22) + (2*z22*(tan(z22**2)**2 + 1) + 4*z22)*sinh(2*z22**2 + tan(z22**2)) 
d_d_f_z22=sm.diff(d_f_z22)# (1 - tanh(z22**3 + z22**2)**2)*(6*z22 + 2) - 2*(1 - tanh(z22**3 + z22**2)**2)*(3*z22**2 + 2*z22)**2*tanh(z22**3 + z22**2) + (2*z22*(tan(z22**2)**2 + 1) + 4*z22)**2*cosh(2*z22**2 + tan(z22**2)) + (8*z22**2*(tan(z22**2)**2 + 1)*tan(z22**2) + 2*tan(z22**2)**2 + 6)*sinh(2*z22**2 + tan(z22**2))
print(f"Input Function in the Block 2, 3rd Dimension,Layer 22: {d_f_z22} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 22: {d_d_f_z22} ")

z23=sm.symbols('z23')
f_z23=sm.sinh(2*z23**5) + 4*z23**3*sm.exp(z23**4) + 5/z23#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z23=sm.diff(f_z23)# -4*z20**2*sin(z20**2) + 2*cos(z20**2) + 2/z20
d_d_f_z23=sm.diff(d_f_z23)# -8*z20**3*cos(z20**2) - 12*z20*sin(z20**2) - 2/z20**2 
print(f"Input Function in the Block 3, 1st Dimension, Layer 23: {d_f_z23} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 23: {d_d_f_z23} ")

z24=sm.symbols('z24')
f_z24=2*z24*(sm.tan(z24**2)**2 + 1) + 2*sm.cosh(2*z24)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z24=sm.diff(f_z24)# -4*z20**2*sin(z20**2) + 2*cos(z20**2) + 2/z20
d_d_f_z24=sm.diff(d_f_z24)# -8*z20**3*cos(z20**2) - 12*z20*sin(z20**2) - 2/z20**2 
print(f"Input Function in the Block 3, 1st Dimension, Layer 24: {d_f_z24} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 24: {d_d_f_z24} ")

z25=sm.symbols('z25')
f_z25=10*z25**4*sm.sinh(2*z25**5) + 4*z25**3*sm.exp(z25**4) + 5/z25#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z25=sm.diff(f_z25)# -4*z20**2*sin(z20**2) + 2*cos(z20**2) + 2/z20
d_d_f_z25=sm.diff(d_f_z25)# -8*z20**3*cos(z20**2) - 12*z20*sin(z20**2) - 2/z20**2 
print(f"Input Function in the Block 3, 1st Dimension, Layer 25: {d_f_z25} ")
print(f"Output Function in the Block 2, 3rd Dimension, Lazer 25: {d_d_f_z25} ")

import sympy as sm  
import scipy.special as sp
from sympy import symbols, diff
#from scipy.special import erf, erf_zeros
y1=sm.symbols('y1')
f_y1= 0.5 * y1 * (1 + sm.tanh(sm.sqrt(2 / sm.pi) * (y1 + 0.044715 * y1**3)))
d_f_y1=sm.diff(f_y1)# 0.5*sqrt(2)*y1*(1 - tanh(sqrt(2)*(0.044715*y1**3 + y1)/sqrt(pi))**2)*(0.134145*y1**2 + 1)/sqrt(pi) + 0.5*tanh(sqrt(2)*(0.044715*y1**3 + y1)/sqrt(pi)) + 0.5 
d_d_f_y1=sm.diff(d_f_y1)# 0.134145*sqrt(2)*y1**2*(1 - tanh(sqrt(2)*(0.044715*y1**3 + y1)/sqrt(pi))**2)/sqrt(pi) - 2.0*y1*(1 - tanh(sqrt(2)*(0.044715*y1**3 + y1)/sqrt(pi))**2)*(0.134145*y1**2 + 1)**2*tanh(sqrt(2)*(0.044715*y1**3 + y1)/sqrt(pi))/pi + 1.0*sqrt(2)*(1 - tanh(sqrt(2)*(0.044715*y1**3 + y1)/sqrt(pi))**2)*(0.134145*y1**2 + 1)/sqrt(pi) 
print(f"Input Function in the Block 2, 3rd Dimension, Layer 1: {d_f_y1} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 1: {d_d_f_y1} ")

y2=sm.symbols('y2')
f_y2= 0.5 * y2 * (1 + sm.tanh(pow(y2,1))) 
d_f_y2=sm.diff(f_y2)# 1.5*y2**3*(1 - tanh(y2**3)**2) + 0.5*tanh(y2**3) + 0.5 
d_d_f_y2=sm.diff(d_f_y2)# -9.0*y2**5*(1 - tanh(y2**3)**2)*tanh(y2**3) + 6.0*y2**2*(1 - tanh(y2**3)**2) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 2: {d_f_y2} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 2: {d_d_f_y2} ")

y3=sm.symbols('y3')
f_y3= 0.5 * y3 * (y3 + 0.044715 * y3**3)
d_f_y3=sm.diff(f_y3)# 0.0223575*y3**3 + 0.5*y3*(0.134145*y3**2 + 1) + 0.5*y3 
d_d_f_y3=sm.diff(d_f_y3)# 0.26829*y3**2 + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 3:{d_f_y3} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer : {d_d_f_y3} ")

y4=sm.symbols('y4')
f_y4= sm.exp(sm.tanh) +  pow(sm.sinh(y4),2)
d_f_y4=sm.diff(f_y4)# 2*sinh(y4)*cosh(y4) 
d_d_f_y4=sm.diff(d_f_y4)# 2*sinh(y4)**2 + 2*cosh(y4)**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 4: {d_f_y4} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 4: {d_d_f_y4} ")

y5=sm.symbols('y5')
f_y5= 0.5 * y5 * (y5 + 0.044715 * y5**3) + sm.cosh(pow(y5,4))
d_f_y5=sm.diff(f_y5)# 4*y5**3*sinh(y5**4) + 0.0223575*y5**3 + 0.5*y5*(0.134145*y5**2 + 1) + 0.5*y5 
d_d_f_y5=sm.diff(d_f_y5)# 16*y5**6*cosh(y5**4) + 12*y5**2*sinh(y5**4) + 0.26829*y5**2 + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 5: {d_f_y5} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 5: {d_d_f_y5} ")

y6=sm.symbols('y6')
f_y6= sm.exp(y6) *sm.cos(y6**2) * (y6 + 0.044715 * y6**3)
d_f_y6=sm.diff(f_y6)# -2*y6*(0.044715*y6**3 + y6)*exp(y6)*sin(y6**2) + (0.134145*y6**2 + 1)*exp(y6)*cos(y6**2) + (0.044715*y6**3 + y6)*exp(y6)*cos(y6**2) 
d_d_f_y6=sm.diff(d_f_y6)# -4*y6**2*(0.044715*y6**3 + y6)*exp(y6)*cos(y6**2) - 4*y6*(0.134145*y6**2 + 1)*exp(y6)*sin(y6**2) - 4*y6*(0.044715*y6**3 + y6)*exp(y6)*sin(y6**2) + 0.26829*y6*exp(y6)*cos(y6**2) + 2*(0.134145*y6**2 + 1)*exp(y6)*cos(y6**2) - 2*(0.044715*y6**3 + y6)*exp(y6)*sin(y6**2) + (0.044715*y6**3 + y6)*exp(y6)*cos(y6**2
print(f"Input Function in the Block 2, 3rd Dimension,Layer 6: {d_f_y6} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 6: {d_d_f_y6} ")

y7=sm.symbols('y7')
f_y7= 0.5 * y7 * (y7 + 0.044715 * y7**3)
d_f_y7=sm.diff(f_y7)# 0.0223575*y7**3 + 0.5*y7*(0.134145*y7**2 + 1) + 0.5*y7 
d_d_f_y7=sm.diff(d_f_y7)# 0.26829*y7**2 + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 7: {d_f_y7} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 7: {d_d_f_y7} ")

y8=sm.symbols('y8')
f_y8=sm.log(y8**4) + sm.tanh(y8 + 0.04475 * pow(y8,3))
d_f_y8=sm.diff(f_y8)# (1 - tanh(0.04475*y8**3 + y8)**2)*(0.13425*y8**2 + 1) + 4/y8 
d_d_f_y8=sm.diff(d_f_y8)# 0.2685*y8*(1 - tanh(0.04475*y8**3 + y8)**2) - 2*(1 - tanh(0.04475*y8**3 + y8)**2)*(0.13425*y8**2 + 1)**2*tanh(0.04475*y8**3 + y8) - 4/y8**2
print(f"Input Function in the Block 2, 3rd Dimension,Layer 8: {d_f_y8} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 8: {d_d_f_y8} ")

y8=sm.symbols('y8')
f_y8=sm.log(y8**4) + sm.tanh(y8 + 0.04475 * pow(y8,3))
d_f_y8=sm.diff(f_y8)# (1 - tanh(0.04475*y8**3 + y8)**2)*(0.13425*y8**2 + 1) + 4/y8 
d_d_f_y8=sm.diff(d_f_y8)# 0.2685*y8*(1 - tanh(0.04475*y8**3 + y8)**2) - 2*(1 - tanh(0.04475*y8**3 + y8)**2)*(0.13425*y8**2 + 1)**2*tanh(0.04475*y8**3 + y8) - 4/y8**2
print(f"Input Function in the Block 2, 3rd Dimension,Layer 8: {d_f_y8} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 8: {d_d_f_y8} ")

y9=sm.symbols('y9')
f_y9= 0.5 * y9 * (y9 + sm.sinh(0.044715 * y9**3))
d_f_y9=sm.diff(f_y9)# 0.5*y9*(0.134145*y9**2*cosh(0.044715*y9**3) + 1) + 0.5*y9 + 0.5*sinh(0.044715*y9**3) 
d_d_f_y9=sm.diff(d_f_y9)# 0.134145*y9**2*cosh(0.044715*y9**3) + 0.5*y9*(0.017994881025*y9**4*sinh(0.044715*y9**3) + 0.26829*y9*cosh(0.044715*y9**3)) + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 9: {d_f_y9} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 9: {d_d_f_y9} ")

y10=sm.symbols('y10')
f_y10= sm.exp(sm.sinh(y10**2)) + sm.tanh(y10**3)
d_f_y10=sm.diff(f_y10)# 3*y10**2*(1 - tanh(y10**3)**2) + 2*y10*exp(sinh(y10**2))*cosh(y10**2) 
d_d_f_y10=sm.diff(d_f_y10)# -18*y10**4*(1 - tanh(y10**3)**2)*tanh(y10**3) + 4*y10**2*exp(sinh(y10**2))*sinh(y10**2) + 4*y10**2*exp(sinh(y10**2))*cosh(y10**2)**2 + 6*y10*(1 - tanh(y10**3)**2) + 2*exp(sinh(y10**2))*cosh(y10**2) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 10: {d_f_y10} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 10: {d_d_f_y10} ")

import pandas as pd
import numpy as np
import sympy as sm  
import scipy.special as sp
#from scipy.special import erf
from sympy import diff, symbols

y11=sm.symbols('y11')
f_y11=sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y11=sm.diff(f_y11)# 3*y11**2*(1 - tanh(y11**3)**2) + 2*y11*exp(sinh(y11**2))*cosh(y11**2) 
d_d_f_y11=sm.diff(d_f_y11)# -18*y11**4*(1 - tanh(y11**3)**2)*tanh(y11**3) + 4*y11**2*exp(sinh(y11**2))*sinh(y11**2) + 4*y11**2*exp(sinh(y11**2))*cosh(y11**2)**2 + 6*y11*(1 - tanh(y11**3)**2) + 2*exp(sinh(y11**2))*cosh(y11**2)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 11: {d_f_y11} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 11: {d_d_f_y11} ")

y12=sm.symbols('y12')
f_y12=sm.exp(sm.cos(y12**2)) + sm.tanh(y12**3) + sm.exp(sm.log(y12))
d_f_y12=sm.diff(f_y12)# 3*y12**2*(1 - tanh(y12**3)**2) - 2*y12*exp(cos(y12**2))*sin(y12**2) + 1 
d_d_f_y12=sm.diff(d_f_y12)# -18*y12**4*(1 - tanh(y12**3)**2)*tanh(y12**3) + 4*y12**2*exp(cos(y12**2))*sin(y12**2)**2 - 4*y12**2*exp(cos(y12**2))*cos(y12**2) + 6*y12*(1 - tanh(y12**3)**2) - 2*exp(cos(y12**2))*sin(y12**2)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 12: {d_f_y12} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 12: {d_d_f_y12} ")

y13=sm.symbols('y13')
f_y13=2*y13*sm.cos(y13**2) + 2*sm.sinh(2*y13) #sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y13=sm.diff(f_y13)# -4*y13**2*sin(y13**2) + 2*cos(y13**2) + 4*cosh(2*y13) 
d_d_f_y13=sm.diff(d_f_y13)# -8*y13**3*cos(y13**2) - 12*y13*sin(y13**2) + 8*sinh(2*y13) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 13: {d_f_y13} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 13: {d_d_f_y13} ")

y14=sm.symbols('y14')
f_y14=12*y14**2 + 3*sm.exp(3*y14) #sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y14=sm.diff(f_y14)# 24*y14 + 9*exp(3*y14) 
d_d_f_y14=sm.diff(d_f_y14)# 27*exp(3*y14) + 24 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 14: {d_f_y14} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 14: {d_d_f_y14} ")

y15=sm.symbols('y15')
f_y15=20*y15**4*(1 - sm.tanh(4*y15**5)**2) + 6*y15*sm.cos(3*y15**2)#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y15=sm.diff(f_y15)# -800*y15**8*(1 - tanh(4*y15**5)**2)*tanh(4*y15**5) + 80*y15**3*(1 - tanh(4*y15**5)**2) - 36*y15**2*sin(3*y15**2) + 6*cos(3*y15**2) 
d_d_f_y15=sm.diff(d_f_y15)# -16000*y15**12*(1 - tanh(4*y15**5)**2)**2 + 32000*y15**12*(1 - tanh(4*y15**5)**2)*tanh(4*y15**5)**2 - 9600*y15**7*(1 - tanh(4*y15**5)**2)*tanh(4*y15**5) - 216*y15**3*cos(3*y15**2) + 240*y15**2*(1 - tanh(4*y15**5)**2) - 108*y15*sin(3*y15**2)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 15: {d_f_y15} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 15: {d_d_f_y15} ")

y16=sm.symbols('y16')
f_y16=10*y16**4*sm.sinh(2*y16**5) + 4*y16**3*sm.exp(y16**4) + 5/y16#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y16=sm.diff(f_y16)# 100*y16**8*cosh(2*y16**5) + 16*y16**6*exp(y16**4) + 40*y16**3*sinh(2*y16**5) + 12*y16**2*exp(y16**4) - 5/y16**2 
d_d_f_y16=sm.diff(d_f_y16)# 1000*y16**12*sinh(2*y16**5) + 64*y16**9*exp(y16**4) + 1200*y16**7*cosh(2*y16**5) + 144*y16**5*exp(y16**4) + 120*y16**2*sinh(2*y16**5) + 24*y16*exp(y16**4) + 10/y16**3
print(f"Input Function in the Block 2, 3rd Dimension,Layer 16: {d_f_y16} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 16: {d_d_f_y16} ")

y17=sm.symbols('y17')
f_y17=2*y17*(sm.tan(y17**2)**2 + 1) + 2*sm.cosh(2*y17)#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y17=sm.diff(f_y17)# 8*y17**2*(tan(y17**2)**2 + 1)*tan(y17**2) + 2*tan(y17**2)**2 + 4*sinh(2*y17) + 2 
d_d_f_y17=sm.diff(d_f_y17)# 16*y17**3*(tan(y17**2)**2 + 1)**2 + 32*y17**3*(tan(y17**2)**2 + 1)*tan(y17**2)**2 + 24*y17*(tan(y17**2)**2 + 1)*tan(y17**2) + 8*cosh(2*y17)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 17: {d_f_y17} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 17: {d_d_f_y17} ")

y18=sm.symbols('y18')
f_y18=2*y18*sm.cos(y18**2) + 2*sm.sinh(2*y18)#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y18=sm.diff(f_y18)# -4*y18**2*sin(y18**2) + 2*cos(y18**2) + 4*cosh(2*y18) 
d_d_f_y18=sm.diff(d_f_y18)# -8*y18**3*cos(y18**2) - 12*y18*sin(y18**2) + 8*sinh(2*y18) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 18: {d_f_y18} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 18: {d_d_f_y18} ")

y19=sm.symbols('y19')
f_y19=-4*y19**2*sm.sin(y19**2) + 2*sm.cos(y19**2) + 4*sm.cosh(2*y19)#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y19=sm.diff(f_y19)# -8*y19**3*cos(y19**2) - 12*y19*sin(y19**2) + 8*sinh(2*y19) 
d_d_f_y19=sm.diff(d_f_y19)# 16*y19**4*sin(y19**2) - 48*y19**2*cos(y19**2) - 12*sin(y19**2) + 16*cosh(2*y19) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 19: {d_f_y19} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 19: {d_d_f_y19} ")

y20=sm.symbols('y20')
f_y20=2*y20*sm.cos(y20**2) + 2*sm.log(2*y20)#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y20=sm.diff(f_y20)# -4*y20**2*sin(y20**2) + 2*cos(y20**2) + 2/y20
d_d_f_y20=sm.diff(d_f_y20)# -8*y20**3*cos(y20**2) - 12*y20*sin(y20**2) - 2/y20**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 20: {d_f_y20} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 20: {d_d_f_y20} ")

y21=sm.symbols('y21')
f_y21= sm.exp(sm.sin(y21**5)) + sm.cosh(2*y21**4) + sm.log(y21**4)  #2*y20*sm.cos(y20**2) + 2*sm.log(2*y20)#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y21=sm.diff(f_y21)# 5*y21**4*exp(sin(y21**5))*cos(y21**5) + 8*y21**3*sinh(2*y21**4) + 4/y21 
d_d_f_y21=sm.diff(d_f_y21)# -25*y21**8*exp(sin(y21**5))*sin(y21**5) + 25*y21**8*exp(sin(y21**5))*cos(y21**5)**2 + 64*y21**6*cosh(2*y21**4) + 20*y21**3*exp(sin(y21**5))*cos(y21**5) + 24*y21**2*sinh(2*y21**4) - 4/y21**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 21: {d_f_y21} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 21: {d_d_f_y21} ")

y22=sm.symbols('y22')
f_y22= sm.tanh(y22**2 + y22**3) + sm.cosh(2*y22**2 + sm.tan(y22 **2))  #2*y20*sm.cos(y20**2) + 2*sm.log(2*y20)#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y22=sm.diff(f_y22)# (1 - tanh(y22**3 + y22**2)**2)*(3*y22**2 + 2*y22) + (2*y22*(tan(y22**2)**2 + 1) + 4*y22)*sinh(2*y22**2 + tan(y22**2)) 
d_d_f_y22=sm.diff(d_f_y22)# (1 - tanh(y22**3 + y22**2)**2)*(6*y22 + 2) - 2*(1 - tanh(y22**3 + y22**2)**2)*(3*y22**2 + 2*y22)**2*tanh(y22**3 + y22**2) + (2*y22*(tan(y22**2)**2 + 1) + 4*y22)**2*cosh(2*y22**2 + tan(y22**2)) + (8*y22**2*(tan(y22**2)**2 + 1)*tan(y22**2) + 2*tan(y22**2)**2 + 6)*sinh(2*y22**2 + tan(y22**2))
print(f"Input Function in the Block 2, 3rd Dimension,Layer 22: {d_f_y22} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 22: {d_d_f_y22} ")

y23=sm.symbols('y23')
f_y23=sm.sinh(2*y23**5) + 4*y23**3*sm.exp(y23**4) + 5/y23#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y23=sm.diff(f_y23)# -4*y20**2*sin(y20**2) + 2*cos(y20**2) + 2/y20
d_d_f_y23=sm.diff(d_f_y23)# -8*y20**3*cos(y20**2) - 12*y20*sin(y20**2) - 2/y20**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 23: {d_f_y23} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 23: {d_d_f_y23} ")

y24=sm.symbols('y24')
f_y24=2*y24*(sm.tan(y24**2)**2 + 1) + 2*sm.cosh(2*y24)#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y24=sm.diff(f_y24)# -4*y20**2*sin(y20**2) + 2*cos(y20**2) + 2/y20
d_d_f_y24=sm.diff(d_f_y24)# -8*y20**3*cos(y20**2) - 12*y20*sin(y20**2) - 2/y20**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 24: {d_f_y24} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 24: {d_d_f_y24} ")

y25=sm.symbols('y25')
f_y25=10*y25**4*sm.sinh(2*y25**5) + 4*y25**3*sm.exp(y25**4) + 5/y25#sm.exp(sm.sinh(y11**2)) + sm.tanh(y11**3)
d_f_y25=sm.diff(f_y25)# -4*y20**2*sin(y20**2) + 2*cos(y20**2) + 2/y20
d_d_f_y25=sm.diff(d_f_y25)# -8*y20**3*cos(y20**2) - 12*y20*sin(y20**2) - 2/y20**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 24: {d_f_y25} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 24: {d_d_f_y25} ")



import sympy as sm  
import scipy.special as sp
from sympy import symbols, diff
from scipy.special import erf, erf_zeros
z1=sm.symbols('z1')
f_z1= 0.5 * z1 * (1 + sm.tanh(sm.sqrt(2 / sm.pi) * (z1 + 0.044715 * z1**3)))
d_f_z1=sm.diff(f_z1)# 0.5*sqrt(2)*z1*(1 - tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi))**2)*(0.134145*z1**2 + 1)/sqrt(pi) + 0.5*tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi)) + 0.5 
d_d_f_z1=sm.diff(d_f_z1)# 0.134145*sqrt(2)*z1**2*(1 - tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi))**2)/sqrt(pi) - 2.0*z1*(1 - tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi))**2)*(0.134145*z1**2 + 1)**2*tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi))/pi + 1.0*sqrt(2)*(1 - tanh(sqrt(2)*(0.044715*z1**3 + z1)/sqrt(pi))**2)*(0.134145*z1**2 + 1)/sqrt(pi) 
print(f"Input Function in the Block 2, 3rd Dimension, Layer 1: {d_f_z1} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 1: {d_d_f_z1} ")

z2=sm.symbols('z2')
f_z2= 0.5 * z2 * (1 + sm.tanh(pow(z2,1))) 
d_f_z2=sm.diff(f_z2)# 1.5*z2**3*(1 - tanh(z2**3)**2) + 0.5*tanh(z2**3) + 0.5 
d_d_f_z2=sm.diff(d_f_z2)# -9.0*z2**5*(1 - tanh(z2**3)**2)*tanh(z2**3) + 6.0*z2**2*(1 - tanh(z2**3)**2) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 2: {d_f_z2} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 2: {d_d_f_z2} ")


z3=sm.symbols('z3')
f_z3= 0.5 * z3 * (z3 + 0.044715 * z3**3)
d_f_z3=sm.diff(f_z3)# 0.0223575*z3**3 + 0.5*z3*(0.134145*z3**2 + 1) + 0.5*z3 
d_d_f_z3=sm.diff(d_f_z3)# 0.26829*z3**2 + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 3: {d_f_z3} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer : {d_d_f_z3} ")

z4=sm.symbols('z4')
f_z4= sm.exp(sm.tanh) +  pow(sm.sinh(z4),2)
d_f_z4=sm.diff(f_z4)# 2*sinh(z4)*cosh(z4) 
d_d_f_z4=sm.diff(d_f_z4)# 2*sinh(z4)**2 + 2*cosh(z4)**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 4: {d_f_z4} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 4: {d_d_f_z4} ")

z5=sm.symbols('z5')
f_z5= 0.5 * z5 * (z5 + 0.044715 * z5**3) + sm.cosh(pow(z5,4))
d_f_z5=sm.diff(f_z5)# 4*z5**3*sinh(z5**4) + 0.0223575*z5**3 + 0.5*z5*(0.134145*z5**2 + 1) + 0.5*z5 
d_d_f_z5=sm.diff(d_f_z5)# 16*z5**6*cosh(z5**4) + 12*z5**2*sinh(z5**4) + 0.26829*z5**2 + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 5: {d_f_z5} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 5: {d_d_f_z5} ")

z6=sm.symbols('z6')
f_z6= sm.exp(z6) *sm.cos(z6**2) * (z6 + 0.044715 * z6**3)
d_f_z6=sm.diff(f_z6)# -2*z6*(0.044715*z6**3 + z6)*exp(z6)*sin(z6**2) + (0.134145*z6**2 + 1)*exp(z6)*cos(z6**2) + (0.044715*z6**3 + z6)*exp(z6)*cos(z6**2) 
d_d_f_z6=sm.diff(d_f_z6)# -4*z6**2*(0.044715*z6**3 + z6)*exp(z6)*cos(z6**2) - 4*z6*(0.134145*z6**2 + 1)*exp(z6)*sin(z6**2) - 4*z6*(0.044715*z6**3 + z6)*exp(z6)*sin(z6**2) + 0.26829*z6*exp(z6)*cos(z6**2) + 2*(0.134145*z6**2 + 1)*exp(z6)*cos(z6**2) - 2*(0.044715*z6**3 + z6)*exp(z6)*sin(z6**2) + (0.044715*z6**3 + z6)*exp(z6)*cos(z6**2
print(f"Input Function in the Block 2, 3rd Dimension,Layer 6: {d_f_z6} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 6: {d_d_f_z6} ")

z7=sm.symbols('z7')
f_z7= 0.5 * z7 * (z7 + 0.044715 * z7**3)
d_f_z7=sm.diff(f_z7)# 0.0223575*z7**3 + 0.5*z7*(0.134145*z7**2 + 1) + 0.5*z7 
d_d_f_z7=sm.diff(d_f_z7)# 0.26829*z7**2 + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 7: {d_f_z7} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 7: {d_d_f_z7} ")

z8=sm.symbols('z8')
f_z8=sm.log(z8**4) + sm.tanh(z8 + 0.04475 * pow(z8,3))
d_f_z8=sm.diff(f_z8)# (1 - tanh(0.04475*z8**3 + z8)**2)*(0.13425*z8**2 + 1) + 4/z8 
d_d_f_z8=sm.diff(d_f_z8)# 0.2685*z8*(1 - tanh(0.04475*z8**3 + z8)**2) - 2*(1 - tanh(0.04475*z8**3 + z8)**2)*(0.13425*z8**2 + 1)**2*tanh(0.04475*z8**3 + z8) - 4/z8**2
print(f"Input Function in the Block 2, 3rd Dimension,Layer 8: {d_f_z8} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 8: {d_d_f_z8} ")

z9=sm.symbols('z9')
f_z9= 0.5 * z9 * (z9 + sm.sinh(0.044715 * z9**3))
d_f_z9=sm.diff(f_z9)# 0.5*z9*(0.134145*z9**2*cosh(0.044715*z9**3) + 1) + 0.5*z9 + 0.5*sinh(0.044715*z9**3) 
d_d_f_z9=sm.diff(d_f_z9)# 0.134145*z9**2*cosh(0.044715*z9**3) + 0.5*z9*(0.017994881025*z9**4*sinh(0.044715*z9**3) + 0.26829*z9*cosh(0.044715*z9**3)) + 1.0 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 9: {d_f_z9} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 9: {d_d_f_z9} ")

z10=sm.symbols('z10')
f_z10= sm.exp(sm.sinh(z10**2)) + sm.tanh(z10**3)
d_f_z10=sm.diff(f_z10)# 3*z10**2*(1 - tanh(z10**3)**2) + 2*z10*exp(sinh(z10**2))*cosh(z10**2) 
d_d_f_z10=sm.diff(d_f_z10)# -18*z10**4*(1 - tanh(z10**3)**2)*tanh(z10**3) + 4*z10**2*exp(sinh(z10**2))*sinh(z10**2) + 4*z10**2*exp(sinh(z10**2))*cosh(z10**2)**2 + 6*z10*(1 - tanh(z10**3)**2) + 2*exp(sinh(z10**2))*cosh(z10**2) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 10: {d_f_z10} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 10: {d_d_f_z10} ")

import pandas as pd
import numpy as np
import sympy as sm
import scipy.special as sp
from scipy.special import erf
from sympy import diff, symbols

z11=sm.symbols('z11')
f_z11=sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z11=sm.diff(f_z11)# 3*z11**2*(1 - tanh(z11**3)**2) + 2*z11*exp(sinh(z11**2))*cosh(z11**2) 
d_d_f_z11=sm.diff(d_f_z11)# -18*z11**4*(1 - tanh(z11**3)**2)*tanh(z11**3) + 4*z11**2*exp(sinh(z11**2))*sinh(z11**2) + 4*z11**2*exp(sinh(z11**2))*cosh(z11**2)**2 + 6*z11*(1 - tanh(z11**3)**2) + 2*exp(sinh(z11**2))*cosh(z11**2)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 11: {d_f_z11} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 11: {d_d_f_z11} ")

z12=symbols('z12')
f_z12=sm.exp(sm.cos(z12**2)) + sm.tanh(z12**3) + sm.exp(sm.log(z12))
d_f_z12=sm.diff(f_z12)# 3*z12**2*(1 - tanh(z12**3)**2) - 2*z12*exp(cos(z12**2))*sin(z12**2) + 1 
d_d_f_z12=sm.diff(d_f_z12)# -18*z12**4*(1 - tanh(z12**3)**2)*tanh(z12**3) + 4*z12**2*exp(cos(z12**2))*sin(z12**2)**2 - 4*z12**2*exp(cos(z12**2))*cos(z12**2) + 6*z12*(1 - tanh(z12**3)**2) - 2*exp(cos(z12**2))*sin(z12**2)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 12: {d_f_z12} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 12: {d_d_f_z12} ")

z13=sm.symbols('z13')
f_z13=2*z13*sm.cos(z13**2) + 2*sm.sinh(2*z13) #sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z13=sm.diff(f_z13)# -4*z13**2*sin(z13**2) + 2*cos(z13**2) + 4*cosh(2*z13) 
d_d_f_z13=sm.diff(d_f_z13)# -8*z13**3*cos(z13**2) - 12*z13*sin(z13**2) + 8*sinh(2*z13) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 13: {d_f_z13} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 13: {d_d_f_z13} ")

z14=sm.symbols('z14')
f_z14=12*z14**2 + 3*sm.exp(3*z14) #sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z14=sm.diff(f_z14)# 24*z14 + 9*exp(3*z14) 
d_d_f_z14=sm.diff(d_f_z14)# 27*exp(3*z14) + 24 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 14: {d_f_z14} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 14: {d_d_f_z14} ")

z15=sm.symbols('z15')
f_z15=20*z15**4*(1 - sm.tanh(4*z15**5)**2) + 6*z15*sm.cos(3*z15**2)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z15=sm.diff(f_z15)# -800*z15**8*(1 - tanh(4*z15**5)**2)*tanh(4*z15**5) + 80*z15**3*(1 - tanh(4*z15**5)**2) - 36*z15**2*sin(3*z15**2) + 6*cos(3*z15**2) 
d_d_f_z15=sm.diff(d_f_z15)# -16000*z15**12*(1 - tanh(4*z15**5)**2)**2 + 32000*z15**12*(1 - tanh(4*z15**5)**2)*tanh(4*z15**5)**2 - 9600*z15**7*(1 - tanh(4*z15**5)**2)*tanh(4*z15**5) - 216*z15**3*cos(3*z15**2) + 240*z15**2*(1 - tanh(4*z15**5)**2) - 108*z15*sin(3*z15**2)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 15: {d_f_z15} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 15: {d_d_f_z15} ")

z16=sm.symbols('z16')
f_z16=10*z16**4*sm.sinh(2*z16**5) + 4*z16**3*sm.exp(z16**4) + 5/z16#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z16=sm.diff(f_z16)# 100*z16**8*cosh(2*z16**5) + 16*z16**6*exp(z16**4) + 40*z16**3*sinh(2*z16**5) + 12*z16**2*exp(z16**4) - 5/z16**2 
d_d_f_z16=sm.diff(d_f_z16)# 1000*z16**12*sinh(2*z16**5) + 64*z16**9*exp(z16**4) + 1200*z16**7*cosh(2*z16**5) + 144*z16**5*exp(z16**4) + 120*z16**2*sinh(2*z16**5) + 24*z16*exp(z16**4) + 10/z16**3
print(f"Input Function in the Block 2, 3rd Dimension,Layer 16: {d_f_z16} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 16: {d_d_f_z16} ")

z17=sm.symbols('z17')
f_z17=2*z17*(sm.tan(z17**2)**2 + 1) + 2*sm.cosh(2*z17)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z17=sm.diff(f_z17)# 8*z17**2*(tan(z17**2)**2 + 1)*tan(z17**2) + 2*tan(z17**2)**2 + 4*sinh(2*z17) + 2 
d_d_f_z17=sm.diff(d_f_z17)# 16*z17**3*(tan(z17**2)**2 + 1)**2 + 32*z17**3*(tan(z17**2)**2 + 1)*tan(z17**2)**2 + 24*z17*(tan(z17**2)**2 + 1)*tan(z17**2) + 8*cosh(2*z17)
print(f"Input Function in the Block 2, 3rd Dimension,Layer 17: {d_f_z17} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 17: {d_d_f_z17} ")

z18=sm.symbols('z18')
f_z18=2*z18*sm.cos(z18**2) + 2*sm.sinh(2*z18)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z18=sm.diff(f_z18)# -4*z18**2*sin(z18**2) + 2*cos(z18**2) + 4*cosh(2*z18) 
d_d_f_z18=sm.diff(d_f_z18)# -8*z18**3*cos(z18**2) - 12*z18*sin(z18**2) + 8*sinh(2*z18) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 18: {d_f_z18} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 18: {d_d_f_z18} ")

z19=sm.symbols('z19')
f_z19=-4*z19**2*sm.sin(z19**2) + 2*sm.cos(z19**2) + 4*sm.cosh(2*z19)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z19=sm.diff(f_z19)# -8*z19**3*cos(z19**2) - 12*z19*sin(z19**2) + 8*sinh(2*z19) 
d_d_f_z19=sm.diff(d_f_z19)# 16*z19**4*sin(z19**2) - 48*z19**2*cos(z19**2) - 12*sin(z19**2) + 16*cosh(2*z19) 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 19: {d_f_z19} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 19: {d_d_f_z19} ")

z20=sm.symbols('z20')
f_z20=2*z20*sm.cos(z20**2) + 2*sm.log(2*z20)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z20=sm.diff(f_z20)# -4*z20**2*sin(z20**2) + 2*cos(z20**2) + 2/z20
d_d_f_z20=sm.diff(d_f_z20)# -8*z20**3*cos(z20**2) - 12*z20*sin(z20**2) - 2/z20**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 20: {d_f_z20} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 20: {d_d_f_z20} ")

z21=sm.symbols('z21')
f_z21= sm.exp(sm.sin(z21**5)) + sm.cosh(2*z21**4) + sm.log(z21**4)  #2*z20*sm.cos(z20**2) + 2*sm.log(2*z20)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z21=sm.diff(f_z21)# 5*z21**4*exp(sin(z21**5))*cos(z21**5) + 8*z21**3*sinh(2*z21**4) + 4/z21 
d_d_f_z21=sm.diff(d_f_z21)# -25*z21**8*exp(sin(z21**5))*sin(z21**5) + 25*z21**8*exp(sin(z21**5))*cos(z21**5)**2 + 64*z21**6*cosh(2*z21**4) + 20*z21**3*exp(sin(z21**5))*cos(z21**5) + 24*z21**2*sinh(2*z21**4) - 4/z21**2 
print(f"Input Function in the Block 2, 3rd Dimension,Layer 21: {d_f_z21} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 21: {d_d_f_z21} ")

z22=sm.symbols('z22')
f_z22= sm.tanh(z22**2 + z22**3) + sm.cosh(2*z22**2 + sm.tan(z22 **2))  #2*z20*sm.cos(z20**2) + 2*sm.log(2*z20)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z22=sm.diff(f_z22)# (1 - tanh(z22**3 + z22**2)**2)*(3*z22**2 + 2*z22) + (2*z22*(tan(z22**2)**2 + 1) + 4*z22)*sinh(2*z22**2 + tan(z22**2)) 
d_d_f_z22=sm.diff(d_f_z22)# (1 - tanh(z22**3 + z22**2)**2)*(6*z22 + 2) - 2*(1 - tanh(z22**3 + z22**2)**2)*(3*z22**2 + 2*z22)**2*tanh(z22**3 + z22**2) + (2*z22*(tan(z22**2)**2 + 1) + 4*z22)**2*cosh(2*z22**2 + tan(z22**2)) + (8*z22**2*(tan(z22**2)**2 + 1)*tan(z22**2) + 2*tan(z22**2)**2 + 6)*sinh(2*z22**2 + tan(z22**2))
print(f"Input Function in the Block 2, 3rd Dimension,Layer 22: {d_f_z22} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 22: {d_d_f_z22} ")

z23=sm.symbols('z23')
f_z23=sm.sinh(2*z23**5) + 4*z23**3*sm.exp(z23**4) + 5/z23#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z23=sm.diff(f_z23)# -4*z20**2*sin(z20**2) + 2*cos(z20**2) + 2/z20
d_d_f_z23=sm.diff(d_f_z23)# -8*z20**3*cos(z20**2) - 12*z20*sin(z20**2) - 2/z20**2 
print(f"Input Function in the Block 3, 1st Dimension, Layer 23: {d_f_z23} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 23: {d_d_f_z23} ")

z24=sm.symbols('z24')
f_z24=2*z24*(sm.tan(z24**2)**2 + 1) + 2*sm.cosh(2*z24)#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z24=sm.diff(f_z24)# -4*z20**2*sin(z20**2) + 2*cos(z20**2) + 2/z20
d_d_f_z24=sm.diff(d_f_z24)# -8*z20**3*cos(z20**2) - 12*z20*sin(z20**2) - 2/z20**2 
print(f"Input Function in the Block 3, 1st Dimension, Layer 24: {d_f_z24} ")
print(f"Output Function in the Block 2, 3rd Dimension, Layer 24: {d_d_f_z24} ")

z25=sm.symbols('z25')
f_z25=10*z25**4*sm.sinh(2*z25**5) + 4*z25**3*sm.exp(z25**4) + 5/z25#sm.exp(sm.sinh(z11**2)) + sm.tanh(z11**3)
d_f_z25=sm.diff(f_z25)# -4*z20**2*sin(z20**2) + 2*cos(z20**2) + 2/z20
d_d_f_z25=sm.diff(d_f_z25)# -8*z20**3*cos(z20**2) - 12*z20*sin(z20**2) - 2/z20**2 
print(f"Input Function in the Block 3, 1st Dimension, Layer 25: {d_f_z25} ")
print(f"Output Function in the Block 2, 3rd Dimension, Lazer 25: {d_d_f_z25} ")
