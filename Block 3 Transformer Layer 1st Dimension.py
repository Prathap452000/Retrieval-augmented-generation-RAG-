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

