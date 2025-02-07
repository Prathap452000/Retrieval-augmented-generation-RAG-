import scipy.special as sp
import sympy as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols,diff

r1=sm.symbols('r1')
f_r1= sm.exp(r1 + 3) + sm.sin(2 * r1) + sm.sqrt(r1 + 1) + sm.log(r1**2 + 3)
d_f_r1=sm.diff(f_r1) # 2*r1/(r1**2 + 3) + exp(r1 + 3) + 2*cos(2*r1) + 1/(2*sqrt(r1 + 1))
d_d_f_r1=sm.diff(d_f_r1) # -4*r1**2/(r1**2 + 3)**2 + exp(r1 + 3) - 4*sin(2*r1) + 2/(r1**2 + 3) - 1/(4*(r1 + 1)**(3/2))
print(f'Input function ,5th Block, (Bottom Crest), Layer 1: {d_f_r1}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 1: {d_d_f_r1}')

r2=sm.symbols('r2')
f_r2= sm.sqrt(r2**3 + 1) + sm.exp(r2**2) + sm.log(2 * r2) + sm.tan(3 * r2)
d_f_r2=sm.diff(f_r2) # 3*r2**2/(2*sqrt(r2**3 + 1)) + 2*r2*exp(r2**2) + 3*tan(3*r2)**2 + 3 + 1/r2
d_d_f_r2=sm.diff(d_f_r2) # -9*r2**4/(4*(r2**3 + 1)**(3/2)) + 4*r2**2*exp(r2**2) + 3*r2/sqrt(r2**3 + 1) + 3*(6*tan(3*r2)**2 + 6)*tan(3*r2) + 2*exp(r2**2) - 1/r2**2
print(f'Input function ,5th Block,(Bottom Crest) , Layer 2: {d_f_r2}')
print(f'Output function ,5th Block, (Bottom Crest) , Layer 2: {d_d_f_r2}')

r3=sm.symbols('r3')
f_r3= sm.exp(2 * r3) + sm.log(r3 + 3) + sm.sin(r3**2) + sm.cosh(r3*3)
d_f_r3=sm.diff(f_r3) # 2*r3*cos(r3**2) + 2*exp(2*r3) + 3*sinh(3*r3) + 1/(r3 + 3)
d_d_f_r3=sm.diff(d_f_r3) # -4*r3**2*sin(r3**2) + 4*exp(2*r3) + 2*cos(r3**2) + 9*cosh(3*r3) - 1/(r3 + 3)**2
print(f'Input function ,5th Block,(Bottom Crest) , Layer 3: {d_f_r3}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 3: {d_d_f_r3}')

r4=sm.symbols('r4')
f_r4= sm.tan(r4 + 3) + sm.exp(r4**2) + sm.sqrt(3 * r4 + 1) + sm.log(2 * r4)
d_f_r4=sm.diff(f_r4) # 2*r4*exp(r4**2) + tan(r4 + 3)**2 + 1 + 3/(2*sqrt(3*r4 + 1)) + 1/r4
d_d_f_r4=sm.diff(d_f_r4) # 4*r4**2*exp(r4**2) + (2*tan(r4 + 3)**2 + 2)*tan(r4 + 3) + 2*exp(r4**2) - 9/(4*(3*r4 + 1)**(3/2)) - 1/r4**2
print(f'Input function ,5th Block, (Bottom Crest) , Layer 4: {d_f_r4}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 4: {d_d_f_r4}')

r5=sm.symbols('r5')
f_r5= sm.log(r5**2 + 1) + sm.cosh(3 * r5) + sm.exp(r5 + 2) + sm.sin(r5**3)
d_f_r5=sm.diff(f_r5) # 3*r5**2*cos(r5**3) + 2*r5/(r5**2 + 1) + exp(r5 + 2) + 3*sinh(3*r5)
d_d_f_r5=sm.diff(d_f_r5) # -9*r5**4*sin(r5**3) - 4*r5**2/(r5**2 + 1)**2 + 6*r5*cos(r5**3) + exp(r5 + 2) + 9*cosh(3*r5) + 2/(r5**2 + 1)
print(f'Input function ,5th Block, (Bottom Crest), Layer 5: {d_f_r5}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 5: {d_d_f_r5}')

r6=sm.symbols('r6')
f_r6= sm.sqrt(r6**3) + sm.exp(3 * r6) + sm.log(2 * r6 + 1) + sm.sin(r6**2) 
d_f_r6=sm.diff(f_r6) # 2*r6*cos(r6**2) + 3*exp(3*r6) + 2/(2*r6 + 1) + 3*sqrt(r6**3)/(2*r6)
d_d_f_r6=sm.diff(d_f_r6) # -4*r6**2*sin(r6**2) + 9*exp(3*r6) + 2*cos(r6**2) - 4/(2*r6 + 1)**2 + 3*sqrt(r6**3)/(4*r6**2)
print(f'Input function ,5th Block, (Bottom Crest), Layer 6: {d_f_r6}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 6: {d_d_f_r6}')

r7=sm.symbols('r7')
f_r7= sm.cosh(r7**2) + sm.exp(r7**3) + sm.log(3 * r7) + sm.tan(2 * r7 + 1)
d_f_r7=sm.diff(f_r7) # 3*r7**2*exp(r7**3) + 2*r7*sinh(r7**2) + 2*tan(2*r7 + 1)**2 + 2 + 1/r7
d_d_f_r7=sm.diff(d_f_r7) # 9*r7**4*exp(r7**3) + 4*r7**2*cosh(r7**2) + 6*r7*exp(r7**3) + 2*(4*tan(2*r7 + 1)**2 + 4)*tan(2*r7 + 1) + 2*sinh(r7**2) - 1/r7**2
print(f'Input function ,5th Block, (Bottom Crest),Layer 7: {d_f_r7}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 7: {d_d_f_r7}')

r8= sm.symbols('r8')
f_r8= sm.exp(2 * r8) + sm.sin(r8**2) + sm.sqrt(r8**3) + sm.log(3 * r8 + 1)
d_f_r8=sm.diff(f_r8) # 2*r8*cos(r8**2) + 2*exp(2*r8) + 3/(3*r8 + 1) + 3*sqrt(r8**3)/(2*r8)
d_d_f_r8=sm.diff(d_f_r8) # -4*r8**2*sin(r8**2) + 4*exp(2*r8) + 2*cos(r8**2) - 9/(3*r8 + 1)**2 + 3*sqrt(r8**3)/(4*r8**2)
print(f'Input function ,5th Block, (Bottom Crest), Layer 8: {d_f_r8}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 8: {d_d_f_r8}')

r9=sm.symbols('r9')
f_r9= sm.sqrt(2 * r9 + 1) + sm.exp(r9 + 3) + sm.log(r9**3) + sm.cosh(r9**2)
d_f_r9=sm.diff(f_r9) # 2*r9*sinh(r9**2) + exp(r9 + 3) + 1/sqrt(2*r9 + 1) + 3/r9
d_d_f_r9=sm.diff(d_f_r9) # 4*r9**2*cosh(r9**2) + exp(r9 + 3) + 2*sinh(r9**2) - 1/(2*r9 + 1)**(3/2) - 3/r9**2
print(f'Input function ,5th Block, (Bottom Crest), Layer 9: {d_f_r9}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 9: {d_d_f_r9}')

r10=sm.symbols('r10')
f_r10= sm.exp(r10**3) + sm.tan(r10**2) + sm.log(r10 + 1) + sm.sqrt(3 * r10)
d_f_r10=sm.diff(f_r10) # 3*r10**2*exp(r10**3) + 2*r10*(tan(r10**2)**2 + 1) + 1/(r10 + 1) + sqrt(3)/(2*sqrt(r10))
d_d_f_r10=sm.diff(d_f_r10) # 9*r10**4*exp(r10**3) + 8*r10**2*(tan(r10**2)**2 + 1)*tan(r10**2) + 6*r10*exp(r10**3) + 2*tan(r10**2)**2 + 2 - 1/(r10 + 1)**2 - sqrt(3)/(4*r10**(3/2))
print(f'Input function ,5th Block, (Bottom Crest), Layer 10: {d_f_r10}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 10: {d_d_f_r10}')

r11=sm.symbols('r11')
f_r11= sm.cosh(r11**2) + sm.exp(3 * r11) + sm.log(r11 + 1) + sm.sin(r11**3 + 2)
d_f_r11=sm.diff(f_r11) # 3*r11**2*cos(r11**3 + 2) + 2*r11*sinh(r11**2) + 3*exp(3*r11) + 1/(r11 + 1)
d_d_f_r11=sm.diff(d_f_r11) # -9*r11**4*sin(r11**3 + 2) + 4*r11**2*cosh(r11**2) + 6*r11*cos(r11**3 + 2) + 9*exp(3*r11) + 2*sinh(r11**2) - 1/(r11 + 1)**2
print(f'Input function ,5th Block, (Bottom Crest), Layer 11: {d_f_r11}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 11: {d_d_f_r11}')

r12=sm.symbols('r12')
f_r12= sm.exp(2 * r12**3) + sm.sin(r12 + 1) + sm.log(r12**2 + 1) + sm.sqrt(3 * r12)
d_f_r12=sm.diff(f_r12) # 6*r12**2*exp(2*r12**3) + 2*r12/(r12**2 + 1) + cos(r12 + 1) + sqrt(3)/(2*sqrt(r12))
d_d_f_r12=sm.diff(d_f_r12) # 36*r12**4*exp(2*r12**3) - 4*r12**2/(r12**2 + 1)**2 + 12*r12*exp(2*r12**3) - sin(r12 + 1) + 2/(r12**2 + 1) - sqrt(3)/(4*r12**(3/2))
print(f'Input function ,5th Block, (Bottom Crest), Layer 12: {d_f_r12}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 12: {d_d_f_r12}')

r13=sm.symbols('r13')
f_r13= sm.tan(r13**3) + sm.cosh(r13**2) + sm.exp(r13 + 2) + sm.log(r13 + 3)
d_f_r13=sm.diff(f_r13) # 
d_d_f_r13=sm.diff(d_f_r13) # 
print(f'Input function ,5th Block,(Bottom Crest), Layer 13: {d_f_r13}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 13: {d_d_f_r13}')


r14=sm.symbols('r14')
f_r14= sm.sqrt(2 * r14) + sm.exp(3 * r14**2) + sm.sin(r14**3 + 1) + sm.log(3 * r14)
d_f_r14=sm.diff(f_r14) # 
d_d_f_r14=sm.diff(d_f_r14) # 
print(f'Input function ,5th Block, (Bottom Crest), Layer 14: {d_f_r14}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 14: {d_d_f_r14}')

r15=sm.symbols('r15')
f_r15= sm.exp(r15**2 + 1) + sm.tanh(2 * r15**3) + sm.log(r15**3) + sm.sqrt(r15 + 1)
d_f_r15=sm.diff(f_r15) # 
d_d_f_r15=sm.diff(d_f_r15) # 
print(f'Input function ,5th Block, (Bottom Crest), Layer 15: {d_f_r15}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 15: {d_d_f_r15}')

r16=sm.symbols('r16')
f_r16= sm.sin(r16**3) + sm.exp(3 * r16**2) + sm.cosh(r16 + 2) + sm.log(r16 + 1)
d_f_r16=sm.diff(f_r16) # 
d_d_f_r16=sm.diff(d_f_r16) # 
print(f'Input function ,5th Block, (Bottom Crest), Layer 16: {d_f_r16}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 16: {d_d_f_r16}')

r17=sm.symbols('r17')
f_r17= sm.log(2 * r17 + 3) + sm.exp(r17**3) + sm.sqrt(r17 + 2) + sm.sin(3 * r17)
d_f_r17=sm.diff(f_r17) # 
d_d_f_r17=sm.diff(d_f_r17) # 
print(f'Input function ,5th Block, (Bottom Crest), Layer 17: {d_f_r17}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 17: {d_d_f_r17}')

r18=sm.symbols('r18')
f_r18= sm.cosh(2 * r18**3) + sm.exp(r18 + 2) + sm.log(r18 + 1) + sm.tan(3 * r18)
d_f_r18=sm.diff(f_r18) # 
d_d_f_r18=sm.diff(d_f_r18) # 
print(f'Input function ,5th Block, (Bottom Crest), Layer 18:{d_f_r18}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 18: {d_d_f_r18}')

r19=sm.symbols('r19')
f_r19= sm.exp(3 * r19) + sm.sin(r19 + 3) + sm.sqrt(r19 + 2) + sm.log(3 * r19**2 + 1)
d_f_r19=sm.diff(f_r19) # 
d_d_f_r19=sm.diff(d_f_r19) # 
print(f'Input function ,5th Block, (Bottom Crest), Layer 19: {d_f_r19}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 19: {d_d_f_r19}')

r20=sm.symbols('r20')
f_r20= sm.tan(r20**2) + sm.exp(2 * r20**3) + sm.cosh(r20 + 1) + sm.log(r20**3)
d_f_r20=sm.diff(f_r20) # 
d_d_f_r20=sm.diff(d_f_r20) #
print(f'Input function ,5th Block, (Bottom Crest), Layer 20: {d_f_r20}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 20: {d_d_f_r20}')

r21=sm.symbols('r21')
f_r21= sm.log(r21**2) + sm.exp(2 * r21 + 3) + sm.sin(r21 + 3) + sm.sqrt(3 * r21 + 1)
d_f_r21=sm.diff(f_r21) # )
d_d_f_r21=sm.diff(d_f_r21) # 
print(f'Input function ,5th Block, (Bottom Crest), Layer 21: {d_f_r21}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 21: {d_d_f_r21}')

r22=sm.symbols('r22')
f_r22= sm.cosh(r22 + 3) + sm.exp(r22 + 2) + sm.sqrt(r22**2) + sm.log(r22 + 1)
d_f_r22=sm.diff(f_r22) # 
d_d_f_r22=sm.diff(d_f_r22) # 
print(f'Input function ,5th Block, (Bottom Crest), Layer 22: {d_f_r22}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 22: {d_d_f_r22}')

r23=sm.symbols('r23')
f_r23= sm.sin(3 * r23**2) + sm.exp(r23**3) + sm.log(r23 + 2) + sm.sqrt(2 * r23 + 1)
d_f_r23=sm.diff(f_r23) # 
d_d_f_r23=sm.diff(d_f_r23) #
print(f'Input function 5th Block, (Bottom Crest), Layer 23: {d_f_r23}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 23: {d_d_f_r23}')

r24=sm.symbols('r24')
f_r24= sm.exp(r24**3) + sm.cosh(2 * r24) + sm.tan(r24**2 + 1) + sm.log(3 * r24)
d_f_r24=sm.diff(f_r24) # 
d_d_f_r24=sm.diff(d_f_r24) # 
print(f'Input function ,5th Block, (Bottom Crest), Layer 24: {d_f_r24}')
print(f'Output function ,5th Block, (Bottom Crest), Layer 24: {d_d_f_r24}')