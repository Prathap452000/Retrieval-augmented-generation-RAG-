import sympy as sm
import scipy.special as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff

q1=sm.symbols('q1')
f_q1=sm.exp(q1**2) + sm.sin(3 * q1) + sm.log(q1 + 1) + sm.sqrt(q1) 
d_f_q1=sm.diff(f_q1) # 2*q1*exp(q1**2) + 3*cos(3*q1) + 1/(q1 + 1) + 1/(2*sqrt(q1))
d_d_f_q1=sm.diff(d_f_q1) # 4*q1**2*exp(q1**2) + 2*exp(q1**2) - 9*sin(3*q1) - 1/(q1 + 1)**2 - 1/(4*q1**(3/2))
print(f'Input function ,5th Block, Layer 1: {d_f_q1}')
print(f'Output function ,5th Block, Layer 1: {d_d_f_q1}')

q2=sm.symbols('q2')
f_q2= sm.cosh(q2) + sm.tan(q2**3) + sm.exp(q2) + sm.sin(q2)
d_f_q2=sm.diff(f_q2) # 3*q2**2*(tan(q2**3)**2 + 1) + exp(q2) + cos(q2) + sinh(q2)
d_d_f_q2=sm.diff(d_f_q2) # 18*q2**4*(tan(q2**3)**2 + 1)*tan(q2**3) + 6*q2*(tan(q2**3)**2 + 1) + exp(q2) - sin(q2) + cosh(q2)
print(f'Input function ,5th Block, Layer 2: {d_f_q2}')
print(f'Output function ,5th Block, Layer 2: {d_d_f_q2}')

import sympy as sm
q3=sm.symbols('q3')
f_q3= sm.log(q3**2 + 1) + sm.sqrt(q3) + sm.tan(2 * q3) + sm.cosh(q3)
d_f_q3=sm.diff(f_q3) # 2*q3/(q3**2 + 1) + 2*tan(2*q3)**2 + sinh(q3) + 2 + 1/(2*sqrt(q3))
d_d_f_q3=sm.diff(d_f_q3) # -4*q3**2/(q3**2 + 1)**2 + 2*(4*tan(2*q3)**2 + 4)*tan(2*q3) + cosh(q3) + 2/(q3**2 + 1) - 1/(4*q3**(3/2))
print(f'Input function ,5th Block, Layer 3: {d_f_q3}')
print(f'Output function ,5th Block, Layer 3: {d_d_f_q3}')

q4=sm.symbols('q4')
f_q4= sm.exp(q4) + sm.sin(q4 + 2) + sm.log(2 * q4 + 3) + sm.cosh(q4 + 3)
d_f_q4=sm.diff(f_q4) # exp(q4) + cos(q4 + 2) + sinh(q4 + 3) + 2/(2*q4 + 3)
d_d_f_q4=sm.diff(d_f_q4) # exp(q4) - sin(q4 + 2) + cosh(q4 + 3) - 4/(2*q4 + 3)**2
print(f'Input function ,5th Block, Layer 5: {d_f_q4}')
print(f'Output function ,5th Block, Layer 5: {d_d_f_q4}')

q5=sm.symbols('q5')
f_q5= sm.sin(q5 + 2) + sm.sqrt(3 * q5) + sm.log(q5 + 2) + sm.exp(q5 ** 3)
d_f_q5=sm.diff(f_q5) # 3*q5**2*exp(q5**3) + cos(q5 + 2) + 1/(q5 + 2) + sqrt(3)/(2*sqrt(q5))
d_d_f_q5=sm.diff(d_f_q5) # 9*q5**4*exp(q5**3) + 6*q5*exp(q5**3) - sin(q5 + 2) - 1/(q5 + 2)**2 - sqrt(3)/(4*q5**(3/2))
print(f'Input function ,5th Block, Layer 6: {d_f_q5}')
print(f'Output function ,5th Block, Layer 6: {d_d_f_q5}')

q6=sm.symbols('q6')
f_q6= sm.log(1 + q6**3) + sm.tanh(q6) + sm.sqrt(q6 + 4) + sm.exp(2 * q6)
d_f_q6=sm.diff(f_q6) # 3*q6**2/(q6**3 + 1) + 2*exp(2*q6) - tanh(q6)**2 + 1 + 1/(2*sqrt(q6 + 4))
d_d_f_q6=sm.diff(d_f_q6) # -9*q6**4/(q6**3 + 1)**2 + 6*q6/(q6**3 + 1) - (2 - 2*tanh(q6)**2)*tanh(q6) + 4*exp(2*q6) - 1/(4*(q6 + 4)**(3/2))
print(f'Input function ,5th Block, Layer 6: {d_f_q6}')
print(f'Output function ,5th Block, Layer 6: {d_d_f_q6}')

q7=sm.symbols('q7')
f_q7= sm.sin(2 * q7) + sm.exp(q7 + 2) + sm.log(q7 + 1) + sm.sqrt(q7**3)
d_f_q7=sm.diff(f_q7) # exp(q7 + 2) + 2*cos(2*q7) + 1/(q7 + 1) + 3*sqrt(q7**3)/(2*q7)
d_d_f_q7=sm.diff(d_f_q7) # exp(q7 + 2) - 4*sin(2*q7) - 1/(q7 + 1)**2 + 3*sqrt(q7**3)/(4*q7**2)
print(f'Input function ,5th Block, Layer 7: {d_f_q7}')
print(f'Output function ,5th Block, Layer 7: {d_d_f_q7}')


q8=sm.symbols('q8')
f_q8= sm.tan(q8**2) + sm.cosh(2 * q8) + sm.sqrt(q8) + sm.log(q8 + 1) + sm.exp(q8)
d_f_q8=sm.diff(f_q8) # 2*q8*(tan(q8**2)**2 + 1) + exp(q8) + 2*sinh(2*q8) + 1/(q8 + 1) + 1/(2*sqrt(q8))
d_d_f_q8=sm.diff(d_f_q8) # 8*q8**2*(tan(q8**2)**2 + 1)*tan(q8**2) + exp(q8) + 2*tan(q8**2)**2 + 4*cosh(2*q8) + 2 - 1/(q8 + 1)**2 - 1/(4*q8**(3/2))
print(f'Input function ,5th Block, Layer 8: {d_f_q8}')
print(f'Output function ,5th Block, Layer 8: {d_d_f_q8}')

q9=sm.symbols('q9')
f_q9= sm.exp(q9**3) + sm.log(2 * q9 + 1) + sm.tan(q9**2) + sm.sin(3 * q9)
d_f_q9=sm.diff(f_q9) # 3*q9**2*exp(q9**3) + 2*q9*(tan(q9**2)**2 + 1) + 3*cos(3*q9) + 2/(2*q9 + 1)
d_d_f_q9=sm.diff(d_f_q9) # 9*q9**4*exp(q9**3) + 8*q9**2*(tan(q9**2)**2 + 1)*tan(q9**2) + 6*q9*exp(q9**3) - 9*sin(3*q9) + 2*tan(q9**2)**2 + 2 - 4/(2*q9 + 1)**2
print(f'Input function ,5th Block, Layer 9: {d_f_q9}')
print(f'Output function ,5th Block, Layer 9: {d_d_f_q9}')

q10=sm.symbols('q10')
f_q10= sm.cosh(2 * q10) + sm.exp(q10) + sm.sqrt(q10 + 1) + sm.log(q10**2 + 1)
d_f_q10=sm.diff(f_q10) # 2*q10/(q10**2 + 1) + exp(q10) + 2*sinh(2*q10) + 1/(2*sqrt(q10 + 1))
d_d_f_q10=sm.diff(d_f_q10) # -4*q10**2/(q10**2 + 1)**2 + exp(q10) + 4*cosh(2*q10) + 2/(q10**2 + 1) - 1/(4*(q10 + 1)**(3/2))
print(f'Input function ,5th Block, Layer 10: {d_f_q10}')
print(f'Output function ,5th Block, Layer 10: {d_d_f_q10}')

q11=sm.symbols('q11')
f_q11=sm.log(q11 + 2) + sm.exp(q11) + sm.sin(3 * q11) + sm.cosh(q11 + 3) 
d_f_q11=sm.diff(f_q11) # exp(q11) + 3*cos(3*q11) + sinh(q11 + 3) + 1/(q11 + 2)
d_d_f_q11=sm.diff(d_f_q11) # exp(q11) - 9*sin(3*q11) + cosh(q11 + 3) - 1/(q11 + 2)**2
print(f'Input function ,5th Block, Layer 11: {d_f_q11}')
print(f'Output function ,5th Block, Layer 11: {d_d_f_q11}')

q12=sm.symbols('q12')
f_q12=sm.sin(q12) + sm.sqrt(q12 + 2) + sm.exp(q12**2) + sm.log(q12 + 1)
d_f_q12=sm.diff(f_q12) # 2*q12*exp(q12**2) + cos(q12) + 1/(2*sqrt(q12 + 2)) + 1/(q12 + 1)
d_d_f_q12=sm.diff(d_f_q12) # 4*q12**2*exp(q12**2) + 2*exp(q12**2) - sin(q12) - 1/(4*(q12 + 2)**(3/2)) - 1/(q12 + 1)**2
print(f'Input function ,5th Block, Layer 12: {d_f_q12}')
print(f'Output function ,5th Block, Layer 12: {d_d_f_q12}')

q13=sm.symbols('q13')
f_q13= sm.exp(q13 + 3) + sm.cosh(q13) + sm.sqrt(q13 + 2) + sm.log(2 * q13 + 1)
d_f_q13=sm.diff(f_q13) # exp(q13 + 3) + sinh(q13) + 2/(2*q13 + 1) + 1/(2*sqrt(q13 + 2))
d_d_f_q13=sm.diff(d_f_q13) # exp(q13 + 3) + cosh(q13) - 4/(2*q13 + 1)**2 - 1/(4*(q13 + 2)**(3/2))
print(f'Input function ,5th Block, Layer 13: {d_f_q13}')
print(f'Output function ,5th Block, Layer 13: {d_d_f_q13}')

q14=sm.symbols('q14')
f_q14= sm.tanh(q14 + 2) + sm.sin(2 * q14) + sm.exp(q14) + sm.log(q14**2 + 3)
d_f_q14=sm.diff(f_q14) # 2*q14/(q14**2 + 3) + exp(q14) + 2*cos(2*q14) - tanh(q14 + 2)**2 + 1
d_d_f_q14=sm.diff(d_f_q14) # -4*q14**2/(q14**2 + 3)**2 - (2 - 2*tanh(q14 + 2)**2)*tanh(q14 + 2) + exp(q14) - 4*sin(2*q14) + 2/(q14**2 + 3)
print(f'Input function ,5th Block, Layer 14: {d_f_q14}')
print(f'Output function ,5th Block, Layer 14: {d_d_f_q14}')

q15=sm.symbols('q15')
f_q15= sm.cosh(q15**3) + sm.log(q15) + sm.sqrt(3 * q15) + sm.exp(q15 + 1)
d_f_q15=sm.diff(f_q15) # 3*q15**2*sinh(q15**3) + exp(q15 + 1) + 1/q15 + sqrt(3)/(2*sqrt(q15))
d_d_f_q15=sm.diff(d_f_q15) # 9*q15**4*cosh(q15**3) + 6*q15*sinh(q15**3) + exp(q15 + 1) - 1/q15**2 - sqrt(3)/(4*q15**(3/2))
print(f'Input function ,5th Block, Layer 15: {d_f_q15}')
print(f'Output function ,5th Block, Layer 15: {d_d_f_q15}')

q16=sm.symbols('q16')
f_q16= sm.exp(q16) + sm.tan(q16 * 2) + sm.log(q16 * 3) + sm.cosh(2 * q16 + 1)
d_f_q16=sm.diff(f_q16) # exp(q16) + 2*tan(2*q16)**2 + 2*sinh(2*q16 + 1) + 2 + 1/q16
d_d_f_q16=sm.diff(d_f_q16) # 2*(4*tan(2*q16)**2 + 4)*tan(2*q16) + exp(q16) + 4*cosh(2*q16 + 1) - 1/q16**2
print(f'Input function ,5th Block, Layer 16: {d_f_q16}')
print(f'Output function ,5th Block, Layer 16: {d_d_f_q16}')

q17=sm.symbols('q17')
f_q17= sm.tanh(3 * q17) + sm.exp(q17**2) + sm.log(q17 + 1) + sm.sin(q17 + 3) 
d_f_q17=sm.diff(f_q17) # 2*q17*exp(q17**2) + cos(q17 + 3) - 3*tanh(3*q17)**2 + 3 + 1/(q17 + 1)
d_d_f_q17=sm.diff(d_f_q17) # 4*q17**2*exp(q17**2) - 3*(6 - 6*tanh(3*q17)**2)*tanh(3*q17) + 2*exp(q17**2) - sin(q17 + 3) - 1/(q17 + 1)**2
print(f'Input function ,5th Block, Layer 17: {d_f_q17}')
print(f'Output function ,5th Block, Layer 17: {d_d_f_q17}')

q18=sm.symbols('q18')
f_q18= sm.sqrt(q18**2) + sm.cosh(2 * q18) + sm.exp(q18) + sm.log(3 * q18 + 1)
d_f_q18=sm.diff(f_q18) # exp(q18) + 2*sinh(2*q18) + 3/(3*q18 + 1) + sqrt(q18**2)/q18
d_d_f_q18=sm.diff(d_f_q18) # exp(q18) + 4*cosh(2*q18) - 9/(3*q18 + 1)**2
print(f'Input function ,5th Block, Layer 18:{d_f_q18}')
print(f'Output function ,5th Block, Layer 18: {d_d_f_q18}')

q19=sm.symbols('q19')
f_q19= sm.tan(q19) + sm.exp(2 * q19 + 3) + sm.log(q19 **2) + sm.cosh(q19)
d_f_q19=sm.diff(f_q19) # 2*exp(2*q19 + 3) + tan(q19)**2 + sinh(q19) + 1 + 2/q19
d_d_f_q19=sm.diff(d_f_q19) # (2*tan(q19)**2 + 2)*tan(q19) + 4*exp(2*q19 + 3) + cosh(q19) - 2/q19**2
print(f'Input function ,5th Block, Layer 19: {d_f_q19}')
print(f'Output function ,5th Block, Layer 19: {d_d_f_q19}')


q20=sm.symbols('q20')
f_q20= sm.exp(q20**3) + sm.sin(q20) + sm.sqrt(2 * q20 + 1) + sm.log(q20**2 + 1)
d_f_q20=sm.diff(f_q20) # 3*q20**2*exp(q20**3) + 2*q20/(q20**2 + 1) + cos(q20) + 1/sqrt(2*q20 + 1)
d_d_f_q20=sm.diff(d_f_q20) # 9*q20**4*exp(q20**3) - 4*q20**2/(q20**2 + 1)**2 + 6*q20*exp(q20**3) - sin(q20) + 2/(q20**2 + 1) - 1/(2*q20 + 1)**(3/2)
print(f'Input function ,5th Block, Layer 20: {d_f_q20}')
print(f'Output function ,5th Block, Layer 20: {d_d_f_q20}')

q21=sm.symbols('q21')
f_q21= sm.log(1 + q21**3) + sm.exp(2 * q21) + sm.cosh(q21) + sm.tanh(q21 + 2)
d_f_q21=sm.diff(f_q21) # 3*q21**2/(q21**3 + 1) + 2*exp(2*q21) + sinh(q21) - tanh(q21 + 2)**2 + 1
d_d_f_q21=sm.diff(d_f_q21) # -9*q21**4/(q21**3 + 1)**2 + 6*q21/(q21**3 + 1) - (2 - 2*tanh(q21 + 2)**2)*tanh(q21 + 2) + 4*exp(2*q21) + cosh(q21)
print(f'Input function ,5th Block, Layer 21: {d_f_q21}')
print(f'Output function ,5th Block, Layer 21: {d_d_f_q21}')

q22=sm.symbols('q22')
f_q22= sm.exp(q22 + 2) + sm.sin(3 * q22) + sm.log(q22 + 1) + sm.sqrt(q22**3)
d_f_q22=sm.diff(f_q22) # exp(q22 + 2) + 3*cos(3*q22) + 1/(q22 + 1) + 3*sqrt(q22**3)/(2*q22)
d_d_f_q22=sm.diff(d_f_q22) # exp(q22 + 2) - 9*sin(3*q22) - 1/(q22 + 1)**2 + 3*sqrt(q22**3)/(4*q22**2)
print(f'Input function ,5th Block, Layer 22: {d_f_q22}')
print(f'Output function ,5th Block, Layer 22: {d_d_f_q22}')

q23=sm.symbols('q23')
f_q23= sm.cosh(q23**2) + sm.tan(2 * q23) + sm.exp(q23) + sm.log(3 * q23)
d_f_q23=sm.diff(f_q23) # 2*q23*sinh(q23**2) + exp(q23) + 2*tan(2*q23)**2 + 2 + 1/q23
d_d_f_q23=sm.diff(d_f_q23) # 4*q23**2*cosh(q23**2) + 2*(4*tan(2*q23)**2 + 4)*tan(2*q23) + exp(q23) + 2*sinh(q23**2) - 1/q23**2
print(f'Input function ,5th Block, Layer 23: {d_f_q23}')
print(f'Output function ,5th Block, Layer 23: {d_d_f_q23}')

q24=sm.symbols('q24')
f_q24= sm.log(q24**3) + sm.sqrt(2 * q24) + sm.exp(q24 + 1) + sm.sin(q24 + 2)
d_f_q24=sm.diff(f_q24) # exp(q24 + 1) + cos(q24 + 2) + 3/q24 + sqrt(2)/(2*sqrt(q24))
d_d_f_q24=sm.diff(d_f_q24) # exp(q24 + 1) - sin(q24 + 2) - 3/q24**2 - sqrt(2)/(4*q24**(3/2))
print(f'Input function ,5th Block, Layer 24: {d_f_q24}')
print(f'Output function ,5th Block, Layer 24: {d_d_f_q24}')

q25=sm.symbols('p25')
f_q25= sm.tanh(q25) + sm.log(q25**2 + 3) + sm.exp(2 * q25) + sm.cosh(3 * q25 + 1)
d_f_q25=sm.diff(f_q25) # 2*p25/(p25**2 + 3) + 2*exp(2*p25) + 3*sinh(3*p25 + 1) - tanh(p25)**2 + 1
d_d_f_q25=sm.diff(d_f_q25) # -4*p25**2/(p25**2 + 3)**2 - (2 - 2*tanh(p25)**2)*tanh(p25) + 4*exp(2*p25) + 9*cosh(3*p25 + 1) + 2/(p25**2 + 3)
print(f'Input function ,5th Block, Layer 25: {d_f_q25}')
print(f'Output function ,5th Block, Layer 25: {d_d_f_q25}')

q26=sm.symbols('p26')
f_q26= sm.cosh(q26**3) + sm.exp(3 * q26) + sm.sqrt(q26 + 1) + sm.log(2 * q26)
d_f_q26=sm.diff(f_q26) # 3*p26**2*sinh(p26**3) + 3*exp(3*p26) + 1/(2*sqrt(p26 + 1)) + 1/p26
d_d_f_q26=sm.diff(d_f_q26) # 9*p26**4*cosh(p26**3) + 6*p26*sinh(p26**3) + 9*exp(3*p26) - 1/(4*(p26 + 1)**(3/2)) - 1/p26**2
print(f'Input function ,5th Block, Layer 26: {d_f_q26}')
print(f'Output function ,5th Block, Layer 26: {d_d_f_q26}')

q27=sm.symbols('q27')
f_q27= sm.exp(q27 + 2) + sm.log(q27 + 3) + sm.tan(3 * q27**3) + sm.cosh(q27)
d_f_q27=sm.diff(f_q27) # 9*q27**2*(tan(3*q27**3)**2 + 1) + exp(q27 + 2) + sinh(q27) + 1/(q27 + 3)
d_d_f_q27=sm.diff(d_f_q27) # 162*q27**4*(tan(3*q27**3)**2 + 1)*tan(3*q27**3) + 18*q27*(tan(3*q27**3)**2 + 1) + exp(q27 + 2) + cosh(q27) - 1/(q27 + 3)**2
print(f'Input function ,5th Block, Layer 27: {d_f_q27}')
print(f'Output function ,5th Block, Layer 27: {d_d_f_q27}')

q28=sm.symbols('q28')
f_q28= sm.tanh(q28**3) + sm.exp(2 * q28) + sm.log(q28**2 + 1) + sm.sqrt(q28 + 1)
d_f_q28=sm.diff(f_q28) # 3*q28**2*(1 - tanh(q28**3)**2) + 2*q28/(q28**2 + 1) + 2*exp(2*q28) + 1/(2*sqrt(q28 + 1))
d_d_f_q28=sm.diff(d_f_q28) # -18*q28**4*(1 - tanh(q28**3)**2)*tanh(q28**3) - 4*q28**2/(q28**2 + 1)**2 + 6*q28*(1 - tanh(q28**3)**2) + 4*exp(2*q28) + 2/(q28**2 + 1) - 1/(4*(q28 + 1)**(3/2))
print(f'Input function ,5th Block, Layer 28: {d_f_q28}')
print(f'Output function ,5th Block, Layer 28: {d_d_f_q28}')

q29=sm.symbols('q29')
f_q29= sm.sin(q29**2) + sm.exp(2 * q29) + sm.sqrt(q29 + 3) + sm.log(2 * q29**2 + 1)
d_f_q29=sm.diff(f_q29) # 2*q29*cos(q29**2) + 4*q29/(2*q29**2 + 1) + 2*exp(2*q29) + 1/(2*sqrt(q29 + 3))
d_d_f_q29=sm.diff(d_f_q29) # -4*q29**2*sin(q29**2) - 16*q29**2/(2*q29**2 + 1)**2 + 4*exp(2*q29) + 2*cos(q29**2) + 4/(2*q29**2 + 1) - 1/(4*(q29 + 3)**(3/2))
print(f'Input function ,5th Block, Layer 29: {d_f_q29}')
print(f'Output function ,5th Block, Layer 29: {d_d_f_q29}')

q30=sm.symbols('q30')
f_q30= sm.cosh(q30) + sm.tan(q30**3) + sm.exp(q30) + sm.log(2 * q30 + 1)
d_f_q30=sm.diff(f_q30) # 3*q30**2*(tan(q30**3)**2 + 1) + exp(q30) + sinh(q30) + 2/(2*q30 + 1)
d_d_f_q30=sm.diff(d_f_q30) # 18*q30**4*(tan(q30**3)**2 + 1)*tan(q30**3) + 6*q30*(tan(q30**3)**2 + 1) + exp(q30) + cosh(q30) - 4/(2*q30 + 1)**2
print(f'Input function ,5th Block, Layer 30: {d_f_q30}')
print(f'Output function ,5th Block, Layer 30: {d_d_f_q30}')


