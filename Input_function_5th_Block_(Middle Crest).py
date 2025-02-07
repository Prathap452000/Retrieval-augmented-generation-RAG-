import scipy.special as sp
import sympy as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols,diff

s1=sm.symbols('s1')
f_s1= sm.exp(s1 + 3) + sm.sin(2 * s1) + sm.sqrt(s1 + 1) + sm.log(s1**2 + 3)
d_f_s1=sm.diff(f_s1) # 2*s1/(s1**2 + 3) + exp(s1 + 3) + 2*cos(2*s1) + 1/(2*sqrt(s1 + 1))
d_d_f_s1=sm.diff(d_f_s1) # -4*s1**2/(s1**2 + 3)**2 + exp(s1 + 3) - 4*sin(2*s1) + 2/(s1**2 + 3) - 1/(4*(s1 + 1)**(3/2))
print(f'Input function ,5th Block, (Middle Crest), Layer 1: {d_f_s1}')
print(f'Output function ,5th Block, (Middle Crest), Layer 1: {d_d_f_s1}')

s2=sm.symbols('s2')
f_s2= sm.cosh(s2**2) + sm.exp(s2 + 1) + sm.log(3 * s2 + 1) + sm.tanh(s2**3)
d_f_s2=sm.diff(f_s2) # 3*s2**2*(1 - tanh(s2**3)**2) + 2*s2*sinh(s2**2) + exp(s2 + 1) + 3/(3*s2 + 1)
d_d_f_s2=sm.diff(d_f_s2) # -18*s2**4*(1 - tanh(s2**3)**2)*tanh(s2**3) + 4*s2**2*cosh(s2**2) + 6*s2*(1 - tanh(s2**3)**2) + exp(s2 + 1) + 2*sinh(s2**2) - 9/(3*s2 + 1)**2
print(f'Input function ,5th Block,(Middle Crest), Layer 2: {d_f_s2}')
print(f'Output function ,5th Block, (Middle Crest), Layer 2: {d_d_f_s2}')

s3=sm.symbols('s3')
f_s3= sm.sin(s3**3) + sm.sqrt(s3 + 2) + sm.exp(2 * s3) + sm.log(1 + 3 * s3)
d_f_s3=sm.diff(f_s3) # 3*s3**2*cos(s3**3) + 2*exp(2*s3) + 3/(3*s3 + 1) + 1/(2*sqrt(s3 + 2))
d_d_f_s3=sm.diff(d_f_s3) # -9*s3**4*sin(s3**3) + 6*s3*cos(s3**3) + 4*exp(2*s3) - 9/(3*s3 + 1)**2 - 1/(4*(s3 + 2)**(3/2))
print(f'Input function ,5th Block,(Middle Crest), Layer 3: {d_f_s3}')
print(f'Output function ,5th Block, (Middle Crest), Layer 3: {d_d_f_s3}')

s4=sm.symbols('s4')
f_s4= sm.tan(s4**2) + sm.cosh(3 * s4) + sm.exp(s4**3) + sm.log(2 * s4 + 1)
d_f_s4=sm.diff(f_s4) # 3*s4**2*exp(s4**3) + 2*s4*(tan(s4**2)**2 + 1) + 3*sinh(3*s4) + 2/(2*s4 + 1)
d_d_f_s4=sm.diff(d_f_s4) # 9*s4**4*exp(s4**3) + 8*s4**2*(tan(s4**2)**2 + 1)*tan(s4**2) + 6*s4*exp(s4**3) + 2*tan(s4**2)**2 + 9*cosh(3*s4) + 2 - 4/(2*s4 + 1)**2
print(f'Input function ,5th Block, (Middle Crest), Layer 4: {d_f_s4}')
print(f'Output function ,5th Block, (Middle Crest), Layer 4: {d_d_f_s4}')

s5=sm.symbols('s5')
f_s5= sm.exp(2 * s5) + sm.sin(s5**2) + sm.sqrt(3 * s5) + sm.log(s5 + 1)
d_f_s5=sm.diff(f_s5) # 2*s5*cos(s5**2) + 2*exp(2*s5) + 1/(s5 + 1) + sqrt(3)/(2*sqrt(s5))
d_d_f_s5=sm.diff(d_f_s5) # -4*s5**2*sin(s5**2) + 4*exp(2*s5) + 2*cos(s5**2) - 1/(s5 + 1)**2 - sqrt(3)/(4*s5**(3/2))
print(f'Input function ,5th Block, (Middle Crest), Layer 5: {d_f_s5}')
print(f'Output function ,5th Block, (Middle Crest), Layer 5: {d_d_f_s5}')

s6=sm.symbols('s6')
f_s6= sm.log(s6**3 + 2) + sm.cosh(s6**2) + sm.exp(2 * s6) + sm.sin(3 * s6)
d_f_s6=sm.diff(f_s6) # 3*s6**2/(s6**3 + 2) + 2*s6*sinh(s6**2) + 2*exp(2*s6) + 3*cos(3*s6)
d_d_f_s6=sm.diff(d_f_s6) # -9*s6**4/(s6**3 + 2)**2 + 4*s6**2*cosh(s6**2) + 6*s6/(s6**3 + 2) + 4*exp(2*s6) - 9*sin(3*s6) + 2*sinh(s6**2)
print(f'Input function ,5th Block, (Middle Crest), Layer 6: {d_f_s6}')
print(f'Output function ,5th Block, (Middle Crest), Layer 6: {d_d_f_s6}')

s7=sm.symbols('s7')
f_s7= sm.sqrt(s7**2) + sm.exp(3 * s7) + sm.sin(s7**3) + sm.log(s7 + 1)
d_f_s7=sm.diff(f_s7) # 3*s7**2*cos(s7**3) + 3*exp(3*s7) + 1/(s7 + 1) + sqrt(s7**2)/s7
d_d_f_s7=sm.diff(d_f_s7) # -9*s7**4*sin(s7**3) + 6*s7*cos(s7**3) + 9*exp(3*s7) - 1/(s7 + 1)**2 
print(f'Input function ,5th Block, (Middle Crest),Layer 7: {d_f_s7}')
print(f'Output function ,5th Block, (Middle Crest), Layer 7: {d_d_f_s7}')

s8= sm.symbols('s8')
f_s8= sm.tanh(s8**3) + sm.exp(s8**2) + sm.log(3 * s8 + 1) + sm.sqrt(2 * s8)
d_f_s8=sm.diff(f_s8) # 3*s8**2*(1 - tanh(s8**3)**2) + 2*s8*exp(s8**2) + 3/(3*s8 + 1) + sqrt(2)/(2*sqrt(s8))
d_d_f_s8=sm.diff(d_f_s8) # -18*s8**4*(1 - tanh(s8**3)**2)*tanh(s8**3) + 4*s 8**2*exp(s8**2) + 6*s8*(1 - tanh(s8**3)**2) + 2*exp(s8**2) - 9/(3*s8 + 1)**2 - sqrt(2)/(4*s8**(3/2)
print(f'Input function ,5th Block, (Middle Crest), Layer 8: {d_f_s8}')
print(f'Output function ,5th Block, (Middle Crest), Layer 8: {d_d_f_s8}')

s9=sm.symbols('s9')
f_s9= sm.exp(s9**3) + sm.sin(2 * s9) + sm.cosh(s9**2) + sm.log(s9 + 3)
d_f_s9=sm.diff(f_s9) # 3*s9**2*exp(s9**3) + 2*s9*sinh(s9**2) + 2*cos(2*s9) + 1/(s9 + 3)
d_d_f_s9=sm.diff(d_f_s9) # 9*s9**4*exp(s9**3) + 4*s9**2*cosh(s9**2) + 6*s9*exp(s9**3) - 4*sin(2*s9) + 2*sinh(s9**2) - 1/(s9 + 3)**2
print(f'Input function ,5th Block, (Middle Crest), Layer 9: {d_f_s9}')
print(f'Output function ,5th Block, (Middle Crest), Layer 9: {d_d_f_s9}')

s10=sm.symbols('s10')
f_s10= sm.sqrt(3 * s10) + sm.exp(s10 + 1) + sm.sin(s10**2) + sm.log(s10**3 + 1)
d_f_s10=sm.diff(f_s10) # 3*s10**2/(s10**3 + 1) + 2*s10*cos(s10**2) + exp(s10 + 1) + sqrt(3)/(2*sqrt(s10))
d_d_f_s10=sm.diff(d_f_s10) # -9*s10**4/(s10**3 + 1)**2 - 4*s10**2*sin(s10**2) + 6*s10/(s10**3 + 1) + exp(s10 + 1) + 2*cos(s10**2) - sqrt(3)/(4*s10**(3/2))
print(f'Input function ,5th Block, (Middle Crest), Layer 10: {d_f_s10}')
print(f'Output function ,5th Block, (Middle Crest), Layer 10: {d_d_f_s10}')

s11=sm.symbols('s11')
f_s11= sm.cosh(s11**3) + sm.tanh(2 * s11) + sm.exp(s11*2) + sm.log(1 + s11)
d_f_s11=sm.diff(f_s11) # 3*s11**2*sinh(s11**3) + 2*exp(2*s11) - 2*tanh(2*s11)**2 + 2 + 1/(s11 + 1)
d_d_f_s11=sm.diff(d_f_s11) # 9*s11**4*cosh(s11**3) + 6*s11*sinh(s11**3) - 2*(4 - 4*tanh(2*s11)**2)*tanh(2*s11) + 4*exp(2*s11) - 1/(s11 + 1)**2
print(f'Input function ,5th Block, (Middle Crest), Layer 11: {d_f_s11}')
print(f'Output function ,5th Block, (Middle Crest), Layer 11: {d_d_f_s11}')

s12=sm.symbols('s12')
f_s12= sm.log(s12**2 + 3) + sm.sin(3 * s12) + sm.sqrt(s12 + 1) + sm.exp(s12**3)
d_f_s12=sm.diff(f_s12) # 3*s12**2*exp(s12**3) + 2*s12/(s12**2 + 3) + 3*cos(3*s12) + 1/(2*sqrt(s12 + 1))
d_d_f_s12=sm.diff(d_f_s12) # 9*s12**4*exp(s12**3) - 4*s12**2/(s12**2 + 3)**2 + 6*s12*exp(s12**3) - 9*sin(3*s12) + 2/(s12**2 + 3) - 1/(4*(s12 + 1)**(3/2))
print(f'Input function ,5th Block, (Middle Crest), Layer 12: {d_f_s12}')
print(f'Output function ,5th Block, (Middle Crest), Layer 12: {d_d_f_s12}')

s13=sm.symbols('s13')
f_s13= sm.exp(s13**3) + sm.tan(2 * s13) + sm.sqrt(s13**2) + sm.log(3 * s13 + 1)
d_f_s13=sm.diff(f_s13) # 3*s13**2*exp(s13**3) + 2*tan(2*s13)**2 + 2 + 3/(3*s13 + 1) + sqrt(s13**2)/s13
d_d_f_s13=sm.diff(d_f_s13) # 9*s13**4*exp(s13**3) + 6*s13*exp(s13**3) + 2*(4*tan(2*s13)**2 + 4)*tan(2*s13) - 9/(3*s13 + 1)**2
print(f'Input function ,5th Block,(Middle Crest), Layer 13: {d_f_s13}')
print(f'Output function ,5th Block, (Middle Crest), Layer 13: {d_d_f_s13}')

s14=sm.symbols('s14')
f_s14= sm.sqrt(s14**3) + sm.sin(s14) + sm.cosh(s14**2) + sm.exp(2 * s14 + 1)
d_f_s14=sm.diff(f_s14) # 2*s14*sinh(s14**2) + 2*exp(2*s14 + 1) + cos(s14) + 3*sqrt(s14**3)/(2*s14)
d_d_f_s14=sm.diff(d_f_s14) # 4*s14**2*cosh(s14**2) + 4*exp(2*s14 + 1) - sin(s14) + 2*sinh(s14**2) + 3*sqrt(s14**3)/(4*s14**2)
print(f'Input function ,5th Block, (Middle Crest), Layer 14: {d_f_s14}')
print(f'Output function ,5th Block, (Middle Crest), Layer 14: {d_d_f_s14}')

s15=sm.symbols('s15')
f_s15= sm.tan(s15**2) + sm.exp(s15**3) + sm.log(s15 + 1) + sm.sqrt(2 * s15 + 3)
d_f_s15=sm.diff(f_s15) # 3*s15**2*exp(s15**3) + 2*s15*(tan(s15**2)**2 + 1) + 1/sqrt(2*s15 + 3) + 1/(s15 + 1)
d_d_f_s15=sm.diff(d_f_s15) # 9*s15**4*exp(s15**3) + 8*s15**2*(tan(s15**2)**2 + 1)*tan(s15**2) + 6*s15*exp(s15**3) + 2*tan(s15**2)**2 + 2 - 1/(2*s15 + 3)**(3/2) - 1/(s15 + 1)**2
print(f'Input function ,5th Block, (Middle Crest), Layer 15: {d_f_s15}')
print(f'Output function ,5th Block, (Middle Crest), Layer 15: {d_d_f_s15}')

s16=sm.symbols('s16')
f_s16= sm.sin(s16**2) + sm.cosh(3 * s16) + sm.exp(s16 + 1) + sm.log(2 * s16**3)
d_f_s16=sm.diff(f_s16) # 2*s16*cos(s16**2) + exp(s16 + 1) + 3*sinh(3*s16) + 3/s16
d_d_f_s16=sm.diff(d_f_s16) # -4*s16**2*sin(s16**2) + exp(s16 + 1) + 2*cos(s16**2) + 9*cosh(3*s16) - 3/s16**2
print(f'Input function ,5th Block, (Middle Crest), Layer 16: {d_f_s16}')
print(f'Output function ,5th Block, (Middle Crest), Layer 16: {d_d_f_s16}')

s17=sm.symbols('p17')
f_s17= sm.exp(2 * s17) + sm.log(s17**3 + 1) + sm.tan(3 * s17) + sm.sqrt(s17 + 1)
d_f_s17=sm.diff(f_s17) # 3*p17**2/(p17**3 + 1) + 2*exp(2*p17) + 3*tan(3*p17)**2 + 3 + 1/(2*sqrt(p17 + 1))
d_d_f_s17=sm.diff(d_f_s17) # -9*p17**4/(p17**3 + 1)**2 + 6*p17/(p17**3 + 1) + 3*(6*tan(3*p17)**2 + 6)*tan(3*p17) + 4*exp(2*p17) - 1/(4*(p17 + 1)**(3/2))
print(f'Input function ,5th Block, (Middle Crest), Layer 17: {d_f_s17}')
print(f'Output function ,5th Block, (Middle Crest), Layer 17: {d_d_f_s17}')

s18=sm.symbols('s18')
f_s18= sm.cosh(s18**2) + sm.exp(3 * s18 + 1) + sm.sin(s18**3) + sm.log(s18 + 2)
d_f_s18=sm.diff(f_s18) # 3*s18**2*cos(s18**3) + 2*s18*sinh(s18**2) + 3*exp(3*s18 + 1) + 1/(s18 + 2)
d_d_f_s18=sm.diff(d_f_s18) # -9*s18**4*sin(s18**3) + 4*s18**2*cosh(s18**2) + 6*s18*cos(s18**3) + 9*exp(3*s18 + 1) + 2*sinh(s18**2) - 1/(s18 + 2)**2
print(f'Input function ,5th Block, (Middle Crest), Layer 18:{d_f_s18}')
print(f'Output function ,5th Block, (Middle Crest), Layer 18: {d_d_f_s18}')

s19=sm.symbols('s19')
f_s19= 	sm.exp(s19**2) + sm.sqrt(3 * s19) + sm.tan(s19 + 1) + sm.log(s19**3)
d_f_s19=sm.diff(f_s19) # 2*s19*exp(s19**2) + tan(s19 + 1)**2 + 1 + 3/s19 + sqrt(3)/(2*sqrt(s19))
d_d_f_s19=sm.diff(d_f_s19) # 4*s19**2*exp(s19**2) + (2*tan(s19 + 1)**2 + 2)*tan(s19 + 1) + 2*exp(s19**2) - 3/s19**2 - sqrt(3)/(4*s19**(3/2))
print(f'Input function ,5th Block, (Middle Crest), Layer 19: {d_f_s19}')
print(f'Output function ,5th Block, (Middle Crest), Layer 19: {d_d_f_s19}')

s20=sm.symbols('s20')
f_s20= sm.sin(2 * s20) + sm.cosh(s20**2) + sm.exp(s20**3) + sm.log(3 * s20 + 1)
d_f_s20=sm.diff(f_s20) # 3*s20**2*exp(s20**3) + 2*s20*sinh(s20**2) + 2*cos(2*s20) + 3/(3*s20 + 1)
d_d_f_s20=sm.diff(d_f_s20) # 9*s20**4*exp(s20**3) + 4*s20**2*cosh(s20**2) + 6*s20*exp(s20**3) - 4*sin(2*s20) + 2*sinh(s20**2) - 9/(3*s20 + 1)**2
print(f'Input function ,5th Block, (Middle Crest), Layer 20: {d_f_s20}')
print(f'Output function ,5th Block, (Middle Crest), Layer 20: {d_d_f_s20}')

import sympy as sm
import scipy.special as sp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sympy import diff, symbols

s21=sm.symbols('s21')
f_s21= sm.log(s21 + 3) + sm.exp(2 * s21**2) + sm.sin(3 * s21) + sm.sqrt(s21** 3)
d_f_s21=sm.diff(f_s21) # 4*s21*exp(2*s21**2) + 3*cos(3*s21) + 1/(s21 + 3) + 3*sqrt(s21**3)/(2*s21)
d_d_f_s21=sm.diff(d_f_s21) # 16*s21**2*exp(2*s21**2) + 4*exp(2*s21**2) - 9*sin(3*s21) - 1/(s21 + 3)**2 + 3*sqrt(s21**3)/(4*s21**2)
print(f'Input function ,5th Block, (Middle Crest), Layer 21: {d_f_s21}')
print(f'Output function ,5th Block, (Middle Crest), Layer 21: {d_d_f_s21}')

s22=sm.symbols('s22')
f_s22= sm.cosh(s22**3) + sm.exp(s22) + sm.log(s22**2 + 1) + sm.tan(2 * s22)
d_f_s22=sm.diff(f_s22) # 3*s22**2*sinh(s22**3) + 2*s22/(s22**2 + 1) + exp(s22) + 2*tan(2*s22)**2 + 2
d_d_f_s22=sm.diff(d_f_s22) # 9*s22**4*cosh(s22**3) - 4*s22**2/(s22**2 + 1)**2 + 6*s22*sinh(s22**3) + 2*(4*tan(2*s22)**2 + 4)*tan(2*s22) + exp(s22) + 2/(s22**2 + 1)
print(f'Input function ,5th Block, (Middle Crest), Layer 22: {d_f_s22}')
print(f'Output function ,5th Block, (Middle Crest), Layer 22: {d_d_f_s22}')

s23=sm.symbols('s23')
f_s23= sm.exp(s23**2) + sm.tanh(3 * s23) + sm.log(s23**3 + 2) + sm.sqrt(s23 + 1)
d_f_s23=sm.diff(f_s23) # 3*s23**2/(s23**3 + 2) + 2*s23*exp(s23**2) - 3*tanh(3*s23)**2 + 3 + 1/(2*sqrt(s23 + 1))
d_d_f_s23=sm.diff(d_f_s23) # -9*s23**4/(s23**3 + 2)**2 + 4*s23**2*exp(s23**2) + 6*s23/(s23**3 + 2) - 3*(6 - 6*tanh(3*s23)**2)*tanh(3*s23) + 2*exp(s23**2) - 1/(4*(s23 + 1)**(3/2))
print(f'Input function 5th Block, (Middle Crest), Layer 23: {d_f_s23}')
print(f'Output function ,5th Block, (Middle Crest), Layer 23: {d_d_f_s23}')

s24=sm.symbols('s24')
f_s24= sm.sin(s24*3) + sm.cosh(2 * s24) + sm.exp(s24**2) + sm.log(s24 + 3)
d_f_s24=sm.diff(f_s24) # 2*s24*exp(s24**2) + 3*cos(3*s24) + 2*sinh(2*s24) + 1/(s24 + 3)
d_d_f_s24=sm.diff(d_f_s24) # 4*s24**2*exp(s24**2) + 2*exp(s24**2) - 9*sin(3*s24) + 4*cosh(2*s24) - 1/(s24 + 3)**2
print(f'Input function ,5th Block, (Middle Crest), Layer 24: {d_f_s24}')
print(f'Output function ,5th Block, (Middle Crest), Layer 24: {d_d_f_s24}')

s25=sm.symbols('s25')
f_s25= sm.sqrt(2 * s25) + sm.exp(s25 + 1) + sm.log(s25 + 3) + sm.sin(s25**2)
d_f_s25=sm.diff(f_s25) # 2*s25*cos(s25**2) + exp(s25 + 1) + 1/(s25 + 3) + sqrt(2)/(2*sqrt(s25))
d_d_f_s25=sm.diff(d_f_s25) # -4*s25**2*sin(s25**2) + exp(s25 + 1) + 2*cos(s25**2) - 1/(s25 + 3)**2 - sqrt(2)/(4*s25**(3/2))
print(f'Input function ,5th Block, (Middle Crest), Layer 25: {d_f_s25}')
print(f'Output function ,5th Block, (Middle Crest), Layer 25: {d_d_f_s25}')

s26=sm.symbols('s26')
f_s26= sm.exp(3 * s26**2) + sm.cosh(s26 + 1) + sm.sqrt(s26 + 3) + sm.log(s26**2 + 3)
d_f_s26=sm.diff(f_s26) # 6*s26*exp(3*s26**2) + 2*s26/(s26**2 + 3) + sinh(s26 + 1) + 1/(2*sqrt(s26 + 3))
d_d_f_s26=sm.diff(d_f_s26) # 36*s26**2*exp(3*s26**2) - 4*s26**2/(s26**2 + 3)**2 + 6*exp(3*s26**2) + cosh(s26 + 1) + 2/(s26**2 + 3) - 1/(4*(s26 + 3)**(3/2)
print(f'Input function ,5th Block, (Middle Crest), Layer 26: {d_f_s26}')
print(f'Output function ,5th Block, (Middle Crest), Layer 26: {d_d_f_s26}')

s27=sm.symbols('s27')
f_s27= sm.sin(2 * s27**3) + sm.exp(s27 + 2) + sm.log(3 * s27 + 1) + sm.sqrt(s27)
d_f_s27=sm.diff(f_s27) # 6*s27**2*cos(2*s27**3) + exp(s27 + 2) + 3/(3*s27 + 1) + 1/(2*sqrt(s27))
d_d_f_s27=sm.diff(d_f_s27) # -36*s27**4*sin(2*s27**3) + 12*s27*cos(2*s27**3) + exp(s27 + 2) - 9/(3*s27 + 1)**2 - 1/(4*s27**(3/2))
print(f'Input function ,5th Block, (Middle Crest), Layer 27: {d_f_s27}')
print(f'Output function ,5th Block, (Middle Crest), Layer 27: {d_d_f_s27}')

s28=sm.symbols('s28')
f_s28= sm.exp(s28 + 3) + sm.tan(s28**2) + sm.cosh(3 * s28) + sm.log(s28 + 1)
d_f_s28=sm.diff(f_s28) # 2*s28*(tan(s28**2)**2 + 1) + exp(s28 + 3) + 3*sinh(3*s28) + 1/(s28 + 1)
d_d_f_s28=sm.diff(d_f_s28) # 8*s28**2*(tan(s28**2)**2 + 1)*tan(s28**2) + exp(s28 + 3) + 2*tan(s28**2)**2 + 9*cosh(3*s28) + 2 - 1/(s28 + 1)**2
print(f'Input function ,5th Block, (Middle Crest), Layer 28: {d_f_s28}')
print(f'Output function ,5th Block, (Middle Crest), Layer 28: {d_d_f_s28}')

s29=sm.symbols('s29')
f_s29= sm.sqrt(s29**3) + sm.exp(2 * s29) + sm.sin(3 * s29) + sm.log(s29**2 + 1)
d_f_s29=sm.diff(f_s29) # 2*s29/(s29**2 + 1) + 2*exp(2*s29) + 3*cos(3*s29) + 3*sqrt(s29**3)/(2*s29)
d_d_f_s29=sm.diff(d_f_s29) # -4*s29**2/(s29**2 + 1)**2 + 4*exp(2*s29) - 9*sin(3*s29) + 2/(s29**2 + 1) + 3*sqrt(s29**3)/(4*s29**2)
print(f'Input function ,5th Block, (Middle Crest), Layer 29: {d_f_s29}')
print(f'Output function ,5th Block, (Middle Crest), Layer 29: {d_d_f_s29}')

s30=sm.symbols('s30')
f_s30= sm.cosh(2 * s30**3) + sm.exp(s30 + 2) + sm.log(s30 + 1) + sm.sin(s30**3)
d_f_s30=sm.diff(f_s30) # 3*s30**2*cos(s30**3) + 6*s30**2*sinh(2*s30**3) + exp(s30 + 2) + 1/(s30 + 1)
d_d_f_s30=sm.diff(d_f_s30) # -9*s30**4*sin(s30**3) + 36*s30**4*cosh(2*s30**3) + 6*s30*cos(s30**3) + 12*s30*sinh(2*s30**3) + exp(s30 + 2) - 1/(s30 + 1)**2
print(f'Input function ,5th Block,(Middle Crest), Layer 30: {d_f_s30}')
print(f'Output function ,5th Block, (Middle Crest), Layer 30: {d_d_f_s30}')

