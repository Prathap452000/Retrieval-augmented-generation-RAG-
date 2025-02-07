import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sm
import scipy.special as sp
from sympy import symbols,diff
#from scipy.special import erf

t1=sm.symbols('t1')
f_t1= sm.exp(t1**4) + sm.log(t1**2 + 1) + sm.sin(3 * t1) + sm.tanh(t1 + 2)
d_f_t1=sm.diff(f_t1) # exp(t1**4) + log(t1**2 + 1) + sin(3*t1) + tanh(t1 + 2)
d_f_f_t1=sm.diff(d_f_t1) # 4*t1**+3*exp(t1**4) + 2*t1/(t1**2 + 1) + 3*cos(3*t1) - tanh(t1 + 2)**2 + 1
print(f'Input function ,4th Block, (Upper Crest), Layer 1: {f_t1}')
print(f'Output function ,4th Block, (Upper Crest), Layer 1: {d_f_t1}')

t2=sm.symbols('t2')
f_t2= sm.sqrt(t2**5 + 1) + sm.exp(2 * t2) + sm.log(3 * t2 + 1) + sm.cosh(t2 + 2)
d_f_t2=sm.diff(f_t2) # sqrt(t2**5 + 1) + exp(2*t2) + log(3*t2 + 1) + cosh(t2 + 2)
d_d_f_t2=sm.diff(d_f_t2) # 5*t2**4/(2*sqrt(t2**5 + 1)) + 2*exp(2*t2) + sinh(t2 + 2) + 3/(3*t2 + 1)
print(f'Input function ,4th Block, (Upper Crest), Layer 2: {f_t2}')
print(f'Output function ,4th Block, (Upper Crest), Layer 2: {d_f_t2}')

t3=sm.symbols('t3')
f_t3= sm.exp(2 * t3) + sm.tan(t3**3) + sm.log(2 * t3 + 3) + sm.sqrt(t3 + 4)
d_f_t3=sm.diff(f_t3) # sqrt(t3 + 4) + exp(2*t3) + log(2*t3 + 3) + tan(t3**3)
d_d_f_t3=sm.diff(d_f_t3) # 3*t3**2*(tan(t3**3)**2 + 1) + 2*exp(2*t3) + 2/(2*t3 + 3) + 1/(2*sqrt(t3 + 4))
print(f'Input function ,4th Block, (Upper Crest), Layer 3: {f_t3}')
print(f'Output function ,4th Block, (Upper Crest), Layer 3: {d_f_t3}')

t4=sm.symbols('t4')
f_t4= sm.tanh(t4**3) + sm.exp(t4 + 5) + sm.sqrt(4 * t4) + sm.log(t4 + 1)
d_f_t4=sm.diff(f_t4) # 2*sqrt(t4) + exp(t4 + 5) + log(t4 + 1) + tanh(t4**3)
d_d_f_t4=sm.diff(d_f_t4) # 3*t4**2*(1 - tanh(t4**3)**2) + exp(t4 + 5) + 1/(t4 + 1) + 1/sqrt(t4)
print(f'Input function ,4th Block, (Upper Crest), Layer 4: {f_t4}')
print(f'Output function ,4th Block, (Upper Crest), Layer 4: {d_f_t4}')

t5=sm.symbols('t5')
f_t5= sm.log(t5**3 + 2) + sm.cosh(3 * t5) + sm.exp(2 * t5 + 1) + sm.sin(t5 + 4)
d_f_t5=sm.diff(f_t5) # exp(2*t5 + 1) + log(t5**3 + 2) + sin(t5 + 4) + cosh(3*t5)
d_d_f_t5=sm.diff(d_f_t5) # 3*t5**2/(t5**3 + 2) + 2*exp(2*t5 + 1) + cos(t5 + 4) + 3*sinh(3*t5)
print(f'Input function ,4th Block, (Upper Crest), Layer 5: {f_t5}')
print(f'Output function ,4th Block, (Upper Crest), Layer 5: {d_f_t5}')

t6=sm.symbols('t6')
f_t6= sm.sqrt(t6**5) + sm.exp(4 * t6) + sm.log(2 * t6 + 1) + sm.tan(t6**2)
d_f_t6=sm.diff(f_t6) # sqrt(t6**5) + exp(4*t6) + log(2*t6 + 1) + tan(t6**2)
d_d_f_t6=sm.diff(d_f_t6) # 2*t6*(tan(t6**2)**2 + 1) + 4*exp(4*t6) + 2/(2*t6 + 1) + 5*sqrt(t6**5)/(2*t6)
print(f'Input function ,4th Block, (Upper Crest), Layer 6: {f_t6}')
print(f'Output function ,4th Block, (Upper Crest), Layer 6: {d_f_t6}')

t7=sm.symbols('t7')
f_t7= sm.cosh(t7 + 4) + sm.exp(t7**5) + sm.log(4 * t7) + sm.sin(3 * t7**3)
d_f_t7=sm.diff(f_t7) # exp(t7**5) + log(4*t7) + sin(3*t7**3) + cosh(t7 + 4)
d_d_f_t7=sm.diff(d_f_t7) # 5*t7**4*exp(t7**5) + 9*t7**2*cos(3*t7**3) + sinh(t7 + 4) + 1/t7
print(f'Input function ,4th Block, (Upper Crest), Layer 7: {f_t7}')
print(f'Output function ,4th Block, (Upper Crest), Layer 7: {d_f_t7}')

t8=sm.symbols('t8')
f_t8= sm.exp(3 * t8) + sm.tanh(t8**2) + sm.sqrt(4 * t8**5) + sm.log(2 * t8 + 1)
d_f_t8=sm.diff(f_t8) # 2*sqrt(t8**5) + exp(3*t8) + log(2*t8 + 1) + tanh(t8**2)
d_d_ft8=sm.diff(d_f_t8) # 2*t8*(1 - tanh(t8**2)**2) + 3*exp(3*t8) + 2/(2*t8 + 1) + 5*sqrt(t8**5)/t8
print(f'Input function ,4th Block, (Upper Crest), Layer 8: {f_t8}')
print(f'Output function ,4th Block, (Upper Crest), Layer 8: {d_f_t8}')

t9=sm.symbols('t9')
f_t9= sm.sqrt(3 * t9 + 1) + sm.exp(t9 + 4) + sm.log(t9**5 + 1) + sm.cosh(t9**3)
d_f_t9=sm.diff(f_t9) # sqrt(3*t9 + 1) + exp(t9 + 4) + log(t9**5 + 1) + cosh(t9**3)
d_d_f_t9=sm.diff(d_f_t9) # 5*t9**4/(t9**5 + 1) + 3*t9**2*sinh(t9**3) + exp(t9 + 4) + 3/(2*sqrt(3*t9 + 1))
print(f'Input function ,4th Block, (Upper Crest), Layer 9: {f_t9}')
print(f'Output function ,4th Block, (Upper Crest), Layer 9: {d_f_t9}')

t10=sm.symbols('t10')
f_t10= sm.exp(t10 + 5) + sm.tan(3 * t10) + sm.log(t10 + 3 + 1) + sm.sqrt(2 * t10**2)
d_f_t10=sm.diff(f_t10) # sqrt(2)*sqrt(t10**2) + exp(t10 + 5) + log(t10 + 4) + tan(3*t10)
d_d_f_t10=sm.diff(d_f_t10) # exp(t10 + 5) + 3*tan(3*t10)**2 + 3 + 1/(t10 + 4) + sqrt(2)*sqrt(t10**2)/t10
print(f'Input function ,4th Block, (Upper Crest), Layer 10: {f_t10}')
print(f'Output function ,4th Block, (Upper Crest), Layer 10: {d_f_t10}')

t11=sm.symbols('t11')
f_t11= sm.cosh(t11 + 3) + sm.exp(4 * t11) + sm.log(t11 + 2) + sm.sin(t11 + 5)
d_f_t11=sm.diff(f_t11) # 4*exp(4*t11) + cos(t11 + 5) + sinh(t11 + 3) + 1/(t11 + 2)
d_d_f_t11=sm.diff(d_f_t11) # 16*exp(4*t11) - sin(t11 + 5) + cosh(t11 + 3) - 1/(t11 + 2)**2
print(f'Input function ,4th Block, (Upper Crest), Layer 11: {d_f_t11}')
print(f'Output function ,4th Block, (Upper Crest), Layer 11: {d_d_f_t11}')

t12=sm.symbols('t12')
f_t12= sm.exp(3 * t12**4) + sm.sin(t12**3) + sm.log(t12 + 1) + sm.sqrt(4 * t12)
d_f_t12=sm.diff(f_t12) # 12*t12**3*exp(3*t12**4) + 3*t12**2*cos(t12**3) + 1/(t12 + 1) + 1/sqrt(t12)
d_d_f_t12=sm.diff(d_f_t12) # 144*t12**6*exp(3*t12**4) - 9*t12**4*sin(t12**3) + 36*t12**2*exp(3*t12**4) + 6*t12*cos(t12**3) - 1/(t12 + 1)**2 - 1/(2*t12**(3/2))
print(f'Input function ,4th Block, (Upper Crest), Layer 12: {d_f_t12}')
print(f'Output function ,4th Block, (Upper Crest), Layer 12: {d_d_f_t12}')

t13=sm.symbols('t13')
f_t13= sm.tan(t13 + 5) + sm.cosh(t13**4) + sm.exp(2 * t13 + 1) + sm.log(t13 + 3)
d_f_t13=sm.diff(f_t13) # 4*t13**3*sinh(t13**4) + 2*exp(2*t13 + 1) + tan(t13 + 5)**2 + 1 + 1/(t13 + 3)
d_d_f_t13=sm.diff(d_f_t13) # 16*t13**6*cosh(t13**4) + 12*t13**2*sinh(t13**4) + (2*tan(t13 + 5)**2 + 2)*tan(t13 + 5) + 4*exp(2*t13 + 1) - 1/(t13 + 3)**2
print(f'Input function ,4th Block, (Upper Crest), Layer 13: {d_f_t13}')
print(f'Output function ,4th Block, (Upper Crest), Layer 13: {d_d_f_t13}')

t14=sm.symbols('t14')
f_t14= sm.sqrt(4 * t14) + sm.exp(3 * t14**2) + sm.tan(t14**5) + sm.log(3 * t14**4 + 1)
d_f_t14=sm.diff(f_t14) # 5*t14**4*(tan(t14**5)**2 + 1) + 12*t14**3/(3*t14**4 + 1) + 6*t14*exp(3*t14**2) + 1/sqrt(t14)
d_d_f_t14=sm.diff(d_f_t14) # 50*t14**8*(tan(t14**5)**2 + 1)*tan(t14**5) - 144*t14**6/(3*t14**4 + 1)**2 + 20*t14**3*(tan(t14**5)**2 + 1) + 36*t14**2*exp(3*t14**2) + 36*t14**2/(3*t14**4 + 1) + 6*exp(3*t14**2) - 1/(2*t14**(3/2))
print(f'Input function ,4th Block, (Upper Crest), Layer 14: {d_f_t14}')
print(f'Output function ,4th Block, (Upper Crest), Layer 14: {d_d_f_t14}') 

t15=sm.symbols('t15')
f_t15= sm.exp(t15**3 + 1) + sm.tanh(2 * t15**5) + sm.log(t15**4) + sm.sqrt(t15 + 1)
d_f_t15=sm.diff(f_t15) # 10*t15**4*(1 - tanh(2*t15**5)**2) + 3*t15**2*exp(t15**3 + 1) + 1/(2*sqrt(t15 + 1)) + 4/t15
d_d_f_t15=sm.diff(d_f_t15) # -200*t15**8*(1 - tanh(2*t15**5)**2)*tanh(2*t15**5) + 9*t15**4*exp(t15**3 + 1) + 40*t15**3*(1 - tanh(2*t15**5)**2) + 6*t15*exp(t15**3 + 1) - 1/(4*(t15 + 1)**(3/2)) - 4/t15**2
print(f'Input function ,4th Block, (Upper Crest), Layer 15: {d_f_t15}')
print(f'Output function ,4th Block, (Upper Crest), Layer 15: {d_d_f_t15}')

t16=sm.symbols('t16')
f_t16= sm.sin(t16**5) + sm.exp(4 * t16 + 3) + sm.cosh(t16 + 2) + sm.log(t16**4 + 1)
d_f_t16=sm.diff(f_t16) # 5*t16**4*cos(t16**5) + 4*t16**3/(t16**4 + 1) + 4*exp(4*t16 + 3) + sinh(t16 + 2)
d_d_f_t16=sm.diff(d_f_t16) # -25*t16**8*sin(t16**5) - 16*t16**6/(t16**4 + 1)**2 + 20*t16**3*cos(t16**5) + 12*t16**2/(t16**4 + 1) + 16*exp(4*t16 + 3) + cosh(t16 + 2)
print(f'Input function ,4th Block, (Upper Crest), Layer 16: {d_f_t16}')
print(f'Output function ,4th Block, (Upper Crest), Layer 16: {d_d_f_t16}')

t17=sm.symbols('t17')
f_t17= sm.log(4 * t17 + 3) + sm.exp(t17**5) + sm.sqrt(t17 + 4) + sm.sin(3 * t17**3)
d_f_t17=sm.diff(f_t17) # 5*t17**4*exp(t17**5) + 9*t17**2*cos(3*t17**3) + 4/(4*t17 + 3) + 1/(2*sqrt(t17 + 4))
d_d_f_t17=sm.diff(d_f_t17) # 25*t17**8*exp(t17**5) - 81*t17**4*sin(3*t17**3) + 20*t17**3*exp(t17**5) + 18*t17*cos(3*t17**3) - 16/(4*t17 + 3)**2 - 1/(4*(t17 + 4)**(3/2))
print(f'Input function ,4th Block, (Upper Crest), Layer 17: {d_f_t17}')
print(f'Output function ,4th Block, (Upper Crest), Layer 17: {d_d_f_t17}')

t18=sm.symbols('t18')
f_t18= sm.cosh(2 * t18**5) + sm.exp(t18 + 3) + sm.log(t18 + 1) + sm.tan(4 * t18**4)
d_f_t18=sm.diff(f_t18) # 10*t18**4*sinh(2*t18**5) + 16*t18**3*(tan(4*t18**4)**2 + 1) + exp(t18 + 3) + 1/(t18 + 1)
d_d_f_t18=sm.diff(d_f_t18) # 100*t18**8*cosh(2*t18**5) + 512*t18**6*(tan(4*t18**4)**2 + 1)*tan(4*t18**4) + 40*t18**3*sinh(2*t18**5) + 48*t18**2*(tan(4*t18**4)**2 + 1) + exp(t18 + 3) - 1/(t18 + 1)**2
print(f'Input function ,4th Block, (Upper Crest), Layer 18: {d_f_t18}')
print(f'Output function ,4th Block, (Upper Crest), Layer 18: {d_d_f_t18}')

t19=sm.symbols('t19')
f_t19= sm.exp(3 * t19 + 4) + sm.sin(t19**5) + sm.sqrt(t19 + 3) + sm.log(t19**3 + 2)
d_f_t19=sm.diff(f_t19) # 5*t19**4*cos(t19**5) + 3*t19**2/(t19**3 + 2) + 3*exp(3*t19 + 4) + 1/(2*sqrt(t19 + 3))
d_d_f_t19=sm.diff(d_f_t19) # -25*t19**8*sin(t19**5) - 9*t19**4/(t19**3 + 2)**2 + 20*t19**3*cos(t19**5) + 6*t19/(t19**3 + 2) + 9*exp(3*t19 + 4) - 1/(4*(t19 + 3)**(3/2))
print(f'Input function ,4th Block, (Upper Crest), Layer 19: {d_f_t19}')
print(f'Output function ,4th Block, (Upper Crest), Layer 19: {d_d_f_t19}')

t20=sm.symbols('t20')
f_t20= 	sm.tan(t20**4) + sm.exp(2 * t20 + 5) + sm.cosh(t20 + 1) + sm.log(t20**3)
d_f_t20=sm.diff(f_t20) # 4*t20**3*(tan(t20**4)**2 + 1) + 2*exp(2*t20 + 5) + sinh(t20 + 1) + 3/t20
d_d_f_t20=sm.diff(d_f_t20) # 32*t20**6*(tan(t20**4)**2 + 1)*tan(t20**4) + 12*t20**2*(tan(t20**4)**2 + 1) + 4*exp(2*t20 + 5) + cosh(t20 + 1) - 3/t20**2
print(f'Input function ,4th Block, (Upper Crest), Layer 20: {d_f_t20}')
print(f'Output function ,4th Block, (Upper Crest), Layer 20: {d_d_f_t20}')

t21=sm.symbols('t21')   
f_t21= sm.log(t21**4) + sm.exp(3 * t21 + 4) + sm.sin(t21 + 5) + sm.sqrt(2 * t21**3 + 1)
d_f_t21=sm.diff(f_t21) # 3*t21**2/sqrt(2*t21**3 + 1) + 3*exp(3*t21 + 4) + cos(t21 + 5) + 4/t21
d_d_f_t21=sm.diff(d_f_t21) # -9*t21**4/(2*t21**3 + 1)**(3/2) + 6*t21/sqrt(2*t21**3 + 1) + 9*exp(3*t21 + 4) - sin(t21 + 5) - 4/t21**2
print(f'Input function ,4th Block, (Upper Crest), Layer 21: {d_f_t21}')
print(f'Output function ,4th Block, (Upper Crest), Layer 21: {d_d_f_t21}')

t22=sm.symbols('t22')
f_t22= sm.cosh(t22**5) + sm.exp(t22 + 2) + sm.sqrt(t22**4) + sm.log(3 * t22 + 1)
d_f_t22=sm.diff(f_t22) # 5*t22**4*sinh(t22**5) + exp(t22 + 2) + 3/(3*t22 + 1) + 2*sqrt(t22**4)/t22
d_d_f_t22=sm.diff(d_f_t22) # 25*t22**8*cosh(t22**5) + 20*t22**3*sinh(t22**5) + exp(t22 + 2) - 9/(3*t22 + 1)**2 + 2*sqrt(t22**4)/t22**2
print(f'Input function ,4th Block, (Upper Crest), Layer 22: {d_f_t22}')
print(f'Output function ,4th Block, (Upper Crest), Layer 22: {d_d_f_t22}')

t23=sm.symbols('t23')
f_t23= sm.sin(4 * t23**4) + sm.exp(t23 + 5) + sm.log(t23 + 2) + sm.sqrt(2 * t23**3)
d_f_t23=sm.diff(f_t23) # 16*t23**3*cos(4*t23**4) + exp(t23 + 5) + 1/(t23 + 2) + 3*sqrt(2)*sqrt(t23**3)/(2*t23)
d_d_f_t23=sm.diff(d_f_t23) # -256*t23**6*sin(4*t23**4) + 48*t23**2*cos(4*t23**4) + exp(t23 + 5) - 1/(t23 + 2)**2 + 3*sqrt(2)*sqrt(t23**3)/(4*t23**2) 
print(f'Input function ,4th Block, (Upper Crest), Layer 23: {d_f_t23}')
print(f'Output function ,4th Block, (Upper Crest), Layer 23: {d_d_f_t23}')

t24=sm.symbols('t24')
f_t24= sm.exp(3 * t24 + 2) + sm.sin(t24**5) + sm.log(t24 + 3) + sm.sqrt(2 * t24**4)
d_f_t24=sm.diff(f_t24) # 5*t24**4*cos(t24**5) + 3*exp(3*t24 + 2) + 1/(t24 + 3) + 2*sqrt(2)*sqrt(t24**4)/t24
d_d_f_t24=sm.diff(d_f_t24) # -25*t24**8*sin(t24**5) + 20*t24**3*cos(t24**5) + 9*exp(3*t24 + 2) - 1/(t24 + 3)**2 + 2*sqrt(2)*sqrt(t24**4)/t24**2
print(f'Input function ,4th Block, (Upper Crest), Layer 24: {d_f_t24}')
print(f'Output function ,4th Block, (Upper Crest), Layer 24: {d_d_f_t24}')

t25=sm.symbols('t25')
f_t25= sm.tan(t25**4) + sm.exp(2 * t25 + 5) + sm.cosh(t25 + 1) + sm.log(t25**3)
d_f_t25=sm.diff(f_t25) # 4*t25**3*(tan(t25**4)**2 + 1) + 2*exp(2*t25 + 5) + sinh(t25 + 1) + 3/t25
d_d_f_t25=sm.diff(d_f_t25) # 32*t25**6*(tan(t25**4)**2 + 1)*tan(t25**4) + 12*t25**2*(tan(t25**4)**2 + 1) + 4*exp(2*t25 + 5) + cosh(t25 + 1) - 3/t25**2
print(f'Input function ,4th Block, (Upper Crest), Layer 25: {d_f_t25}')
print(f'Output function ,4th Block, (Upper Crest), Layer 25: {d_d_f_t25}')

t26=sm.symbols('t26')
f_t26= sm.log(t26**4) + sm.exp(3 * t26 + 4) + sm.sin(t26 + 5) + sm.sqrt(2 * t26**3 + 1)
d_f_t26=sm.diff(f_t26) # 3*t26**2/sqrt(2*t26**3 + 1) + 3*exp(3*t26 + 4) + cos(t26 + 5) + 4/t26
d_d_f_t26=sm.diff(d_f_t26) # -9*t26**4/(2*t26**3 + 1)**(3/2) + 6*t26/sqrt(2*t26**3 + 1) + 9*exp(3*t26 + 4) - sin(t26 + 5) - 4/t26**2
print(f'Input function ,4th Block, (Upper Crest), Layer 26: {d_f_t26}')
print(f'Output function ,4th Block, (Upper Crest), Layer 26: {d_d_f_t26}')

t27=sm.symbols('t27')   
f_t27= sm.sin(t27**5 + 2) + sm.exp(t27 + 3) + sm.cosh(t27 + 4) + sm.log(4 * t27**2 + 1)
d_f_t27=sm.diff(f_t27) # 5*t27**4*cos(t27**5 + 2) + 8*t27/(4*t27**2 + 1) + exp(t27 + 3) + sinh(t27 + 4)
d_d_f_t27=sm.diff(d_f_t27) #-25*t27**8*sin(t27**5 + 2) + 20*t27**3*cos(t27**5 + 2) - 64*t27**2/(4*t27**2 + 1)**2 + exp(t27 + 3) + cosh(t27 + 4) + 8/(4*t27**2 + 1)
print(f'Input function ,4th Block, (Upper Crest), Layer 27: {d_f_t27}')
print(f'Output function ,4th Block, (Upper Crest), Layer 27: {d_d_f_t27}')

t28=sm.symbols('t28')
f_t28= sm.tanh(4 * t28) + sm.exp(t28 + 3) + sm.sqrt(t28**5 + 1) + sm.log(t28**4 + 2)
d_f_t28=sm.diff(f_t28) # 5*t28**4/(2*sqrt(t28**5 + 1)) + 4*t28**3/(t28**4 + 2) + exp(t28 + 3) - 4*tanh(4*t28)**2 + 4
d_d_f_t28=sm.diff(d_f_t28) # -25*t28**8/(4*(t28**5 + 1)**(3/2)) - 16*t28**6/(t28**4 + 2)**2 + 10*t28**3/sqrt(t28**5 + 1) + 12*t28**2/(t28**4 + 2) - 4*(8 - 8*tanh(4*t28)**2)*tanh(4*t28) + exp(t28 + 3)
print(f'Input function ,4th Block, (Upper Crest), Layer 28: {d_f_t28}')
print(f'Output function ,4th Block, (Upper Crest), Layer 28: {d_d_f_t28}')

t29=sm.symbols('t29')
f_t29= sm.exp(3 * t29**5) + sm.cosh(t29 + 4) + sm.log(t29**2 + 3) + sm.sin(4 * t29**3)
d_f_t29=sm.diff(f_t29) # 15*t29**4*exp(3*t29**5) + 12*t29**2*cos(4*t29**3) + 2*t29/(t29**2 + 3) + sinh(t29 + 4)
d_d_f_t29=sm.diff(d_f_t29) # 225*t29**8*exp(3*t29**5) - 144*t29**4*sin(4*t29**3) + 60*t29**3*exp(3*t29**5) - 4*t29**2/(t29**2 + 3)**2 + 24*t29*cos(4*t29**3) + cosh(t29 + 4) + 2/(t29**2 + 3)
print(f'Input function ,4th Block, (Upper Crest), Layer 29: {d_f_t29}')
print(f'Output function ,4th Block, (Upper Crest), Layer 29: {d_d_f_t29}')

t30=sm.symbols('t30')
f_t30= sm.sqrt(3 * t30 + 1) + sm.exp(t30 + 5) + sm.cosh(t30**4) + sm.log(4 * t30 + 2)
d_f_t30=sm.diff(f_t30) # 4*t30**3*sinh(t30**4) + exp(t30 + 5) + 4/(4*t30 + 2) + 3/(2*sqrt(3*t30 + 1))
d_d_f_t30=sm.diff(d_f_t30) # 16*t30**6*cosh(t30**4) + 12*t30**2*sinh(t30**4) + exp(t30 + 5) - 16/(4*t30 + 2)**2 - 9/(4*(3*t30 + 1)**(3/2))
print(f'Input function ,4th Block, (Upper Crest), Layer 30: {d_f_t30}')
print(f'Output function ,4th Block, (Upper Crest), Layer 30: {d_d_f_t30}')