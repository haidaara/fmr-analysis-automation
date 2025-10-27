

### used model for fitting: 
#### 1) Symmetric Lorentzian (with linear baseline)
\[
F_{\text{sym}}(H)= 
A\,\frac{\Delta H^{2}}{(H-H_{0})^{2}+\Delta H^{2}} 
\;+\; C \;+\; D\,H
\]

---

#### 2) Asymmetric Lorentzian (exactly as in the lab model)
\[
F_{\text{asym}}(H)=
A\frac{\Delta H^{2}}{(H-H_{0})^{2}+\Delta H^{2}}
\;+\;
B\frac{\Delta H\,(H-H_{\mathrm{res}})}{(H-H_{0})+\Delta H^{2}}
\;+\; C \;+\; D\,H
\]

---

#### 3) Two Asymmetric Peaks (sum of two copies of the MD line shape + one baseline)
\[
\begin{aligned}
F_{2\text{pk}}(H)=&\;
A_{1}\frac{\Delta H_{1}^{2}}{(H-H_{01})^{2}+\Delta H_{1}^{2}}
+\;
B_{1}\frac{\Delta H_{1}\,(H-H_{\mathrm{res}1})}{(H-H_{01})+\Delta H_{1}^{2}}
\\[3pt]
&+\;
A_{2}\frac{\Delta H_{2}^{2}}{(H-H_{02})^{2}+\Delta H_{2}^{2}}
+\;
B_{2}\frac{\Delta H_{2}\,(H-H_{\mathrm{res}2})}{(H-H_{02})+\Delta H_{2}^{2}}
\;+\; C \;+\; D\,H
\end{aligned}
\]


#### Kittel: 
\[
    f = γ/2π√{H(H +M)}
\]

#### damping linear:
\[ 
    ΔH = \frac{2α}{γ}f + ΔH_0    
\] 

#### α from slope: 
\[ 
    α = γ/2*slope
\]




### Initial Parameter Estimation

#### 1. Extract key extrema
- \(H_{\min},\,H_{\max}\) — field values at the local minima and maxima  
- \(y_{\min},\,y_{\max}\) — corresponding signal amplitudes  

#### 2. Center guess
\[
H_{0} = \frac{H_{\min} + H_{\max}}{2}
\]

#### 3. Width guess
From the derivative Lorentzian identity:
\[
H_{\max} - H_{\min} \approx \frac{2}{\sqrt{3}}\,\Delta H
\quad\Rightarrow\quad
\Delta H \approx \frac{\sqrt{3}}{2}\,(H_{\max} - H_{\min})
\]



### decided package is lmfit:

use lmfit library to fit data. 