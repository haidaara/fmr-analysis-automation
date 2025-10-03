Implementation Order:
Lorentzian derivative fitting (Step 1) - This is the foundation because we need accurate H_res and ΔH for each curve.

Kittel fitting (Step 2) - Requires H_res from multiple frequencies.

Damping extraction (Step 3) - Requires γ from Kittel fit and ΔH from Lorentzian fits.

Plotting updates (Step 4) - To visualize the fits.

Workflow integration (Step 5) - Put it all together.

Let's break down the tasks for each step:

Task 1A: Lorentzian Derivative Fitting Function
Create a function fit_lorentzian_derivative(H, signal) in processor.py.

Use scipy.optimize.curve_fit to fit the function:

f
(
H
)
=
A
(
H
−
H
r
e
s
)
[
(
H
−
H
r
e
s
)
2
+
(
Δ
H
2
)
2
]
2
f(H)= 
[(H−H 
res
​
 ) 
2
 +( 
2
ΔH
​
 ) 
2
 ] 
2
 
A(H−H 
res
​
 )
​
 
The parameters to fit are: A, H_res, ΔH.

Initial guess for H_res: the field where the signal crosses zero (if available) or the minimum of the absolute signal.

Initial guess for ΔH: the width between the maximum and minimum of the signal.

Return H_res and ΔH (and maybe the full fit parameters for plotting).

Task 1B: Update find_resonance to use the Lorentzian fit
Modify find_resonance in processor.py to use the new fitting function.

Task 2A: Create Kittel Fitting Function
Create src/physics/kittel_fitter.py with function fit_kittel(frequencies, H_res).

The Kittel equation for in-plane geometry:

f
=
γ
μ
0
2
π
H
r
e
s
(
H
r
e
s
+
M
s
)
f= 
2π
γμ 
0
​
 
​
  
H 
res
​
 (H 
res
​
 +M 
s
​
 )
​
 
We fit for γ and M_s.

Use curve_fit with the Kittel equation.

Task 3A: Create Damping Extraction Function
Create src/physics/damping_analyzer.py with function extract_damping(frequencies, linewidths, gamma).

The equation:

Δ
H
=
Δ
H
0
+
2
α
f
γ
ΔH=ΔH 
0
​
 + 
γ
2αf
​
 
We have γ from the Kittel fit, so we can compute the slope = 2α/γ.

Use linear regression (e.g., numpy.polyfit or scipy.stats.linregress).

Task 4A: Update Plotting
In plotter.py, add:

plot_lorentzian_fit

plot_kittel_fit

plot_damping_fit

Task 5A: Integrate into Workflow
In analyzer.py, after processing all curves, we have a list of curves with H_res and ΔH for each frequency.

Group by sample and frequency, then for each sample:

Collect frequencies and H_res, then fit Kittel to get γ and M_s.

Use γ and the linewidths (ΔH) vs frequency to get α and ΔH_0.

Task 6A: Unit Conversions
We'll handle units as we go, ensuring that:

Frequencies are in GHz (as in the data) but note that the Kittel equation uses Hz? We have to be consistent.

Magnetic field in Tesla.

Let's note: the Kittel equation in the poster uses:

f
=
γ
μ
0
2
π
(
H
r
e
s
+
M
e
f
f
)
H
r
e
s
f= 
2π
γμ 
0
​
 
​
  
(H 
res
​
 +M 
eff
​
 )H 
res
​
 
​
 
and they have γ/2π in GHz/T. So we can work in GHz and T.

We'll keep frequencies in GHz and fields in T.