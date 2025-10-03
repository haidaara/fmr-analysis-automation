# Ultra-thin high-quality magnetic insulator films

**Authors:** B. El-Khoury¹, M. Haidar¹*, and S. Isber¹  
**Affiliation:** ¹ Department of Physics, American University of Beirut, P.O. Box 11-0236, Riad El-Solh  
**Contact:** *mh280@aub.edu.lb*

---

## Introductory Paragraph (top block)
Due to fundamental limitations, the miniaturization of CMOS devices becomes very difficult, and alternative concepts that allow for higher storage density at low power are required [1]. Spin waves, the elementary low-energy excitations in magnetic systems, exist in the high-frequency regime and are considered a potential technology that can complement the CMOS devices. However, the common magnetic materials, such as Nickel, iron, cobalt, and their alloys used in devices are not ideal for spin-wave propagation due to their high magnetic losses, which translated into shorter propagation lengths. The solution lies in using materials with ultralow damping, such as yttrium iron garnet (YIG) [2,3]. Typically, YIG films are prepared by liquid phase epitaxy (LPE) with a thickness range of tens of microns which is not ideal for applications. Here, we prepared ultra-thin YIG films with a thickness down to 30 nm by the pulsed laser deposition technique. We study the static and dynamic properties of these films using a broadband ferromagnetic resonance (FMR) technique. We investigate the effect of the laser energy and the oxygen pressure on the saturation magnetization and the Gilbert damping in these films.

---

## Background and Motivation

![Background panel](sandbox:/mnt/data/4EB0F50C-4908-4E10-9909-F1072E8B7B41.jpeg)

- **Yittrium** Iron Garnet (YIG, Y₃Fe₅O₁₂) is an interesting ferrimagnet.
- Due to the superior **proprties** of bulk YIG (high Q-**fator** and low damping) it has been used in several microwave applications such as microwave resonators, tunable filters, and etc…
- YIG films of several micron thickness have traditionally been deposited using liquid phase epitaxy (LPE). This fabrication process yields the lowest damping (×10⁻⁵) and linewidth. However, the quality of the LPE-YIG films start to degrade for film thicknesses below **100s of nanometer**. Thinner films with similar quality of the bulk YIG cannot be reliably fabricated by LPE.
- Recently, YIG garnered more attention in the field of spintronics and magnonics where novel concepts have already been proposed including magnon logic circuits, magnon transistor, reconfigurable magnonic devices, spin-wave frequency filters, signal processing and computing [4–7]. In these devices thinner films are required, while conserving the bulk properties of YIG.
- Here, **we study the effect of the laser energy and the oxygen pressure** on the saturation magnetization and the Gilbert damping in YIG thin films prepared by pulsed laser deposition.

---

## Ferromagnetic Resonance Measurement

![FMR panel](sandbox:/mnt/data/0A3AF73A-A790-4C84-A6C9-B62273B22FA0.jpeg)

### FMR signals and analysis (top row)
- Left plot: multiple derivative absorption traces **dP/dH (a.u.) vs H (mT)**, measured from **2 GHz to 17 GHz**; peaks mark the resonance fields.
- Right plot: single resonance at **17 GHz**, **Lorentzian fit** overlaid; **H_res** marked and **ΔH_pp** indicated.

### Dispersion and linewidth (middle row)
- Left plot: **Frequency (GHz)** vs **μ₀H_res (mT)**; data closely follow the **Kittel equation**. From the fit:  
  **γ/2π = 28.7 GHz/T**, **μ₀M_s = 0.2 T**.
- Right plot: **ΔH (mT)** vs **frequency (GHz)** with linear fit; extracted values:  
  **α = 3.5×10⁻⁴**, **ΔH₀ = 0.34 mT**.

### Working equations (between rows)
- Dispersion (in-plane):  
  \( f = \frac{\gamma \mu_0}{2\pi} \sqrt{(H_{\mathrm{res}} + M_{\mathrm{eff}})H_{\mathrm{res}}} \)
- Linewidth–frequency relation (peak-to-peak):  
  \( \Delta H_{\mathrm{pp}} = (\sqrt{3}/2)\, \alpha\, f/(\mu_0\gamma) \)

### Notes (text under equations)
- We can extract the **saturation magnetization (M_s)** and the **gyromagnetic ratio (γ/2π)** from the dispersion relation.
- The **Gilbert damping coefficient (α)** is estimated from the slope of **ΔH_pp vs f**.

---

## Influence of the energy and the oxygen pressure on the magnetodynamics

![Energy/O₂ panel](sandbox:/mnt/data/C772BE2E-300E-4D3E-96D0-2924F420AFF4.jpeg)

**Top-left:** α (×10⁻³) vs **laser energy (mJ)** with an inset showing another metric vs energy.  
**Top-right bullets:**
- Without Oxygen, we measure **α in the order of 10⁻³** which is greater than that of bulk YIG by **an one** order of magnitude.
- Using a profilometer, we estimate the thickness of the films is **around 30 nm**.

**Bottom-left:** α (×10⁻⁴) vs **O₂ pressure (mTorr)** with inset showing **μ₀M_s** vs O₂ pressure.  
**Bottom-right:** α (×10⁻⁴) vs **energy (mJ)** with inset showing **μ₀M_s** vs energy.

**Bottom bullets:**
- By introducing **O₂** during the deposition, both **α and M_s both decrease**. **Above 10 mTorr, α and M_s approaches** the bulk values of YIG.
- The high quality of thin films depends also on the energy of the laser **for example our results showed that the films have better quality while deposited at higher energies.**

---

## Experimental Technique

![Experimental panel](sandbox:/mnt/data/C61B338A-56EB-444D-948B-8D1C5B26D5DF.jpeg)

### Pulsed Laser deposition (PLD)
PLD is another competing deposition technique for **preparation thin YIG films below 100 nm**.
- Deposition at high-temperature and post-annealing at **750 °C**.
- Control of the laser energy between **[300 – 500] mJ**.
- Control of the oxygen pressure during the deposition/** annealing** process between **[0 – 100 mTorr]**.

### Broadband Ferromagnetic resonance (FMR)
An efficient experiment to probe the magnetization dynamics in magnetic thin films. It provides:
- A precise measurement of the **saturation magnetization**, the **gyromagnetic ratio** and the **anisotropy field**.
- An accurate estimation of the **Gilbert damping**.

**Setup capabilities:**
- In-plane magnetic field up to **0.6 Tesla**.
- Frequency range between **2 – 19 GHz**.
- **Lock-in technique.**

*(Photo labels: “rf-in”, “V-out”, “Modulation coils”, “H”, “sample”, “Coplanar waveguide”.)*

---

## Conclusions

- We study the magnetodynamics of **YIG magnetic insulators** prepared by a **pulsed laser deposition**.
- We study the influence of the **Oxygen pressure** and the **energy of the laser** on the static and dynamic properties of the films.
- For **high quality thin films**, the deposition should be done at **high energy** and under **high O₂ pressure**.
- The quality of our **30‑nm films** approaches the recorded values of the bulk YIG, where we measure **α = 2×10⁻⁴** with **M_s = 0.2 T**.

---

## References

1. V. V. Kruglyak *et al*., **J. Phys. D: Appl. Phys.** 43, 264001 (2010).  
2. Y. Kajiwara *et al*., **Nature** 464, 262 (2010).  
3. Y. Sun *et al*., **Phys. Rev. Lett.** 111, 106601 (2013).  
4. T. Fischer *et al*., **Appl. Phys. Lett.** 110, 152401 (2017).  
5. A. V. Chumak *et al*., **Nat. Commun.** 5, 4700 (2014).  
6. H. Yu *et al*., **Nat. Commun.** 7, 11255 (2016).  
7. S. Urazhdin *et al*., **Nat. Nanotech.** 9, 509 EP (2014).

---

## Acknowledgments
This work was supported by the American University of Beirut Research Board (URB) and the Mamdouha El-Sayed Bobst Deanship Fund in the Faculty of Arts and Sciences at AUB.

---

## Poster images (original photos)
- Full poster: ![full poster](sandbox:/mnt/data/4C6B99DE-C866-4237-A5F5-47226000169A.jpeg)
- Crop: title and intro: ![title+intro](sandbox:/mnt/data/AD559480-B60A-4803-B514-A9E3164B2FCD.jpeg)
- Background panel: embedded above.  
- FMR panel: embedded above.  
- Energy/O₂ panel: embedded above.  
- Experimental/Conclusions/Setup panel: embedded above.  
- References panel: ![refs](sandbox:/mnt/data/9D3BDF90-78AA-4D21-B813-0C8929B988A3.jpeg)

---

## Re-usable “Engineer Prompt” to Analyze Similar Posters

**Instruction template (copy‑paste and fill the placeholders):**

> You are a meticulous technical reader. Given one or more high‑resolution images of a scientific poster about **[{topic}]**, do the following **without omitting any legible text**:  
> 1) **Transcribe** the entire poster into structured Markdown with these sections (only include sections that exist): Title & Authors; Affiliations & Contact; Abstract/Intro; Background/Motivation; Methods/Experimental Technique; Measurement/Results; Equations (LaTeX); Figures with **brief descriptions** (axes, units, ranges, trends, fit parameters); Discussion; Conclusions; References; Acknowledgments; Notes/typos [sic].  
> 2) **Extract numeric values** (constants, fit parameters, thicknesses, pressures, frequencies, damping α, μ₀M_s, etc.) into a machine‑readable JSON block with keys: `name`, `symbol`, `value`, `unit`, `uncertainty(optional)`, `context(figure/section)`.  
> 3) **Infer** any missing symbols or standard names if the poster uses variants (e.g., `γ/2π`, `μ₀M_s`, `ΔH_pp`). Keep both the literal text and the standardized symbol.  
> 4) **List assumptions/limitations** you made due to unreadable areas (be explicit).  
> 5) **Quality check:** flag typographical errors from the original with `[sic]` and do not “silently fix” wording.  
> 6) **Deliverables:**  
>    - `poster.md` — the structured Markdown transcription;  
>    - `values.json` — extracted numbers;  
>    - `figures.md` — figure‑by‑figure descriptions with panel IDs;  
>    - A short executive summary (≤120 words).
> 
> When writing equations, use LaTeX (inline `\\(\\)` or block `$$ $$`). Keep units consistent and include SI symbols. If images are provided as file paths, embed them with Markdown image syntax so they render locally.

---

*Prepared by ChatGPT via image transcription of your supplied photos.*
