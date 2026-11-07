"""Core numeric tests for 'One Object, Three Readings: The Aethel-Psi Synthesis'.
All numbers recomputed here from the values stated in the three PDFs.
No LVK / IceCube / LIGO / Planck data were downloaded or analysed.
"""
import numpy as np, json, math

G = 6.67430e-11; c = 299792458.0; Msun = 1.98847e30
hbar_eVs = 6.582119569e-16  # eV s
out = {}

# ---------------------------------------------------------------- 1. D(E)
# UAF Eq.11 as typeset: D(E) = 4 - 2 exp[ -(E/Mpl)^2 * beta*nu^2/Psi^2 ] (reading
# the garbled fraction). Test its endpoints.
def D_uaf_typeset(x, beta_nu2_over_Psi2=1.0):
    return 4 - 2*np.exp(-(x**2)*beta_nu2_over_Psi2)
xs = np.array([0, 0.01, 0.1, 0.5, 1.0, 1.2])
print("D(E) as typeset in UAF Eq.11 (exp factor = 1):")
for x in xs:
    print(f"  E/Mpl={x:5.2f}  D={D_uaf_typeset(x):.4f}")
out['D_typeset'] = {str(float(x)): float(D_uaf_typeset(x)) for x in xs}
# NOTE: at E=0 this gives D=2, at E->inf D=4 -- OPPOSITE of the claimed 4->2
# and opposite of Fig.1. So the typeset formula cannot be the plotted one.
# The synthesis paper instead writes D = 4 - 2*F(E/Mpl)*exp(-beta nu^2/Psi^2)
# with F an unspecified increasing 0->1 function. Test with the two natural F:
def D_logistic(x, x0=0.5, k=8.0, s=1.0):
    F = 1/(1+np.exp(-k*(x-x0)))
    return 4 - 2*F*s
def D_rational(x, s=1.0):
    F = x**2/(1+x**2)
    return 4 - 2*F*s
print("\nD(E) with logistic F (x0=0.5,k=8), s=exp(-beta nu^2/Psi^2)=1:")
for x in xs:
    print(f"  E/Mpl={x:5.2f}  D={D_logistic(x):.4f}")
print("D(E) with rational F=x^2/(1+x^2):")
for x in xs:
    print(f"  E/Mpl={x:5.2f}  D={D_rational(x):.4f}")
out['D_logistic'] = {str(float(x)): float(D_logistic(x)) for x in xs}
out['D_rational'] = {str(float(x)): float(D_rational(x)) for x in xs}
# Power suppression P_UAF/P_GR = (D-2)/2 at merger, implied D from Bilby floor:
# Bilby amplitude modifier at merger = 1-0.4 = 0.6 -> power ratio 0.36
mod_merger = 1-0.4
print(f"\nBilby modifier at merger = {mod_merger:.2f}, power ratio = {mod_merger**2:.3f}")
print(f"  -> implied (D-2)/2 = {mod_merger**2:.3f}  =>  D_merger = {2+2*mod_merger**2:.3f}")
print(f"  -> floor modifier 0.5 => power floor 0.25 => D_floor = {2+2*0.25:.2f}")
out['D_implied_by_bilby_merger'] = 2+2*mod_merger**2
out['D_implied_by_bilby_floor'] = 2.5

# ---------------------------------------------------------------- 2. Chirp modifier
def modifier(f, fm, gamma=0.4, width=0.15, width_is_fractional=False):
    w = width*fm if width_is_fractional else width
    return np.clip(1-gamma*np.exp(-0.5*((f-fm)/w)**2), 0.5, 1.0)
fm = 150.0  # Hz, typical BBH merger frequency
f = np.linspace(50, 250, 401)
m_abs = modifier(f, fm, width_is_fractional=False)   # width = 0.15 Hz as written
m_frac = modifier(f, fm, width_is_fractional=True)   # width = 0.15 * fm = 22.5 Hz
print("\nChirp modifier, width as written (0.15 Hz absolute):")
for ff in [100, 140, 149, 150, 151, 160, 200]:
    print(f"  f={ff} Hz  modifier={modifier(np.array([ff]),fm)[0]:.4f}")
print("Chirp modifier, width fractional (0.15*fm = 22.5 Hz):")
for ff in [100, 130, 150, 170, 200]:
    print(f"  f={ff} Hz  modifier={modifier(np.array([ff]),fm, width_is_fractional=True)[0]:.4f}")
# width of the dip (FWHM) in each reading:
print(f"  absolute-width dip FWHM = {2.3548*0.15:.3f} Hz (invisible: LIGO freq bins ~0.25-1 Hz, template mismatch tiny)")
print(f"  fractional-width dip FWHM = {2.3548*0.15*fm:.1f} Hz")
# integrated power loss over a flat band 50-250 Hz:
print(f"  mean modifier (abs width) over 50-250 Hz = {m_abs.mean():.5f}")
print(f"  mean modifier (frac width) over 50-250 Hz = {m_frac.mean():.4f}")
print(f"  min modifier both readings = {m_abs.min():.2f} (floor 0.5 never reached at gamma=0.4; floor only bites if gamma>0.5)")
out['chirp'] = {'modifier_at_merger': 0.6, 'fwhm_abs_Hz': float(2.3548*0.15),
                'fwhm_frac_Hz': float(2.3548*0.15*fm),
                'mean_modifier_abs': float(m_abs.mean()), 'mean_modifier_frac': float(m_frac.mean())}

# ---------------------------------------------------------------- 3. Gate 2 parameter conflict
A_chart, lam_chart = 54.0, 945.0
A_phys, omega_phys = 0.1, 470.0
omega_chart = 2*math.pi/lam_chart
print(f"\nGate 2: chart omega = 2pi/945 = {omega_chart:.6f} (chart units)")
print(f"  physical omega = {omega_phys} rad/s; ratio omega_phys/omega_chart = {omega_phys/omega_chart:.1f}")
print(f"  A ratio chart/phys = {A_chart/A_phys:.0f}; ln(A ratio) = {math.log(A_chart/A_phys):.3f}")
print(f"  f_phys = omega/2pi = {omega_phys/(2*math.pi):.2f} Hz")
# If A(mu) = A0 * (mu/mu0)^gamma with mu ratio = omega ratio:
mu_ratio = omega_phys/omega_chart
gamma_needed = math.log(A_phys/A_chart)/math.log(mu_ratio)
print(f"  power-law exponent needed if mu-ratio = omega-ratio: gamma = {gamma_needed:.3f}")
print(f"  cf. claimed beta = 1.5; with beta=1.5, A_pred = 54 * {mu_ratio:.0f}^(-1.5) = {54*mu_ratio**(-1.5):.3e} (vs 0.1 needed)")
print(f"  with beta=+1.5 (growing), A_pred = {54*mu_ratio**(1.5):.3e}")
# epsilon check: if epsilon runs the same way from an O(1) bare value:
print(f"  epsilon 1e-29 from O(1) needs ln ratio {math.log(1e-29):.1f}; at gamma=-0.563 gives mu-ratio needed {math.exp(math.log(1e-29)/-0.563):.3e} (vs omega ratio {mu_ratio:.3e})")
out['gate2'] = {'omega_chart': omega_chart, 'omega_ratio': mu_ratio,
                'A_ratio': A_chart/A_phys, 'gamma_needed': gamma_needed,
                'A_pred_beta1.5': 54*mu_ratio**(-1.5)}

# ---------------------------------------------------------------- 4. Neutrino delay orders of magnitude
# Ordinary Shapiro delay scale 2GM/c^3 for a compact core:
for M in [1.4, 10.0]:
    print(f"\nShapiro scale 2GM/c^3 for {M} Msun = {2*G*M*Msun/c**3*1e6:.1f} microseconds")
# Neutrino mass delay: dt = (D/2c)(m c^2/E)^2
D_kpc = 10.0; D_m = D_kpc*3.085677581e19
for E_MeV, m_eV in [(10, 0.1), (10, 0.05), (1e6, 0.1)]:  # 1e6 MeV = 1 TeV
    dt = (D_m/(2*c))*(m_eV/(E_MeV*1e6))**2
    print(f"  mass delay D=10kpc, m={m_eV} eV, E={E_MeV} MeV: {dt*1e6:.2f} us")
# SN1987A check: D=51.4 kpc, E~10 MeV, spread observed ~12 s
D87 = 51.4*3.085677581e19
dt87 = (D87/(2*c))*(0.1/1e7)**2
print(f"  SN1987A mass-delay scale (m=0.1eV, E=10MeV, D=51.4kpc): {dt87:.4f} s")
print(f"  -> a microsecond UAF differential is ~1e5 times smaller than the SN1987A burst width;")
print(f"     it is also the same order as the STANDARD mass delay (tens of us at 10 MeV), so the")
print(f"     UAF term is degenerate with unknown neutrino mass unless m is known independently.")
# Ordinary Shapiro for the Sun (for scale): ~120 us for a grazing ray
print(f"  Solar Shapiro delay (grazing) ~ 120 us -- the claimed 'microsecond' UAF signal is")
print(f"  SMALLER than the ordinary solar Shapiro delay, so it requires subtracting a ~100x larger known term.")
out['neutrino'] = {'shapiro_1.4Msun_us': 2*G*1.4*Msun/c**3*1e6,
                   'shapiro_10Msun_us': 2*G*10*Msun/c**3*1e6,
                   'mass_delay_10MeV_10kpc_us': float((D_m/(2*c))*(0.1/1e7)**2*1e6)}

# ---------------------------------------------------------------- 5. Atomic clock / Higgs / CMB reach
print(f"\nAtomic clock: predicted dT/T ~ 1e-29; best 2025 NIST Al+ 5.5e-19; best 2026 Lu+ 1e-19")
print(f"  gap = {5.5e-19/1e-29:.1e}x (10.7 orders of magnitude) below current reach")
print(f"  Toy-model phase alpha*A: alpha=n/w is O(1) by quantisation, A~0.1 -> phase ~0.1 rad,")
print(f"  i.e. a ~10% effect in the toy model, NOT 1e-29. The 1e-29 comes from epsilon (metric")
print(f"  coupling), a DIFFERENT coupling. The paper never reconciles alpha vs epsilon.")
print(f"\nHiggs: SM Gamma_H = 4.1 MeV; CMS Jan 2026: 3.9 +2.7/-2.2 MeV; ATLAS WW limit 13.1 MeV")
print(f"  '0.1% oscillatory shift' = 0.0041 MeV = 4.1 keV, vs uncertainty ~2 MeV -> {2/0.0041:.0f}x below reach")
print(f"  epsilon-scale shift (1e-29 relative) = {4.1e6*1e-29:.1e} eV -- 26 orders below the 0.1% claim.")
print(f"  The two numbers quoted for Prediction 6 (0.1% vs epsilon~1e-29) contradict each other by 1e26.")
print(f"\nCMB: 0.05 deg at l~2200. l=2200 <-> angular scale 180/2200 = {180/2200:.3f} deg = {180/2200*60:.1f} arcmin.")
print(f"  Planck beam ~5-10 arcmin, ACT DR6 reaches l~8500, so the SCALE is resolvable in principle,")
print(f"  but no ripple template (spectrum, phase, amplitude vs l) is derived in any of the three PDFs.")
out['reach'] = {'clock_gap_orders': float(math.log10(5.5e-19/1e-29)),
                'higgs_0.1pct_MeV': 0.0041, 'higgs_eps_eV': 4.1e6*1e-29,
                'cmb_scale_arcmin': 180/2200*60}

# ---------------------------------------------------------------- 6. Bell-EEG
S_qm = 2*math.sqrt(2)
eta_crit = 2/(1+math.sqrt(2))  # Garg-Mermin symmetric-efficiency threshold for CHSH
print(f"\nBell: CHSH local bound 2, QM max 2*sqrt(2) = {S_qm:.4f}")
print(f"  Garg-Mermin efficiency threshold eta_c = 2/(1+sqrt2) = {eta_crit:.4f} = {eta_crit*100:.2f}%  (paper: >82.8% -- CORRECT)")
print(f"  Aethel-Psi claimed modulated S = 0.115 +- 0.04 vs control 2.821 +- 0.04:")
print(f"    0.115 is BELOW the local bound 2: that is NO violation at all, not an enhanced one.")
print(f"    Difference from control = {(2.821-0.115)/0.04:.1f} sigma, but in the direction of DESTROYING the violation.")
print(f"  Timescale mismatch: gamma band 50-80 Hz -> period {1000/50:.0f}-{1000/80:.1f} ms;")
print(f"    a Bell trial (photon flight / detection) is ns-us. EEG cannot gate a single trial;")
print(f"    only a slow statistical correlation of S binned by EEG power is possible, and no")
print(f"    functional form delta-S = h(EEG, theta_Psi) is given, so the test is unformulated.")
print(f"  Required: pre-registered h, N trials for 5 sigma at a stated delta-S. E.g. if delta-S=0.05")
print(f"    with per-trial S scatter ~1, N ~ (5*1/0.05)^2 = {int((5/0.05)**2):,} trials per EEG bin.")
out['bell'] = {'S_qm': S_qm, 'eta_crit': eta_crit, 'modulated_S': 0.115}

# ---------------------------------------------------------------- 7. Trace-equation algebra check (sympy-free, by hand)
# Contraction identity: g^{mu nu}(nabla_mu nabla_nu - g_{mu nu} Box) = Box - 4 Box = -3 Box.
# Paper's Eq.(4): R(1+2 k xi Psi^2) = -k T + 6 k xi Box Psi^2.
# If the modified Einstein eq is G + k xi (nabla nabla - g Box)Psi^2 = k T_matter + ...,
# contracting gives -R - 3 k xi Box Psi^2 ... (sign of the Box term) plus the xi R Psi^2
# term contributes -R * (2 k xi Psi^2)? The paper's proof as printed skips the
# (nabla nabla) contraction and the variation of xi R Psi^2 itself (Remark 2.1).
# Verdict below is therefore INCONCLUSIVE-by-inspection, not a verified derivation.
print("\nTrace equation: contraction identity g^(munu)(nabla_mu nabla_nu - g_munu Box) = -3 Box.")
print("  The paper's proof does not show the step from Eq.(5) to Eq.(4); with the sign")
print("  convention as printed the Box term's sign is not derivable from the text alone.")
print("  This is exactly Gate 1 / Remark 2.1: the full variation is deferred, so Eq.(4)")
print("  is UNVERIFIED as a derivation, though it is internally used consistently thereafter.")

json.dump(out, open('results_core.json','w'), indent=2)
print("\nWrote results_core.json")
