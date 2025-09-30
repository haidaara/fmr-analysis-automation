
import numpy as np

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

__all__ = ["lorentzian_deriv", "dysonian_deriv", "fit_model", "PREDEFINED_MODELS"]

def lorentzian_deriv(x, A, Hres, dH):
    return A * (x - Hres) / ((((x - Hres)**2 + (dH/2.0)**2 )**2) + 1e-30)

def dysonian_deriv(x, A, Hres, dH, beta):
    Lp = 1.0 / ( (x - Hres)**2 + (dH/2.0)**2 + 1e-30 )
    Ld = (x - Hres) / ((((x - Hres)**2 + (dH/2.0)**2 )**2) + 1e-30)
    return A * ((1-beta)*Ld + beta * ( (x - Hres)*Lp ))

PREDEFINED_MODELS = {
    'lorentzian_deriv': {'fn': lorentzian_deriv, 'p0': (1.0, 0.2, 0.03)},
    'dysonian_deriv':   {'fn': dysonian_deriv,   'p0': (1.0, 0.2, 0.03, 0.2)},
}

def fit_model(x, y, model='lorentzian_deriv', p0=None, bounds=None):
    if callable(model):
        fn = model
        model_name = getattr(model, '__name__', 'custom_model')
        _p0 = p0 if p0 is not None else (1.0, float(np.median(x)), 0.1*(float(np.max(x))-float(np.min(x)))+1e-6)
        _bnd = bounds if bounds is not None else (-np.inf, np.inf)
    else:
        if model not in PREDEFINED_MODELS:
            raise ValueError(f"Unknown model '{model}'. Use one of {list(PREDEFINED_MODELS)} or pass a callable.")
        fn = PREDEFINED_MODELS[model]['fn']
        model_name = model
        _p0 = PREDEFINED_MODELS[model]['p0'] if p0 is None else p0
        _bnd = (-np.inf, np.inf) if bounds is None else bounds

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if SCIPY_AVAILABLE:
        try:
            popt, pcov = curve_fit(fn, x, y, p0=_p0, bounds=_bnd, maxfev=20000)
            res = {'params': popt, 'cov': pcov, 'success': True, 'fun': fn, 'model_name': model_name, 'message': 'ok'}
        except Exception as e:
            res = {'params': None, 'cov': None, 'success': False, 'fun': fn, 'model_name': model_name, 'message': str(e)}
    else:
        try:
            idx = int(np.argmin(np.abs(y)))
            Hres_est = float(x[idx])
            dH_est = 0.1*(float(np.max(x))-float(np.min(x))) + 1e-6
            A_est = (float(np.max(y))-float(np.min(y))) * dH_est
            if model_name == 'dysonian_deriv':
                popt = (A_est, Hres_est, dH_est, 0.2)
            else:
                popt = (A_est, Hres_est, dH_est)
            res = {'params': popt, 'cov': None, 'success': True, 'fun': fn, 'model_name': model_name, 'message': 'naive'}
        except Exception as e:
            res = {'params': None, 'cov': None, 'success': False, 'fun': fn, 'model_name': model_name, 'message': str(e)}

    if res['params'] is not None:
        if model_name == 'dysonian_deriv':
            A, Hres, dH, beta = res['params']
            res['H_res'] = float(Hres)
            res['deltaH'] = float(abs(dH))
            res['beta'] = float(beta)
        else:
            A, Hres, dH = res['params']
            res['H_res'] = float(Hres)
            res['deltaH'] = float(abs(dH))
    return res
