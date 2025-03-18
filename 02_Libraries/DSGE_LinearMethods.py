# Module created for ...
# ...
# Written by Alan Ledesma and Juan Carlos Aquino
# (c) Oct-2022

from sympy.utilities import lambdify
import sympy as sp
from numpy import log, dot, transpose, diag, r_, c_, arange, concatenate, where
from numpy import zeros, identity, real, ndarray, all, array, isin, argsort, setdiff1d
from numpy import empty, float64, isscalar, isnan, delete, reshape, tile, nan
from numpy import sum as suma
from numpy import sqrt as sqrta
from numpy.linalg import matrix_rank, inv, solve, det, norm, pinv
from numpy.random import normal
from math import sqrt
from scipy.linalg import ordqz, qz, solve_discrete_lyapunov
from pandas import DataFrame, DatetimeIndex, Timestamp, Period, to_datetime, date_range
import pandas as pd
from HandleTimeSeries import quarterly_dates_from_end, generate_quarterly_dates, get_EndHistory, get_FirstHistory
import sys
python_version = sys.version_info
import os
import importlib.util
import re
import HandleTimeSeries as HTS
from HandleTimeSeries import transformar_matriz_varnames
from HandleTimeSeries import DB2excel

# flatten list
def flatten(ll):
    return flatten(ll[0]) + (flatten(ll[1:]) if len(ll) > 1 else []) if type(ll) is list else [ll]


# Function to report matrices nicely
def matprint(mat, fmt="g"):
    nd = mat.ndim
    if nd == 1:
        nc = mat.shape[0]
        mat = mat.reshape(1, nc)

    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

# find columns with only zeros (symbolic)
def find_all_zero_columns(matrix):
    num_cols = matrix.cols
    all_zero_columns = []
    all_nonzero_columns = []
    for col_idx in range(num_cols):
        column = matrix[:, col_idx]  # Extract the column
        if column == sp.zeros(matrix.rows, 1):
            all_zero_columns.append(col_idx)
        else:
            all_nonzero_columns.append(col_idx)
    return array(all_zero_columns), array(all_nonzero_columns)


def sympy_to_list(sympy_array):    
    string_list = []
    for element in sympy_array:
        string_list.append(str(element))

    return string_list

# Function to find matrices from symbolic equations (dynare format)
def gen_ModMatFuns(EQ, Yoriginal, Yp, Y, Yl, U, CC, Yobs=None):
    ftildexp = EQ.jacobian(Yp)
    ftildex  = EQ.jacobian(Y)
    ftildexl = EQ.jacobian(Yl)
    fu       = EQ.jacobian(U)
    n        = len(Y)
    subsYp   = [(Yp[ii], Y[ii]) for ii in range(n)]
    subsYl   = [(Yl[ii], Y[ii]) for ii in range(n)]
    subsU    = [(var, 0) for var in U]
    f0 = EQ
    f0 = f0.subs(subsYp)
    f0 = f0.subs(subsYl)
    f0 = f0.subs(subsU)
    f0 = sp.simplify(f0)
    # size of the system
    nu       = len(U)
    nEQ      = len(EQ)
    if n>nEQ:
        raise('Problem: there are more variables declared than equations!')
    elif n<nEQ:
        raise('Problem: there are more equations declared than variables!')

    # find states: k
    _,kx = find_all_zero_columns(ftildexl)
    nk     = len(kx)

    # split states, k, into pure backwards, k2, and with leads, k1
    _,xk1x = find_all_zero_columns(ftildexp[:,kx.tolist()])
    nk1    = len(xk1x)
    nk2    = nk-nk1
    k1x    = kx[xk1x]
    k2x    = kx[~isin(kx, k1x)]
    kx     = concatenate((k1x,k2x))
    
    # rest of variables
    yx = arange(n)
    yx = yx[~isin(yx, kx)]
    ny = len(yx)

    fyp  = ftildexp[:,yx.tolist() ]
    fk2  = ftildex[: ,k2x.tolist()]
    fy   = ftildex[: ,yx.tolist() ]
    fk2l = ftildexl[:,k2x.tolist()]
    if nk1==0:
        solabA = fk2.row_join(fyp)
        solabB = fk2l.row_join(fy)
        solabB = -1*solabB
        solabD = -1*fu
    else:
        fk1p   = ftildexp[:,k1x.tolist()]
        fk1    = ftildex[:, k1x.tolist()]
        fk1l   = ftildexl[:,k1x.tolist()]
        solabA = fk1.row_join(fk2).row_join(fyp).row_join(fk1p)
        auxmat = sp.eye(nk1).row_join(sp.zeros(nk1,n))
        solabA = solabA.col_join(auxmat)
        solabB = fk1l.row_join(fk2l).row_join(fy).row_join(sp.zeros(n,nk1))
        solabB = -1*solabB
        auxmat = sp.zeros(nk1,n).row_join(sp.eye(nk1))
        solabB = solabB.col_join(auxmat)
        solabD = -1*fu.col_join(sp.zeros(nk1,nu))


    Order_declaration2sol = concatenate((kx,yx))
    Order_sol2declaration = argsort(Order_declaration2sol)
    fxp  = ftildexp[:,Order_declaration2sol.tolist()]
    fx   = ftildex[:,Order_declaration2sol.tolist()]
    fxl  = ftildexl[:,Order_declaration2sol.tolist()]

    funfxp  = lambdify(CC, fxp)
    funfx   = lambdify(CC, fx)
    funfxl  = lambdify(CC, fxl)
    funfu   = lambdify(CC, fu)
    funf0   = lambdify(CC.col_join(Y), f0)
    funsolabA   = lambdify(CC, solabA)
    funsolabB   = lambdify(CC, solabB)
    funsolabD   = lambdify(CC, solabD)

    def funvecfxp(x): return funfxp(*x)
    def funvecfx(x):  return funfx(*x)
    def funvecfxl(x): return funfxl(*x)
    def funvecfu(x):  return funfu(*x)
    def funvecf0(x):  return funf0(*x)
    def funvecsolabA(x): return funsolabA(*x)
    def funvecsolabB(x): return funsolabB(*x)
    def funvecsolabD(x): return funsolabD(*x)

    if Yobs is not None:
        Ysolorder = Y[Order_declaration2sol.tolist(),:]
        Z         = Ysolorder.jacobian(Yobs).T
        nobs      = len(Yobs)
        Z         = array(Z).astype(float64)
    else:
        Z         = empty(shape=(1,n))
        nobs      = 0

    modelo = {'Desc.': {'Endogenous': Y,
                        'Declared Endogenous': Yoriginal,
                        'States': Yl[kx.tolist(),0],
                        'Shocks': U,
                        'Coefficients': CC,
                        'n':  n,
                        'ny': ny,
                        'nk': nk,
                        'nk1': nk1,
                        'nk2': nk2,                        
                        'nu': nu,
                        'nobs': nobs},
              'FunMatForm': {'funfxp':  funvecfxp,
                             'funfx':   funvecfx,
                             'funfxl':  funvecfxl,
                             'funfu':   funvecfu,
                             'funf0':   funvecf0,
                             'funsolabA': funvecsolabA,
                             'funsolabB': funvecsolabB,
                             'funsolabD': funvecsolabD},
              'Rearranging_index': {'declaration2solution': Order_declaration2sol,
                                      'solution2declaration': Order_sol2declaration},
              'StateSpaceForm': {'obs_names': Yobs,
                                 'alpha_names': Y[Order_declaration2sol.tolist(),0],
                                 'shock_names': U,
                                 'Z': Z,
                                 'H': zeros(shape=(nobs,nobs))}}

    return modelo

def solab(a, b, nk,flag_print=False):
    """
    Function: solab
    Written by Paul Klein
    Rewritten in November 2007 after a suggestion by Alexander Meyer-Gohde
    Format: [f,p] = solab(a,b,nk);
    Purpose: Solves for the recursive representation of the stable solution to a system
    of linear difference equations.
    Inputs: Two square matrices a and b and a natural number nk
    a and b are the coefficient matrices of the difference equation
    a*x(t+1) = b*x(t)
    where x(t) is arranged so that the state variables come first, and
    nk is the number of state variables.
    Outputs: the decision rule f and the law of motion p. If we write
    x(t) = [k(t-1);u(t)] where k(t-1) contains precisely the state variables, then
    u(t) = f*k(t-1) and
    k(t) = p*k(t-1).
    Calls: qzdiv, qzswitch

    Translated to python by Alan Ledesma in 2020
    """

    s, t,_,_,_,z = ordqz(a, b, sort='ouc')  # Reordered of generalized eigenvalues
    z21 = z[nk:, :nk]
    z11 = z[:nk, :nk]
    
    Estable = True # Stability conditions
    if matrix_rank(z11) < nk: # Checking Invertibility
        if not flag_print:
            print("Invertibility condition violated")
        Estable = False
    
    if abs(t[nk-1, nk-1]) > abs(s[nk-1, nk-1]):
        if not flag_print:
            print('The equilibrium is locally indeterminate.')
        Estable = False

    elif abs(t[nk, nk]) < abs(s[nk, nk]):
        if not flag_print:
            print('There is no local equilibrium.')
        Estable = False

    if Estable:
        z11i = inv(z11)
        s11  = s[:nk, :nk]
        t11  = t[:nk, :nk]
        dyn  = solve(s11, t11)
        f    = real(dot(z21,z11i))
        p    = real(dot(z11,dot(dyn,z11i)))
    else:
        f = empty(shape=(nk,nk))
        p = empty(shape=(nk,nk))

    return f,p,Estable

def SolveModel(modelo, txt_file_arg=None, flag_print=False):
    tol = 1e-10 # Tolerance to push to zero
    # variables original order within the model
    CC = modelo["Desc."]["Coefficients"]
    Y = modelo["Desc."]["Endogenous"]
    if txt_file_arg is not None:
        txt_file = txt_file_arg
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"No se encontró el archivo {txt_file}.")    
    # generate temporal .py
    base_name = os.path.splitext(os.path.basename(txt_file))[0]
    py_file = base_name + ".py"
    # Leer el contenido del archivo .txt
    with open(txt_file, "r", encoding="utf-8") as f_txt:
        txt_content = f_txt.read()
    with open(py_file, "w", encoding="utf-8") as f_py:
        # Cabecera y definición de f1()
        f_py.write("import numpy as np\n\n")
        f_py.write("def f1():\n")
        # Insertar contenido del .txt con indentación
        for line in txt_content.splitlines():
            if not line.startswith("    "):
                f_py.write("    " + line + "\n")
            else:
                f_py.write(line + "\n")
        
        # Procesar variables Y y escribir asignaciones
        for var in Y:
            var_str = str(var)
            match = re.search(r"Ͱ([^Ͱ]+)Ͱ", var_str)  # Ajusta el regex según necesidad
            if match:
                captured = match.group(1)
                f_py.write(f"    {var_str} = {captured}\n")
        f_py.write("    return np.array([\n")
        for i, param in enumerate(CC):
            sep = "," if i < len(CC) - 1 else ""
            f_py.write(f"        {param}{sep}\n")
        f_py.write("    ]), np.array([\n")
        for i, var in enumerate(Y):
            sep = "," if i < len(Y) - 1 else ""
            f_py.write(f"        {var}{sep}\n")
        f_py.write("    ])\n")    
    spec = importlib.util.spec_from_file_location(base_name, py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.remove(py_file)
    # f1() retorns two arrays: CC & Y
    CCvals, Y0 = module.f1()

    n   = modelo['Desc.']['n']
    ny  = modelo['Desc.']['ny']
    nk  = modelo['Desc.']['nk']
    nk1 = modelo['Desc.']['nk1']
    nu  = modelo['Desc.']['nu']
    solabA = modelo['FunMatForm']['funsolabA'](CCvals)
    solabB = modelo['FunMatForm']['funsolabB'](CCvals)
    solabD = modelo['FunMatForm']['funsolabD'](CCvals)

    # Starting point evaluation
    Estable0 = True
    f0zero = modelo['FunMatForm']['funf0'](r_[CCvals,Y0])
    if (abs(f0zero)>tol).any():
        if flag_print:
            print('Problem with steady state (or initial point):')
            print('Check equations:')
            eqn = arange(1,n+1).reshape(-1,1)
            print(eqn[abs(f0zero)>tol])
            Estable0 = False

    # gyk & gkk
    gyk,gkk,Estable = solab(solabA,solabB,nk,flag_print)

    Estable = Estable and Estable0
    if Estable:
        if flag_print:
            print('-> Stability conditions met')

        gyk[abs(gyk) < tol] = 0.0
        gkk[abs(gkk) < tol] = 0.0

        # gk0 & gy0
        kx    = modelo['Rearranging_index']['declaration2solution'][:nk].tolist()
        z1bar = Y0[kx].reshape(-1,1)
        ky    = modelo['Rearranging_index']['declaration2solution'][nk:].tolist()
        k1x   = modelo['Rearranging_index']['declaration2solution'][:nk1].tolist()
        z2bar = r_[Y0[ky].reshape(-1,1),Y0[k1x].reshape(-1,1)]
        gk0   = dot( identity(nk)-gkk , z1bar )
        gy0   = z2bar - dot( gyk , z1bar )
        gk0[abs(gk0) < tol] = 0.0
        gy0[abs(gy0) < tol] = 0.0

        # gyu & gku
        auxMA = r_[identity(nk),gyk]
        auxMB = r_[zeros(shape=(nk,ny+nk1)),identity(ny+nk1)]
        auxM  = c_[dot(solabA,auxMA),-1*dot(solabB,auxMB)]
        gkyu  = solve( auxM , solabD )
        gku   = gkyu[:nk,:].reshape(-1,nu)
        gyu   = gkyu[nk:,:].reshape(-1,nu)
        gku[abs(gku) < tol] = 0.0
        gyu[abs(gyu) < tol] = 0.0

        """
        It must happend that
        gy0[ny:,:]-gk0[:nk1,:]=0
        gyk[ny:,:]-gkk[:nk1,:]=0
        gyu[ny:,:]-gku[:nk1,:]=0
        """
        gy0f = gy0
        gykf = gyk
        gyuf = gyu

        gy0 = gy0[:ny,0].reshape(-1,1)
        gyk = gyk[:ny,:].reshape(-1,nk)
        gyu = gyu[:ny,:].reshape(-1,nu)

        if flag_print:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_colwidth', None)
            posy = modelo['Rearranging_index']['declaration2solution'].tolist()
            varnames = modelo['Desc.']['Endogenous'][posy,:]
            varnames = HTS.transformar_matriz_varnames(varnames)
            ynames =  varnames[nk:,:]
            k0names = varnames[:nk,:]
            knames =  modelo['Desc.']['States']
            knames = HTS.transformar_matriz_varnames(knames)
            unames =  modelo['Desc.']['Shocks']
            unames = HTS.transformar_matriz_varnames(unames)
            index = ['Int.']+sympy_to_list(knames)+sympy_to_list(unames)

            print('\nSteady state')
            # Numerical fixed point
            iT     = 250
            # ordering k1, k2, y
            c = r_[gk0,gy0]
            # [k\\y] = [gkk 0\\ gyk 0][k\\y] 
            Tk = c_[gkk,zeros(shape=(nk,ny))]
            Ty = c_[gyk,zeros(shape=(ny,ny))]
            T  = r_[Tk,Ty]
            intp   = zeros((n,n))
            Tpower = identity(n)
            for ii in range(iT):
                intp   = intp + Tpower
                Tpower = dot(Tpower,T)
            Y0user = Y0[posy].reshape(-1,1)
            Y0num  = dot(intp,c.reshape(-1,1)) + dot(Tpower,Y0user)
            Y0num[abs(Y0num)<tol]=0.0
            M2S = c_[Y0user,Y0num]
            mat2show = DataFrame(M2S, columns=['User supplied','Numerical'], index=sympy_to_list(varnames))
            print(mat2show)

            print('\nPolicy function: Endogeous')
            columns = sympy_to_list(ynames)
            M2S = r_[gy0.T,gyk.T,gyu.T]
            mat2show = DataFrame(M2S, columns=columns, index=index)
            print(mat2show)

            print('\nPolicy function: States')
            columns = sympy_to_list(k0names)
            M2S = r_[gk0.T,gkk.T,gku.T]
            mat2show = DataFrame(M2S, columns=columns, index=index)
            print(mat2show)
            pd.reset_option('display.max_rows')
            pd.reset_option('display.max_columns')
            pd.reset_option('display.max_colwidth')


    else:
        print('* Stability conditions NOT met: Empty outcome!')
        gk0  = empty(shape=(nk, 1))
        gy0  = empty(shape=(ny, nk))
        gkk  = empty(shape=(nk, nk))
        gyk  = empty(shape=(ny, nk))
        gku  = empty(shape=(nk, nu))
        gyu  = empty(shape=(ny, nu))
        gy0f = empty(shape=(ny+nk1,1))
        gykf = empty(shape=(ny+nk1,nk))
        gyuf = empty(shape=(ny+nk1,nu))

    solucion = {'Stable': Estable, 
                'Y0':   Y0[modelo['Rearranging_index']['declaration2solution'].tolist()],
                'gk0':  gk0,
                'gy0':  gy0,
                'gkk':  gkk,
                'gyk':  gyk,
                'gku':  gku,
                'gyu':  gyu,
                'gy0f': gy0f,
                'gykf': gykf,
                'gyuf': gyuf}
    modelo['Solution'] = solucion

    # RSS
    fxp = modelo['FunMatForm']['funfxp'](CCvals)
    fx = modelo['FunMatForm']['funfx'](CCvals)
    # ordering k1, k2, y
    c = r_[gk0,gy0]
    # [k\\y] = [gkk 0\\ gyk 0][k\\y] 
    Tk = c_[gkk,zeros(shape=(nk,ny))]
    Ty = c_[gyk,zeros(shape=(ny,ny))]
    T  = r_[Tk,Ty]
    # [k\\y] = [gku\\ gyu]u
    R = r_[gku,gyu]

    modelo['StateSpaceForm']['c']   = c
    modelo['StateSpaceForm']['T']   = T
    modelo['StateSpaceForm']['R']   = R
    modelo['StateSpaceForm']['Q']   = identity(modelo['Desc.']['nu'])
    modelo['StateSpaceForm']['fxp'] = fxp
    modelo['StateSpaceForm']['fx']  = fx

    return modelo

def getIRF(modelo,Horizon=20):
    nu = modelo['Desc.']['nu']
    n  = modelo['Desc.']['n']
    ny = modelo['Desc.']['ny']
    nk = modelo['Desc.']['nk']
    gkk = modelo['Solution']['gkk']
    gyk = modelo['Solution']['gyk']
    gku = modelo['Solution']['gku']
    gyu = modelo['Solution']['gyu']
    IRF = zeros(shape=(Horizon,n,nu))
    gkkpower = identity(nk)
    for hh in range(Horizon):
        if hh == 0:
            yresponse = gyu
            kresponse = gku
        else:
            yresponse = dot(gyk,kresponse) # kresponse at "t-1"
            gkkpower = dot(gkkpower,gkk)
            kresponse = dot(gkkpower,gku)
        k2report = kresponse
        k2report = k2report.reshape(1,nk,nu)
        IRF[hh,:nk,:] = k2report
        y2report = yresponse
        y2report = y2report.reshape(1,ny,nu)
        IRF[hh,nk:,:] = y2report
    
    index      = arange(Horizon)
    Order_declaration2sol=modelo['Rearranging_index']['declaration2solution']
    varnames   = sympy_to_list(HTS.transformar_matriz_varnames(modelo['Desc.']['Endogenous'][Order_declaration2sol.tolist(),0]))
    shocknames = sympy_to_list(HTS.transformar_matriz_varnames(modelo['Desc.']['Shocks']))
    IRFdict = dict()
    for ss in range(nu):
        IRFdict[shocknames[ss]] = DataFrame(IRF[:,:,ss], columns=varnames, index=index)

    return IRFdict

def simulateDSGE(modelo,Thorizon,dateend):
    order  = modelo['Rearranging_index']['declaration2solution'].tolist()
    vnames = sympy_to_list(HTS.transformar_matriz_varnames(modelo['Desc.']['Endogenous'][order,:]))
    c      = modelo['StateSpaceForm']['c']
    T      = modelo['StateSpaceForm']['T']
    R      = modelo['StateSpaceForm']['R']
    nburn = 1000
    nsim  = Thorizon
    nvar  = T.shape[0]
    nshck = R.shape[1]
    SIM = zeros(shape=(nburn+nsim,nvar))
    shocks  = normal(0,1,nshck*(nburn+nsim)).reshape(-1,nshck)
    for tt in range(1,nburn+nsim):
        ylagged = SIM[tt-1,:].reshape(-1,1)
        ycontp  = c + dot(T,ylagged) + dot(R,shocks[tt,:].reshape(-1,1))
        SIM[tt,:] = ycontp.reshape(-1,)

    SIM = SIM[nburn:,:]

    dates = quarterly_dates_from_end(dateend, Thorizon)
    simDF = DataFrame(SIM, columns=vnames, index=dates)    

    return simDF

def KFplus(Y, mod, ops=None):

    # Check dimensions
    YY = Y.T
    np, n = YY.shape
    nZ = nH = nT = nQ = nR = nc = 1
    if len(mod['Z'].shape)==3:
        aZ, bZ, nZ = mod['Z'].shape
    else:
        aZ, bZ = mod['Z'].shape
    if len(mod['H'].shape)==3:
        aH, bH, nH = mod['H'].shape
    else:
        aH, bH = mod['H'].shape
    if len(mod['T'].shape)==3:
        aT, bT, nT = mod['T'].shape
    else:
        aT, bT = mod['T'].shape
    if len(mod['Q'].shape)==3:
        aQ, bQ, nQ = mod['Q'].shape
    else:
        aQ, bQ = mod['Q'].shape  

    # Default setup for matrix R in the state equation
    if 'R' not in mod:
        mod['R'] = identity(aT)
    if len(mod['R'].shape)==3:
        aR, bR, nR = mod['R'].shape
    else:
        aR, bR = mod['R'].shape  

    ns = bZ # Dimension of state vector
    # Dimension of perturbation vector in state equation
    nr = bR if not isscalar(mod['R']) else bQ

    # Check comfortability in modtem matrices
    if aZ != np:
        raise ValueError('Wrong dimension: Z should be (p x s)')
    if aH != np or bH != np:
        raise ValueError('Wrong dimension: H should be (p x p)')
    if aT != ns or bT != ns:
        raise ValueError('Wrong dimension: T should be (s x s)')
    if not isscalar(mod['R']):
        if aQ != nr or bQ != nr:
            raise ValueError('Wrong dimension: Q should be (r x r)')
    if not isscalar(mod['R']):
        if aR != ns:
            raise ValueError('Wrong dimension: R should be (s x r)')

    # Deterministic vectors in the measurement equation
    if 'd' in mod:
        flagd = True
        flagX = False
        ad, bd, nd = mod['d'].shape
        if ad != np or bd != 1:
            raise ValueError('Wrong dimension: d should be (p x 1)')
    else:
        flagd = False
        flagX = False
        # Regressors vectors in the measurement equation
        if 'X' in mod:
            flagX = True
            if 'D' not in mod:
                raise ValueError('Matrix D not specified')
            XX = mod['X'].T
            kx, nx = XX.shape
            rowsD, colsD = mod['D'].shape
            if kx != colsD:
                raise ValueError('Wrong dimension: X should be (n x kx) and D should be (p x kx)')
            if np != rowsD:
                raise ValueError('Wrong dimension: D should be (p x kx)')

    # Deterministic vectors in the state equation
    if 'c' in mod:
        flagc = True
        flagW = False
        if len(mod['c'].shape)==3:
            ac, bc, nc = mod['c'].shape
        else:
            ac, bc = mod['c'].shape
        if ac != ns or bc != 1:
            raise ValueError('Wrong dimension: c should be (s x 1)')
    else:
        flagc = False
        flagW = False
        # Regressors vectors in the state equation
        if 'W' in mod:
            flagW = True
            if 'C' not in mod:
                raise ValueError('Matrix C not specified')
            WW = mod['W'].T
            kw, nw = WW.shape
            rowsC, colsC = mod['C'].shape
            if kw != colsC:
                raise ValueError('Wrong dimension: W should be (s x kw) and C should be (s x kw)')
            if ns != rowsC:
                raise ValueError('Wrong dimension: C should be (s x kw)')

    # Defaults and options
    if ops is not None:
        compute_updated = ops.get('compute_updated',True)
        compute_smoothed = ops.get('compute_smoothed',True)
        compute_stderrors = ops.get('compute_stderrors',True)
        only_likelihood = ops.get('only_likelihood',False)
        compute_grad = ops.get('compute_grad',False)
        nf = ops.get('nf',0)
        a_initial = ops.get('a_initial',zeros(ns))
        P_initial = ops.get('P_initial',identity(ns) * 1e+10)
        removefirst = ops.get('removefirst',5)
        computeP = ops.get('computeP',True)
        
    else:
        compute_updated, compute_smoothed, compute_stderrors = True, True, True
        only_likelihood, compute_grad, computeP, nf = False, False, True, 0
        a_initial,  P_initial = zeros(ns), identity(ns)*1e+10
        removefirst = 5

    if only_likelihood:
        compute_updated = compute_smoothed = compute_stderrors = False

    # Check if forecasting is possible
    if flagd and nd > 1:
        if nd < (n + nf):
            raise ValueError('Wrong dimension: d does not contain data for forecasting')

    if flagX and nx < (n + nf):
        raise ValueError('Wrong dimension: X does not contain data for forecasting')

    if flagc and nc > 1:
        if nc < (n + nf):
            raise ValueError('Wrong dimension: c does not contain data for forecasting')

    if flagW and nw < (n + nf):
        raise ValueError('Wrong dimension: W does not contain data for forecasting')

    # Check for missing values
    miss = isnan(Y)
    all_miss = suma(miss, axis=1) == np
    any_miss = suma(miss, axis=1) > 0
    some_miss = any_miss & ~all_miss

    # Check if system is time-invariant
    if all([nZ == 1, nH == 1, nT == 1, nQ == 1, nR == 1]):
        time_invariant = True
    else:
        time_invariant = False

    # *********************************
    # Generate objects to store computations
    # KF: Prediction
    a = zeros((ns, n + 1))
    P = zeros((ns, ns, n + 1))
    L = zeros((ns, ns, n + 1))
    v = zeros((np, n))
    F = zeros((np, np, n))
    K = zeros((ns, np, n))
    invF = zeros((np, np, n))
    if compute_stderrors:
        sd_v = zeros((np, n))

    # KF: Updating
    if compute_updated:
        au = zeros((ns, n))
        Pu = zeros((ns, ns, n))
        if compute_stderrors:
            sd_au = zeros((ns, n))

    # KF: Smoothing
    if compute_smoothed:
        r = zeros((ns, n))
        N = zeros((ns, ns, n))
        as_ = zeros((ns, n))
        Ps = zeros((ns, ns, n))
        if compute_stderrors:
            sd_as = zeros((ns, n))
        ee = zeros((np, n))
        Ve = zeros((np, np, n))
        uu = zeros((np, n))
        DD = zeros((np, np, n))
        hh = zeros((nr, n))
        Vh = zeros((nr, nr, n))

    # KF: Forecasting
    if nf > 0:
        af = zeros((ns, nf))
        Pf = zeros((ns, ns, nf))
        yf = zeros((np, nf))
        Ff = zeros((np, np, nf))
        if compute_stderrors:
            sd_yf = zeros((np, nf))
            sd_af = zeros((ns, nf))

    a[:, 0] = a_initial
    P[:, :, 0] = P_initial

    result = mod

    Dx = 0
    Cw = 0

    if time_invariant:
        ZZ = mod['Z']
        HH = mod['H']
        TT = mod['T']
        RR = mod['R']
        QQ = mod['Q']

    minuslogL = 0
    logLi = zeros((n, 1))

    for t in range(1, n + nf + 1):
        if not time_invariant:
            ZZ = mod['Z'] if nZ == 1 else mod['Z'][:, :, t - 1]
            HH = mod['H'] if nH == 1 else mod['H'][:, :, t - 1]
            TT = mod['T'] if nT == 1 else mod['T'][:, :, t - 1]
            RR = mod['R'] if nR == 1 else mod['R'][:, :, t - 1]
            QQ = mod['Q'] if nQ == 1 else mod['Q'][:, :, t - 1]

        if flagd:
            Dx = mod['d'][:, :, t - 1] if nd > 1 else mod['d']
        elif flagX:
            Dx = mod['D'].dot(XX[:, t - 1])

        if flagc:
            Cw = mod['c'][:, :, t - 1] if nc > 1 else mod['c']
        elif flagW:
            Cw = mod['C'].dot(WW[:, t - 1])

        if t <= n:
            # In-sample operations: Prediction and filtering
            yy = YY[:, t - 1] - Dx

            if all_miss[t - 1]:
                v[:, t - 1] = 0
            elif some_miss[t - 1]:
                W = identity(np)
                W = delete(W,miss[t - 1, :], axis=0)
                yy[miss[t - 1, :]] = 0
                v[:, t - 1] = dot(dot(W.T,W),yy - dot(ZZ,a[:, t - 1]))
                computeP = True
            else:
                v[:, t - 1] = yy - dot(ZZ,a[:, t - 1])
                W = identity(np)

            if computeP:
                MM = dot(P[:, :, t - 1],ZZ.T)
                if not all_miss[t - 1]:
                    F[:, :, t - 1] = dot(dot(dot(W.T,W),dot(ZZ,MM) + HH),dot(W.T,W))
                    invF[:, :, t - 1] = dot(W.T,( solve(dot(dot(W,F[:, :, t-1]),W.T), W) ))
                    K[:, :, t - 1] = dot(dot(TT,MM),invF[:, :, t - 1])
                # Recursion for the conditional MSE
                L[:, :, t - 1] = TT - dot(K[:, :, t - 1],ZZ)
                P[:, :, t] = dot(dot(TT,P[:, :, t - 1]),L[:, :, t - 1].T) + dot(dot(RR,QQ),RR.T)
            else:
                # Filter converged to a steady state rule
                if not any_miss[t - 1]:
                    F[:, :, t - 1] = F[:, :, t - 2]
                    invF[:, :, t - 1] = invF[:, :, t - 2]
                    L[:, :, t - 1] = L[:, :, t - 2]
                    K[:, :, t - 1] = K[:, :, t - 2]
                    P[:, :, t] = P[:, :, t - 1]
                else:
                    computeP = True
                    P[:, :, t] = dot(dot(TT,P[:, :, t - 1]),L[:, :, t - 1].T) + dot(dot(RR,QQ),RR.T)
            # Recursion for the conditional mean
            a[:, t] = reshape(Cw + dot(TT,reshape(a[:, t - 1],(ns,1))) + dot(K[:, :, t - 1],reshape(v[:, t - 1],(np,1))),(ns,))

            # Log-likelihood evaluation
            dF = det(dot(dot(W,F[:, :, t - 1]),W.T))
            if dF > 0:
                logLi[t - 1, :] = -0.5 * (log(dF) + dot(dot(v[:, t - 1].T,invF[:, :, t - 1]),v[:, t - 1]))

            if t > (removefirst + 1):
                minuslogL = minuslogL - logLi[t - 1, :] / n

            # Check if the filter has reached the steady state
            if time_invariant and (t > 1) and computeP and not any_miss[t - 1]:
                if norm(P[:, :, t] - P[:, :, t - 1]) < 1e-8:
                    computeP = False

            # Standard errors
            if compute_stderrors:
                sd_v[:, t - 1] = sqrta(diag(F[:, :, t - 1]))

            # Updated quantities
            if compute_updated:
                au[:, t - 1] = a[:, t - 1] + dot(dot(MM,invF[:, :, t - 1]),v[:, t - 1])
                Pu[:, :, t - 1] = P[:, :, t - 1] - dot(dot(MM,invF[:, :, t - 1]),MM.T)
                if compute_stderrors:
                    sd_au[:, t - 1] = sqrta(diag(Pu[:, :, t - 1]))

        else:
            # Forecasting
            h = t - n
            if h == 1:
                if compute_updated:
                    an = au[:, n - 1]
                    Pn = Pu[:, :, n - 1]
                else:
                    an = a[:, n - 1] + dot(dot(MM,invF[:, :, n - 1]),v[:, n - 1])
                    Pn = P[:, :, n - 1] - dot(dot(MM,invF[:, :, n - 1]),MM.T)
                
                af[:, h-1] = Cw + dot(TT,an)
                Pf[:, :, h-1] = dot(dot(TT,Pn),TT.T) + dot(dot(RR,QQ),RR.T)

            else:
                af[:,    h-1] = Cw + TT*af[:, h - 2]
                Pf[:, :, h-1] = dot(dot(TT,Pf[:, :, h - 1]),TT.T) + dot(dot(RR,QQ),RR.T)
            
            yf[:, h-1] = dot(ZZ,af[:, h-1]) + Dx
            Ff[:, :, h-1] = dot(dot(ZZ,Pf[:,:,h-1]),ZZ.T) + HH
            if compute_stderrors:
                sd_af[:,h-1] = sqrta(diag(Pf[:, :, h-1]))
                sd_yf[:,h-1] = sqrta(diag(Ff[:, :, h-1]))


     # Storing the results
    if not only_likelihood:
        result = {'a_p': a.T, 'Sa_p': P, 'e': v.T, 'Se': F, 'invSe': invF, 'K': K}

    if compute_stderrors:
        result['e_sd'] = sd_v.T

    result['logLi'] = logLi

    if compute_updated:
        result['a_u'] = au.T
        result['Sa_u'] = Pu
        if compute_stderrors:
            result['a_u_std'] = sd_au.T

    if nf>0:
        result['a_f'] = af.T
        result['Sa_f'] = Pf
        result['y_f'] = yf.T
        result['Sy_f'] = Ff
        if compute_stderrors:
            result['y_f_std'] = sd_yf.T
            result['a_f_std'] = sd_af.T

    # Smoother
    if compute_smoothed:
        if compute_grad:
            Sr = zeros((ns, ns))

        for t in range(n, 0, -1):
            if nZ > 1:
                ZZ = mod.Z[:, :, t - 1]
            if t > 1:
                r[:, t - 2] = dot(dot(ZZ.T,invF[:, :, t - 1]),v[:, t - 1]) + dot(L[:, :, t - 1].T,r[:, t - 1])
                N[:, :, t - 2] = dot(dot(ZZ.T,invF[:, :, t - 1]),ZZ) + dot(dot(L[:, :, t - 1].T,N[:, :, t - 1]),L[:, :, t - 1])

                as_[:, t - 1] = a[:, t - 1] + dot(P[:, :, t - 1],r[:, t - 2])
                Ps[:, :, t - 1] = P[:, :, t - 1] - dot(dot(P[:, :, t - 1],N[:, :, t - 2]),P[:, :, t - 1].T)

                if compute_grad:
                    Sr += 0.5 * (dot(r[:, t - 2],r[:, t - 2].T) - N[:, :, t - 2])

            else:
                r0 = dot(dot(ZZ.T,invF[:, :, t - 1]),v[:, t - 1]) + dot(L[:, :, t - 1].T,r[:, t - 1])
                N0 = dot(dot(ZZ.T,invF[:, :, t - 1]),ZZ) + dot(dot(L[:, :, t - 1].T,N[:, :, t - 1]),L[:, :, t - 1])

                as_[:, t - 1] = a[:, t - 1] + dot(P[:, :, t - 1],r0)
                Ps[:, :, t - 1] = P[:, :, t - 1] - dot(dot(P[:, :, t - 1],N0),P[:, :, t - 1].T)

            if compute_stderrors:
                sd_as[:, t - 1] = sqrta(diag(Ps[:, :, t - 1]))

        # Storing the results
        result['a_s'] = as_.T
        result['Sa_s'] = Ps
        if compute_stderrors:
            result['a_s_std'] = sd_as.T

    # Disturbance Smoother
    if compute_smoothed:
        if compute_grad:
            Su = zeros((np, np))

        for t in range(n):
            if nH > 1:
                HH = mod.H[:, :, t]
            if nQ > 1:
                QQ = mod.Q[:, :, t]
            if nR > 1:
                RR = mod.R[:, :, t]

            uu[:, t] = dot(invF[:, :, t],v[:, t]) - dot(K[:, :, t].T,r[:, t])
            DD[:, :, t] = invF[:, :, t] + dot(dot(K[:, :, t].T,N[:, :, t]),K[:, :, t])

            ee[:, t] = dot(HH,uu[:, t])
            Ve[:, :, t] = HH - dot(dot(HH,DD[:, :, t]),HH)

            hh[:, t] = dot(dot(QQ,RR.T),r[:, t])
            Vh[:, :, t] = QQ - dot(dot(dot(dot(QQ,RR.T),N[:, :, t]),RR),QQ)

            if compute_grad:
                Su += 0.5 * (dot(uu[:, t],uu[:, t].T) - DD[:, :, t])

        # Storing the results
        # result['u'] = uu.T; result['D'] = DD
        result['eps'] = ee.T; result['Veps'] = Ve
        result['eta'] = hh.T; result['Veta'] = Vh

        if compute_grad:
            result['Su'] = Su
            result['Sr'] = Sr

        result['minuslogL'] = minuslogL.item()
        
    if only_likelihood:
        return minuslogL.item()
    else:
        return result

def KFts(DF, modelo, ops=None):
    obs_names   = sympy_to_list(modelo['StateSpaceForm']['obs_names'])
    alpha_names = sympy_to_list(modelo['StateSpaceForm']['alpha_names'])
    DF1  = DF[obs_names]
    Yobs = DF1.values
    mod  = modelo['StateSpaceForm']   

    if ops is not None:
        ops0 = ops
        ops0['only_likelihood'] = False
        ops0['compute_stderrors'] = False
    else:
        ops0 = {'only_likelihood':False, 'compute_stderrors': False, 'diffuse': True}
    
    if 'diffuse' not in ops0:
        ops0['diffuse'] = True 

    if not ops0['diffuse']:
        a_initial = modelo['Solution']['Y0']
        # T*P*T' -P + R*R' = 0
        T = modelo['StateSpaceForm']['T']
        R = modelo['StateSpaceForm']['R']
        P_initial = solve_discrete_lyapunov(T,dot(R,R.T))
        ops0['a_initial'] = a_initial
        ops0['P_initial'] = P_initial

    result = KFplus(Yobs, mod, ops0)
    
    a_s = result['a_s']
    a_u = result['a_u']
    dates = DF.index
    alpha_names = sympy_to_list(HTS.transformar_matriz_varnames(alpha_names))
    DF_s = DataFrame(a_s, columns=alpha_names, index=dates)
    DF_u = DataFrame(a_u, columns=alpha_names, index=dates) 
    minuslogL = result['minuslogL']
    return DF_s, DF_u, minuslogL

def string2beginning(string_list, string_to_add):
    new_list = []
    for original_string in string_list:
        new_string = string_to_add + original_string
        new_list.append(new_string)
    return new_list

'''
def DSGEforecast(modelo,DFHistory,Horizon=8,StartForecastDate=None,Cond=None):
    # Names and default shocks names
    alpha_names    = sympy_to_list(modelo['StateSpaceForm']['alpha_names'])
    #shock_names    = string2beginning(alpha_names, 'res_')
    shock_names    = sympy_to_list(modelo['StateSpaceForm']['shock_names'])
    # Making sure the variables ordering coincide with alpha_names' ordering
    DFHistory      = DFHistory[alpha_names]
    # Working with dates in history
    Fechas         = DFHistory.index
    EndHistory     = Period(get_EndHistory(DFHistory),freq='Q')
    FirstHistory   = Period(get_FirstHistory(DFHistory),freq='Q')
    if StartForecastDate is not None:
        try:
            UserFirstForecastDate = Period(StartForecastDate)
        except ValueError:
            raise ValueError("Invalid StartForecastDate.  Must be in YYYYQ format (e.g., '2023Q4').")
        UserLastHistoryDate = UserFirstForecastDate - 1
    else:
        UserLastHistoryDate = EndHistory
    UserPreferredEndForecast = UserLastHistoryDate + Horizon
    if EndHistory < UserLastHistoryDate:
        print('Although you entered '+StartForecastDate+' as the projection start date,')
        print('the database only contains complete data up to '+str(EndHistory)+'.')
        print('Therefore, the projection has been adjusted to begin in '+str(EndHistory+1))
        print('and to finish in '+str(UserPreferredEndForecast)+'.')
        Horizon = Horizon + UserLastHistoryDate - EndHistory
        UserLastHistoryDate = EndHistory
        UserFirstForecastDate = UserLastHistoryDate+1


    if EndHistory > UserLastHistoryDate:
        print('Although you entered '+StartForecastDate+' as the projection start date,')
        print('the database goes up to '+str(EndHistory)+'.')
        print('Therefore, the projection has been adjusted to begin in '+str(EndHistory+1))
        print('and to finish in '+str(UserPreferredEndForecast)+'.')
        Horizon = Horizon + UserLastHistoryDate - EndHistory
        UserLastHistoryDate = EndHistory
        UserFirstForecastDate = UserLastHistoryDate+1
    
    # Working with dates in forecast
    ForeFechas     = generate_quarterly_dates(str(UserLastHistoryDate), Horizon)
    ForeFechas     = ForeFechas[1:]
    # Database
    RealFechas = (Fechas <= UserLastHistoryDate.to_timestamp(how='E')) & (Fechas >= FirstHistory.to_timestamp(how='S'))
    DFRealHistory  = DFHistory[ RealFechas ]
    # Default conditional
    DFDefaultCond  = DFHistory[Fechas > UserFirstForecastDate.to_timestamp(how='S')]
    # History as values
    ArrayHistory = DFRealHistory.values
    # Find historical residuals as values
    c = modelo['StateSpaceForm']['c']
    T = modelo['StateSpaceForm']['T']
    R = modelo['StateSpaceForm']['R']
    nobs      = ArrayHistory.shape[0]
    ArrayRes  = ArrayHistory[1:,:] - ( tile(c.T,(nobs-1,1)) + dot(ArrayHistory[0:-1,:],T.T) )
    ArrayRes  = dot(pinv(R),ArrayRes.T).T
    # Dates are good, but internally we work with indices
    timeindex = arange(nobs)
    foreindex = arange(nobs,nobs+Horizon)
    # Forecast
    if Cond is None:
        if DFDefaultCond.empty:
            # Simplest case: unconditional forecast
            ArrayFore    = UnconditionalForecast(c,T,ArrayHistory,Horizon)
            ArrayForeRes = zeros(shape=(Horizon,modelo['Desc.']['nu']))
        else:
            # Conditional forecast in database taken as anticipated
            Cond['Avar']['var']    = DFDefaultCond
            Cond['Avar']['shock']  = string2beginning(DFDefaultCond.columns, 'res_')
            ArrayFore,ArrayForeRes = ConditionalForecast(modelo,ArrayHistory,ArrayRes,Horizon,Cond)
    else:
        if DFDefaultCond.empty:
            if 'UShock' in Cond:
                ushock_names = list(Cond['UShock'].keys())
                ushock_pos   = [shock_names.index(item) for item in ushock_names if item in shock_names]
                ushock_time  = dict()
                ushock_val   = dict()
                for uu in range(len(ushock_names)):
                    ushock_time[ushock_pos[uu]] = where(~isnan(Cond['UShock'][ushock_names[uu]].values))[0]
                    ushock_val[ushock_pos[uu]] = Cond['UShock'][ushock_names[uu]][ushock_time[ushock_pos[uu]]].values
            else:
                ushock_pos  = array([])
                ushock_time = dict()
                ushock_val  = dict()
            
            if 'Uvar' in Cond:
                uvar_names   = list(Cond['Uvar']['var'].keys())
                ushock_names = Cond['Uvar']['shock']
                uvar_pos     = [alpha_names.index(item) for item in uvar_names if item in alpha_names]
                uvshock_pos  = [shock_names.index(item) for item in ushock_names if item in shock_names]
                uvar_time    = dict()
                uvar_val     = dict()
                for uu in range(len(uvar_names)):
                    uvar_time[uvar_pos[uu]] = where(~isnan(Cond['Uvar']['var'][uvar_names[uu]].values))[0]
                    uvar_val[uvar_pos[uu]]  = Cond['Uvar']['var'][uvar_names[uu]][uvar_time[uvar_pos[uu]]].values
            else:
                uvar_pos    = array([])
                uvshock_pos = array([])
                uvar_time   = dict()
                uvar_val    = dict()  

            ArrayFore,ArrayForeRes = UConditionalForecast(c,T,R,ArrayHistory,Horizon,ushock_pos,ushock_time,ushock_val,uvar_pos,uvshock_pos,uvar_time,uvar_val)
        else:
            # Conditional forecast in database taken as anticipated
            # Merge DefaultCond with Cond[Avar], if they overlap "DefaultCond" should prevail
            # Remove observations in Cond[Uvar] if they overlap with "DefaultCond"
            ArrayFore,ArrayForeRes = ConditionalForecast(modelo,ArrayHistory,ArrayRes,Horizon,Cond)
        
    AllDataVar  = r_[DFHistory[Fechas <= UserLastHistoryDate.to_timestamp(how='E')].values,ArrayFore]
    AllDataRes  = r_[zeros(shape=(1,modelo['Desc.']['nu']))*nan,ArrayRes,ArrayForeRes]
    AllData     = c_[AllDataVar,AllDataRes]
    AllNames    = alpha_names+shock_names
    AllDates    = DatetimeIndex(concatenate([Fechas,ForeFechas]),freq='Q')
    return DataFrame(AllData, columns=AllNames, index=AllDates)
'''

def Dates_in_History(DFHistory,StartForecastDate,Horizon):
    EndHistory     = Period(get_EndHistory(DFHistory),freq='Q')
    if StartForecastDate is not None:
        try:
            UserFirstForecastDate = Period(StartForecastDate)
        except ValueError:
            raise ValueError("Invalid StartForecastDate.  Must be in YYYYQ format (e.g., '2023Q4').")
        UserLastHistoryDate = UserFirstForecastDate - 1
    else:
        UserLastHistoryDate = EndHistory
        
    UserPreferredEndForecast = UserLastHistoryDate + Horizon
    if EndHistory != UserLastHistoryDate:
        print('Although you entered '+StartForecastDate+' as the forecast start date,')
        print('the database contains complete data up to '+str(EndHistory)+'.')
        print('Therefore, the forecast has been adjusted to begin in '+str(EndHistory+1))
        print('and to finish in '+str(UserPreferredEndForecast)+'.')
        Horizon = Horizon + UserLastHistoryDate - EndHistory
        UserLastHistoryDate = EndHistory
        UserFirstForecastDate = UserLastHistoryDate+1

    return Horizon, UserLastHistoryDate, UserFirstForecastDate

def ProcessShock(Cond,shock_names,tp):
    shock = tp+'Shock'
    if shock in Cond:
        xshock_names = list(Cond[shock].keys())
        xshock_pos   = [shock_names.index(item) for item in xshock_names if item in shock_names]
        xshock_pos   = [int(x) for x in xshock_pos]
        xshock_time  = dict()
        xshock_val   = dict()
        for uu in range(len(xshock_names)):
            xshock_time[xshock_pos[uu]] = where(~isnan(Cond[shock][xshock_names[uu]].values))[0]
            xshock_val[xshock_pos[uu]] = Cond[shock][xshock_names[uu]].iloc[xshock_time[xshock_pos[uu]]].values
    else:
        xshock_pos  = []
        xshock_time = dict()
        xshock_val  = dict()
    return xshock_pos,xshock_time,xshock_val

def ProcessVar(Cond,shock_names,alpha_names,tp):
    Xvar = tp+'var'
    if Xvar in Cond:
        xvar_names   = list(Cond[Xvar]['var'].keys())
        xshock_names = Cond[Xvar]['shock']
        xvar_pos     = [alpha_names.index(item) for item in xvar_names if item in alpha_names]
        xvar_pos     = [int(x) for x in xvar_pos]
        xvshock_pos  = [shock_names.index(item) for item in xshock_names if item in shock_names]
        xvshock_pos  = [int(x) for x in xvshock_pos]
        xvar_time    = dict()
        xvar_val     = dict()
        for uu in range(len(xvar_names)):
            xvar_time[xvar_pos[uu]] = where(~isnan(Cond[Xvar]['var'][xvar_names[uu]].values))[0]
            xvar_val[xvar_pos[uu]]  = Cond[Xvar]['var'][xvar_names[uu]].iloc[xvar_time[xvar_pos[uu]]].values
    else:
        xvar_pos    = []
        xvshock_pos = []
        xvar_time   = dict()
        xvar_val    = dict()  

    return xvar_pos,xvshock_pos,xvar_time,xvar_val

def ProcessCond(Cond,shock_names,alpha_names):
    ushock_pos,ushock_time,ushock_val = ProcessShock(Cond,shock_names,'U')
    uvar_pos,uvshock_pos,uvar_time,uvar_val = ProcessVar(Cond,shock_names,alpha_names,'U')
    ashock_pos,ashock_time,ashock_val = ProcessShock(Cond,shock_names,'A')
    avar_pos,avshock_pos,avar_time,avar_val = ProcessVar(Cond,shock_names,alpha_names,'A')
 
    return ushock_pos,ushock_time,ushock_val,uvar_pos,uvshock_pos,uvar_time,uvar_val,ashock_pos,ashock_time,ashock_val,avar_pos,avshock_pos,avar_time,avar_val

def DSGEforecast0(modelo,DFHistory,Horizon=8,StartForecastDate=None,Cond=None):
    # Names and default shocks names
    alpha_names    = modelo['StateSpaceForm']['alpha_names']
    alpha_names    = sympy_to_list(HTS.transformar_matriz_varnames(alpha_names))
    shock_names    = sympy_to_list(modelo['StateSpaceForm']['shock_names'])
    # Making sure the variables ordering coincide with alpha_names' ordering
    DFHistory      = DFHistory[alpha_names]
    Fechas         = DFHistory.index
    # Working with dates in history
    Horizon, UserLastHistoryDate, UserFirstForecastDate = Dates_in_History(DFHistory,StartForecastDate,Horizon)
    # Working with dates in forecast
    ForeFechas     = generate_quarterly_dates(str(UserLastHistoryDate), Horizon)
    ForeFechas     = ForeFechas[1:]
    # Database
    DFRealHistory  = DFHistory[ Fechas <= UserLastHistoryDate.to_timestamp(how='E') ]
    # History as values
    ArrayHistory = DFRealHistory.values
    # Rquired matrices
    c   = modelo['StateSpaceForm']['c']
    T   = modelo['StateSpaceForm']['T']
    R   = modelo['StateSpaceForm']['R']
    fxp = modelo['StateSpaceForm']['fxp']
    fx  = modelo['StateSpaceForm']['fx']
    # Find historical residuals as values
    nobs      = ArrayHistory.shape[0]
    ArrayRes  = ArrayHistory[1:,:] - ( tile(c.T,(nobs-1,1)) + dot(ArrayHistory[0:-1,:],T.T) )
    ArrayRes  = dot(pinv(R),ArrayRes.T).T
    # Forecast
    X0 = ArrayHistory[-1,:]
    if Cond is None:
        # Simplest case: unconditional forecast
        ArrayFore    = UnconditionalForecast(c,T,X0,Horizon)
        ArrayForeRes = zeros(shape=(Horizon,modelo['Desc.']['nu']))
    else:
        ushock_pos,ushock_time,ushock_val,uvar_pos,uvshock_pos,uvar_time,uvar_val,ashock_pos,ashock_time,ashock_val,avar_pos,avshock_pos,avar_time,avar_val = ProcessCond(Cond,shock_names,alpha_names)
        # ashock_pos,ashock_time,ashock_val,avar_pos,avshock_pos,avar_time,avar_val
        if (not ashock_time) & (not avar_time): # Only not anticipated shocks
            ArrayFore,ArrayForeRes = UConditionalForecast(c,T,R,X0,Horizon,ushock_pos,ushock_time,ushock_val,uvar_pos,uvshock_pos,uvar_time,uvar_val)
        else:
            # Priorities: anticipated > unanticipated
            # Priorities: hardtuned > softuned
            array_ushock_pos  = array(ushock_pos, dtype=int)   # Arrays are easier to handle (for me)
            array_ashock_pos  = array(ashock_pos, dtype=int)   # Arrays are easier to handle (for me)
            array_uvar_pos    = array(uvar_pos, dtype=int)     # Arrays are easier to handle (for me)
            array_avar_pos    = array(avar_pos, dtype=int)     # Arrays are easier to handle (for me)
            array_uvshock_pos = array(uvshock_pos, dtype=int)  # Arrays are easier to handle (for me)
            array_avshock_pos = array(avshock_pos, dtype=int)  # Arrays are easier to handle (for me)
    
            array_ushock_not_in_ashock   = array_ushock_pos[~isin(array_ushock_pos, array_ashock_pos)]   
            array_ushock_pos0            = concatenate((array_ashock_pos, array_ushock_not_in_ashock)) 
            array_uvar_not_in_avar       = array_uvar_pos[~isin(array_uvar_pos, array_avar_pos)]
            array_uvar_pos0              = concatenate((array_avar_pos, array_uvar_not_in_avar)) 
            if len(array_uvar_not_in_avar)==0:
                array_uvshock_not_in_avshock = array([])
            else:
                array_uvshock_not_in_avshock = array_uvshock_pos[~isin(array_uvshock_pos, array_uvar_not_in_avar)]
            array_uvshock_pos0           = concatenate((array_avshock_pos, array_uvshock_not_in_avshock)) 
            ushock_time0           = ashock_time
            ushock_val0            = ashock_val
            for item in array_ushock_not_in_ashock.tolist():
                ushock_time0[item] = ushock_time[item]
                ushock_val0[item]  = ushock_val[item]
            uvar_time0             = avar_time
            uvar_val0              = avar_val
            for item in array_uvar_not_in_avar.tolist():
                uvar_time0[item]   = uvar_time[item]
                uvar_val0[item]    = uvar_val[item]
            ushock_pos0  = array_ushock_pos0.tolist()
            uvar_pos0    = array_uvar_pos0.tolist()
            uvshock_pos0 = array_uvshock_pos0.tolist()
            ArrayFore0,ArrayForeRes0 = UConditionalForecast(c,T,R,X0,Horizon,ushock_pos0,ushock_time0,ushock_val0,uvar_pos0,uvshock_pos0,uvar_time0,uvar_val0)
            ArrayFore,ArrayForeRes   = AConditionalForecast(c,T,R,fxp,fx,X0,Horizon,ArrayFore0,ArrayForeRes0,ushock_pos,ushock_time,uvar_pos,uvshock_pos,uvar_time,uvar_val,ashock_pos,ashock_time,avar_pos,avshock_pos,avar_time,avar_val)

    AllDataVar  = r_[DFHistory[Fechas <= UserLastHistoryDate.to_timestamp(how='E')].values,ArrayFore]
    AllDataRes  = r_[zeros(shape=(1,modelo['Desc.']['nu']))*nan,ArrayRes,ArrayForeRes]
    AllData     = c_[AllDataVar,AllDataRes]
    AllNames    = alpha_names+shock_names
    if python_version.minor >=12:
        AllDates    = DatetimeIndex(concatenate([Fechas,ForeFechas]),freq='QE')
    else:
        AllDates    = DatetimeIndex(concatenate([Fechas,ForeFechas]),freq='Q')
    return DataFrame(AllData, columns=AllNames, index=AllDates)

def UnconditionalForecast(c,T,X0,Horizon):
    n = T.shape[0]
    ArrayFore = zeros(shape=(Horizon,n))
    for tt in range(Horizon):
        if tt==0:
            ArrayFore[tt,:] = c.T + dot(X0,T.T)
        else:
            ArrayFore[tt,:] = c.T + dot(ArrayFore[tt-1,:],T.T)
        
    return ArrayFore

def UConditionalForecast(c,T,R,X0,Horizon,ushock_pos,ushock_time,ushock_val,uvar_pos,uvshock_pos,uvar_time,uvar_val):
    # Dimensions
    ny = T.shape[0] # # of endogenous
    nu = R.shape[1] # # of shocks
    # Storage
    ArrayFore         = zeros(shape=(Horizon,ny))
    ArrayForeRes      = zeros(shape=(Horizon,nu))

    for tt in range(Horizon):
        # Is there a conditional at time "t"?
        ushock_pos_t = cond_t(tt,ushock_pos,ushock_time) # unanticipated shocks at moment t
        uvar_pos_t, uvshock_pos_t = cond_t(tt,uvar_pos,uvar_time,uvshock_pos) # unanticipated vars and shocks to match them at moment t
        # Priorities: hardtuning > softuning
        ushock_pos_t = setdiff1d(ushock_pos_t, uvshock_pos_t)
        # Unanticipated shocks at moment t
        ushock_t = zeros(shape=(1,nu))
        for ff in range(len(ushock_pos_t)):
            ushock_t[0,ushock_pos_t[ff]] = ushock_val[ushock_pos_t[ff]][ushock_time[ushock_pos_t[ff]]==tt]
        # Forecast
        if tt==0:
            laggedX = X0.reshape(-1,1)
        else:
            laggedX = ArrayFore[tt-1,:].reshape(-1,1)
        if len(uvar_pos_t)==0: # There is no hardtuning at period t
            U = ushock_t.reshape(-1,1)
            X = c + dot(T,laggedX) + dot(R,U)
        else: # Hardtunning
            HardX_t = zeros(shape=(len(uvar_pos_t),1))
            for vv in range(len(uvar_pos_t)):
                tpos = where(uvar_time[uvar_pos_t[vv]] == tt)[0]
                HardX_t[vv,0] = uvar_val[uvar_pos_t[vv]][tpos]
            ushock_t = ushock_t.reshape(-1,1)
            # Reordering
            nyu            = len(uvar_pos_t)
            OriginalOrderU = arange(nu)
            NewOrderU      = concatenate((uvshock_pos_t,setdiff1d(OriginalOrderU, uvshock_pos_t)))
            index2revertU  = argsort(NewOrderU)
            OriginalOrderV = arange(ny)
            NewOrderV      = concatenate((uvar_pos_t,setdiff1d(OriginalOrderV, uvar_pos_t)))
            index2revertV  = argsort(NewOrderV)
            # Reordering c, T, R, laggedX, ushock_at_t
            rc = c[NewOrderV,0].reshape(-1,1)
            rT = T[NewOrderV,:]
            rT = rT[:,NewOrderV]
            rR = R[NewOrderV,:]
            rR = rR[:,NewOrderU]
            rlaggedX  = laggedX[NewOrderV,0].reshape(-1,1)
            rushock_t = ushock_t[NewOrderU,0].reshape(-1,1)
            # Partitions
            cu, cf = rc[:nyu,0].reshape(-1,1), rc[nyu:,0].reshape(-1,1)
            Tuu, Tuf = rT[:nyu,:nyu], rT[:nyu,nyu:]
            Tfu, Tff = rT[nyu:,:nyu], rT[nyu:,nyu:]
            Ruu, Rufu = rR[:nyu,:nyu], rR[:nyu,nyu:]
            Rfu, Rffu = rR[nyu:,:nyu], rR[nyu:,nyu:]
            laggedXu, laggedXf = rlaggedX[:nyu,0].reshape(-1,1), rlaggedX[nyu:,0].reshape(-1,1)
            ufu = rushock_t[nyu:,0].reshape(-1,1)
            # Forecast
            iRuu = inv(Ruu)
            ErX  = HardX_t - cu - dot(Tuu,laggedXu) - dot(Tuf,laggedXf) - dot(Rufu,ufu)
            uu   = dot( iRuu , ErX )
            Xf   = cf + dot(Tfu,laggedXu) + dot(Tff,laggedXf) + dot(Rfu,uu) + dot(Rffu,ufu)
            rX   = r_[HardX_t,Xf]
            X    = rX[index2revertV,0].reshape(-1,1)
            rU   = r_[uu,ufu]
            U    = rU[index2revertU,0].reshape(-1,1)
        
        ArrayFore[tt,:]    = X.reshape(-1,)
        ArrayForeRes[tt,:] = U.reshape(-1,)

    return ArrayFore,ArrayForeRes

def AConditionalForecast(c,T,R,fxp,fx,X0,Horizon,ArrayFore0,ArrayForeRes0,ushock_pos,ushock_time,uvar_pos,uvshock_pos,uvar_time,uvar_val,ashock_pos,ashock_time,avar_pos,avshock_pos,avar_time,avar_val):

    # Dimensions
    ny = T.shape[0] # # of endogenous
    nu = R.shape[1] # # of shocks
    # Storage
    ArrayFore         = zeros(shape=(Horizon,ny))
    ArrayForeRes      = zeros(shape=(Horizon,nu))

    # Rj and anticipated shocks to impose
    Rj = zeros(shape=(ny,nu,Horizon))
    for tt in range(Horizon):
        if tt==0:
            Rj[:,:,tt] = solve( -1*(dot(fxp,T)+fx) , dot(fxp,R) )
        else:
            Rj[:,:,tt] = solve( -1*(dot(fxp,T)+fx) , dot(fxp,Rj[:,:,tt-1]) )
        
    matbool_avshock = unpackavshock(avar_pos,avshock_pos,avar_time,Horizon,nu)   # Hardtuned anticipated shocks
    matbool_ashock  = unpackashock(ashock_pos,ashock_time,Horizon,nu,matbool_avshock) # Softtuned anticipated shocks

    

    flag_noconvergence = True
    rr = -1
    Array_old = c_[ArrayFore0,ArrayForeRes0]

    while  flag_noconvergence:
        rr += 1  
        for tt in range(Horizon):  
            # Is there a conditional at time "t"?
            ashock_pos_t = cond_t(tt,ashock_pos,ashock_time) # anticipated shocks at moment t
            ushock_pos_t = cond_t(tt,ushock_pos,ushock_time) # unanticipated shocks at moment t
            avar_pos_t, avshock_pos_t = cond_t(tt,avar_pos,avar_time,avshock_pos) # anticipated vars and shocks to match them at moment t
            uvar_pos_t, uvshock_pos_t = cond_t(tt,uvar_pos,uvar_time,uvshock_pos) # unanticipated vars and shocks to match them at moment t
            # Priorities: hardtuning > softuning
            ushock_pos_t = setdiff1d(ushock_pos_t, uvshock_pos_t) # Remove unanticipated shocks that are also shocks to hardtune unanticipated variables
            ashock_pos_t = setdiff1d(ashock_pos_t, avshock_pos_t) # Remove anticipated shocks that are also shocks to hardtune anticipated variables
            # Priorities: anticipated > unanticipated
            ushock_pos_t = setdiff1d(ushock_pos_t, ashock_pos_t) # Remove unanticipated shocks that are also anticipated shocks
            uvar_pos_t   = setdiff1d(uvar_pos_t, avar_pos_t)     # Remove unanticipated vars that are also anticipated vars
            # Dimensions at t
            nya_t        = len(avar_pos_t)
            nyu_t        = len(uvar_pos_t)
            nufa_t       = len(ashock_pos_t)
            # shocks at moment t
            if rr==0:
                shocks_tT = ArrayForeRes0[tt:,:].T
            else:
                shocks_tT = ArrayForeRes[tt:,:].T
            # Present values of anticipated shocks at "t"
            PV_t = PV_ashock(tt,Horizon,Rj,shocks_tT,matbool_avshock,matbool_ashock)
            # lagged X
            if tt==0:
                laggedX = X0.reshape(-1,1)
            else:
                laggedX = ArrayFore[tt-1,:].reshape(-1,1)
            # Anticipated hardtuned variables at t
            Xa_t = zeros(shape=(nya_t,1))
            for vv in range(nya_t):
                tpos = where(avar_time[avar_pos_t[vv]] == tt)[0]
                Xa_t[vv,0] = avar_val[avar_pos_t[vv]][tpos]
            # Unanticipated hardtuned variables at t
            Xu_t = zeros(shape=(nyu_t,1))
            for vv in range(nyu_t):
                tpos = where(uvar_time[uvar_pos_t[vv]] == tt)[0]
                Xu_t[vv,0] = uvar_val[uvar_pos_t[vv]][tpos]
            # Reordering (relevant for moment t)
            OriginalOrderU = arange(nu)
            NewOrderU      = concatenate((avshock_pos_t,uvshock_pos_t,ashock_pos_t,setdiff1d(OriginalOrderU, concatenate((avshock_pos_t,uvshock_pos_t,ashock_pos_t))))).astype(int)
            index2revertU  = argsort(NewOrderU).astype(int)
            OriginalOrderV = arange(ny)
            NewOrderV      = concatenate((avar_pos_t,uvar_pos_t,setdiff1d(OriginalOrderV,concatenate((avar_pos_t,uvar_pos_t))))).astype(int)
            index2revertV  = argsort(NewOrderV).astype(int)
            # Reordering c, T, R, laggedX, ushock_t
            rc  = c[NewOrderV,0].reshape(-1,1)
            rT  = T[NewOrderV,:]
            rT  = rT[:,NewOrderV]
            rR  = R[NewOrderV,:]
            rR  = rR[:,NewOrderU]
            rRj = Rj[NewOrderV,:,:]
            rRj = rRj[:,NewOrderU,:]
            rlaggedX  = laggedX[NewOrderV,0].reshape(-1,1)
            rPV_t     = PV_t[NewOrderV,0].reshape(-1,1)
            rshocks_t = shocks_tT[NewOrderU,0].reshape(-1,1)
            # Partitions
            ca, cu, cf  =  rc[:nya_t,0].reshape(-1,1), rc[nya_t:(nya_t+nyu_t),0].reshape(-1,1), rc[(nya_t+nyu_t):,0].reshape(-1,1)
            Taa, Tau, Taf  =  rT[:nya_t         ,:nya_t], rT[:nya_t         ,nya_t:(nya_t+nyu_t)], rT[:nya_t         ,(nya_t+nyu_t):]
            Tua, Tuu, Tuf  =  rT[nya_t:(nya_t+nyu_t),:nya_t], rT[nya_t:(nya_t+nyu_t),nya_t:(nya_t+nyu_t)], rT[nya_t:(nya_t+nyu_t),(nya_t+nyu_t):]
            Tfa, Tfu, Tff  =  rT[(nya_t+nyu_t):   ,:nya_t], rT[(nya_t+nyu_t):   ,nya_t:(nya_t+nyu_t)], rT[(nya_t+nyu_t):   ,(nya_t+nyu_t):]
            Raa, Rau, Rafa, Rafu  =  rR[:nya_t         ,:nya_t], rR[:nya_t         ,nya_t:(nya_t+nyu_t)], rR[:nya_t         ,(nya_t+nyu_t):(nya_t+nyu_t+nufa_t)], rR[:nya_t         ,(nya_t+nyu_t+nufa_t):]
            Rua, Ruu, Rufa, Rufu  =  rR[nya_t:(nya_t+nyu_t),:nya_t], rR[nya_t:(nya_t+nyu_t),nya_t:(nya_t+nyu_t)], rR[nya_t:(nya_t+nyu_t),(nya_t+nyu_t):(nya_t+nyu_t+nufa_t)], rR[nya_t:(nya_t+nyu_t),(nya_t+nyu_t+nufa_t):]
            Rfa, Rfu, Rffa, Rffu  =  rR[(nya_t+nyu_t):   ,:nya_t], rR[(nya_t+nyu_t):   ,nya_t:(nya_t+nyu_t)], rR[(nya_t+nyu_t):   ,(nya_t+nyu_t):(nya_t+nyu_t+nufa_t)], rR[(nya_t+nyu_t):   ,(nya_t+nyu_t+nufa_t):]
            laggedXa,laggedXu,laggedXf = rlaggedX[:nya_t,0].reshape(-1,1), rlaggedX[nya_t:(nya_t+nyu_t),0].reshape(-1,1), rlaggedX[(nya_t+nyu_t):,0].reshape(-1,1)
            PVa,PVu,PVf = rPV_t[:nya_t,0].reshape(-1,1), rPV_t[nya_t:(nya_t+nyu_t),0].reshape(-1,1), rPV_t[(nya_t+nyu_t):,0].reshape(-1,1)
            uu_old, ufa, ufu = rshocks_t[nya_t:(nya_t+nyu_t),0].reshape(-1,1), rshocks_t[(nya_t+nyu_t):(nya_t+nyu_t+nufa_t),0].reshape(-1,1), rshocks_t[(nya_t+nyu_t+nufa_t):,0].reshape(-1,1)
            # Forecast
            # Anticipated and hardtuned
            ErX0  = Xa_t - ( ca + dot(Taa,laggedXa) + dot(Tau,laggedXu) + dot(Taf,laggedXf) )
            mErX1 = dot(Rau,uu_old[:,0].reshape(-1,1)) + dot(Rafa,ufa[:,0].reshape(-1,1)) + dot(Rafu,ufu[:,0].reshape(-1,1))
            mErX2 = PVa
            ErX = ErX0 - mErX1 - mErX2
            ua_new_t = solve( Raa , ErX ).reshape(-1,1)
            # Unanticipated and hardtuned
            ErX0  = Xu_t - ( cu + dot(Tua,laggedXa) + dot(Tuu,laggedXu) + dot(Tuf,laggedXf) )
            mErX1  = dot(Rua,ua_new_t) + dot(Rufa,ufa[:,0].reshape(-1,1)) + dot(Rufu,ufu[:,0].reshape(-1,1))
            mErX2  = PVu
            ErX = ErX0 - mErX1 - mErX2
            uu_new_t = solve( Ruu , ErX ).reshape(-1,1)
            # Free endogenous
            Xf0 = cf + dot(Tfa,laggedXa) + dot(Tfu,laggedXu) + dot(Tff,laggedXf) 
            Xf1 = dot(Rfa,ua_new_t) + dot(Rfu,uu_new_t) + dot(Rffa,ufa[:,0].reshape(-1,1)) + dot(Rffu,ufu[:,0].reshape(-1,1))
            Xf2 = PVf
            Xf_t = Xf0 + Xf1 + Xf2

            rX   = r_[Xa_t,Xu_t,Xf_t]
            X    = rX[index2revertV,0].reshape(-1,1)
            rU   = r_[ua_new_t,uu_new_t,ufa[:,0].reshape(-1,1),ufu[:,0].reshape(-1,1)]
            U    = rU[index2revertU,0].reshape(-1,1)
                
            ArrayFore[tt,:] = X.reshape(-1,)
            ArrayForeRes[tt,:] = U.reshape(-1,)

        Array_new = c_[ArrayFore,ArrayForeRes]
        if norm(Array_old-Array_new)>1e-12:
            Array_old = Array_new
        else:
            flag_noconvergence = False
            

    return ArrayFore,ArrayForeRes

def unpackashock(xshock_pos,xshock_time,Horizon,nu,matbool_avshock):
    matbool_ashock = zeros(shape=(Horizon,nu),dtype=bool)
    for vv in range(len(xshock_pos)):
        matbool_ashock[xshock_time[xshock_pos[vv]],xshock_pos[vv]] = True
    matbool_ashock[matbool_avshock] = False # Hardtuned > Softuned
    return matbool_ashock

def unpackavshock(xvar_pos,xvshock_pos,xvar_time,Horizon,nu):
    matbool_avshock = zeros(shape=(Horizon,nu),dtype=bool)
    for vv in range(len(xvshock_pos)):
        matbool_avshock[xvar_time[xvar_pos[vv]],xvshock_pos[vv]] = True
    return matbool_avshock

def cond_t(tt,xy_pos,xy_time,xyv_pos=None):
    array_xy_pos = array(xy_pos)
    if xyv_pos is not None:
        array_xyv_pos = array(xyv_pos)
    bool_xy_pos_t = zeros(len(xy_pos),dtype=bool)
    for vv in range(len(xy_pos)):
        bool_xy_pos_t[vv] = any(xy_time[xy_pos[vv]] == tt)                       # True for "y" at moment "t"
    xy_pos_t    = array_xy_pos[where(bool_xy_pos_t)[0]].astype(int).tolist()     # "y" at moment "t"
    if xyv_pos is not None:
        xyv_pos_t = array_xyv_pos[where(bool_xy_pos_t)[0]].astype(int).tolist()  # shocks to match "y" at moment "t"
        return xy_pos_t, xyv_pos_t
    else:
        return xy_pos_t
    
def PV_ashock(tt,Horizon,Rj,shocks_tT,matbool_avshock,matbool_ashock):
    matbool_avshock = matbool_avshock.T
    matbool_ashock  = matbool_ashock.T
    matbool_avshock_tT = matbool_avshock[:,tt:]
    matbool_ashock_tT  = matbool_ashock[:,tt:]
    fshocks_tT = shocks_tT*0
    fshocks_tT[matbool_ashock_tT]  = shocks_tT[matbool_ashock_tT]
    fshocks_tT[matbool_avshock_tT] = shocks_tT[matbool_avshock_tT]
    PV = zeros(shape=(Rj.shape[0],1))
    for hh in range(tt+1,Horizon):
        PV = PV + dot(Rj[:,:,hh-tt-1],fshocks_tT[:,hh-tt].reshape(-1,1))

    return PV