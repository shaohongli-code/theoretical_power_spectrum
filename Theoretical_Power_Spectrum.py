import numpy as np
import sympy as sp
from scipy.interpolate import InterpolatedUnivariateSpline
import mcfit



###############################################################################################################
# legendre polynomial
import numpy as np
def L_ell(k,l):
    if k>l:
        t=k
        k=l
        l=t
    return np.prod(np.arange(l-k+1,l+1))/np.prod(np.arange(1,k+1))

def legendre_polynomial(x,l):
    if isinstance(l,int):
        LL=np.array([L_ell(i,l) for i in range(l+1)])
        k=np.arange(l+1)
        X,LL=np.meshgrid(x,LL)
        X,k=np.meshgrid(x,k)
        if isinstance(x[0],list) or isinstance(x[0],np.ndarray):
            return (np.sum(1/2**l*LL**2*(X-1)**(l-k)*(X+1)**k,0)).reshape(np.size(x,0),np.size(x,1))
        else:return (np.sum(1/2**l*LL**2*(X-1)**(l-k)*(X+1)**k,0))
    elif isinstance(l,list) or isinstance(l,np.ndarray):
        lg_ell=[]
        for l in l:
            LL=np.array([L_ell(i,l) for i in range(l+1)])
            k=np.arange(l+1)
            X,LL=np.meshgrid(x,LL)
            X,k=np.meshgrid(x,k)
            if isinstance(x[0],list) or isinstance(x[0],np.ndarray):
                lg_ell.append((np.sum(1/2**l*LL**2*(X-1)**(l-k)*(X+1)**k,0)).reshape(np.size(x,0),np.size(x,1)))
            else: lg_ell.append((np.sum(1/2**l*LL**2*(X-1)**(l-k)*(X+1)**k,0)))
        return np.array(lg_ell)
def LT(Pkmu,mu,l):
    if not (isinstance(mu[0],list) or isinstance(mu[0], np.ndarray)):
        mu = mu.reshape(len(mu),1)
    Dmu=np.sum(mu[1:]-mu[:-1])/len(mu[1:])
    lg=legendre_polynomial(mu,l)
    PP,LG=(Pkmu,lg)
    Sum=PP*LG
    if isinstance(l,int):
        return (2*l+1)/2*(np.sum(Sum,0)-1/2*(Sum[0]+Sum[-1]))*Dmu
    else:  
        Sum0=np.sum(Sum,1)
        for i in range(len(l)):
            Sum0[i]=(2*l[i]+1)/2*(Sum0[i]-1/2*(Sum[i][0]+Sum[i][-1]))*Dmu
        return Sum0
    
def legendre_polynomial_sympy(x,l):
    LL=np.array([L_ell(i,l) for i in range(l+1)])
    k=np.arange(l+1)
    return (np.sum(1/2**l*LL**2*(x-1)**(l-k)*(x+1)**k,0))

def mu_ns_ells_integrate(mu_ns,ells):
    mu_ns=mu_ns*np.array([1],dtype=np.int8)
    ells=ells*np.array([1],dtype=np.int8)
    mu=sp.symbols('mu')
    result={}
    for n in mu_ns:
        temp=np.array([])
        for ell in ells:
            temp=np.append(temp,(2*ell+1)/2*((legendre_polynomial_sympy(mu,ell)*mu**n).integrate(mu).replace(mu,1)-(legendre_polynomial_sympy(mu,ell)*mu**n).integrate(mu).replace(mu,-1)))
        result[n]=np.float64(temp.reshape(-1,1))
    return result
###############################################################################################################

###############################################################################################################
# see Table 3 and 4 of arXiv:2510.27227.
f_nm_ell_n_coeff={}

f_nm_ell_n_coeff['00']={'000':1219/1470,'02-2':1/6,
                            '11-1':62/35,
                            '200':671/1029,'22-2':1/3,
                            '31-1':8/35,
                            '400':32/1715}

f_nm_ell_n_coeff['10']={'000':41/42,'02-2':1/6,
                            '11-1':66/35,
                            '200':11/21,'22-2':1/3,
                            '31-1':4/35}

f_nm_ell_n_coeff['02']={'000':-18/35,
                            '11-1':-2/5,
                            '200':22/49,
                            '31-1':2/5,
                            '400':16/245}

f_nm_ell_n_coeff['12']={'000':-38/105,
                            '11-1':-2/5,
                            '200':34/147,
                            '31-1':2/5,
                            '400':32/245}

f_nm_ell_n_coeff['22']={'000':11/14,'02-2':1/6,
                            '11-1':62/35,
                            '200':5/7,'22-2':1/3,
                            '31-1':8/35}

f_nm_ell_n_coeff['30']={'000':-14/3,
                            '11-1':-38/5,
                            '200':-4/3,'22-2':-2,
                            '31-1':-2/5}

f_nm_ell_n_coeff['13']={'000':8/3,
                            '11-1':4,
                            '200':1/3,'22-2':1}

f_nm_ell_n_coeff['32']={'000':-112/15,'02-2':8/3,
                            '11-1':-16/5,
                            '200':152/21,'22-2':-8/3,
                            '31-1':16/5,
                            '400':8/35}

f_nm_ell_n_coeff['01']={'000':1003/1470,'02-2':1/6,
                            '11-1':58/35,
                            '200':803/1029,'22-2':1/3,
                            '31-1':12/35,
                            '400':64/1715}

f_nm_ell_n_coeff['11']={'000':851/1470,'02-2':1/6,
                            '11-1':54/35,
                            '200':871/1029,'22-2':1/3,
                            '31-1':16/35,
                            '400':128/1715}

f_nm_ell_n_coeff['20']={'000':356/105,'02-2':2/3,
                            '11-1':50/7,
                            '200':374/147,'22-2':4/3,
                            '31-1':6/7,
                            '400':16/245}

f_nm_ell_n_coeff['21']={'000':292/105,'02-2':2/3,
                            '11-1':234/35,
                            '200':454/147,'22-2':4/3,
                            '31-1':46/35,
                            '400':32/245}

f_nm_ell_n_coeff['03']={'000':4/3,'02-2':-2/3,
                            '11-1':2/5,
                            '200':-4/3,'22-2':2/3,
                            '31-1':-2/5}

f_nm_ell_n_coeff['31']={'000':-1/3,'02-2':1/3,
                            '200':1/3,'22-2':-1/3}

f_nm_ell_n_coeff['23']={'000':8/5,
                            '200':-16/7,
                            '400':24/35}

f_nm_ell_n_coeff['33']={'000':168/5,
                            '11-1':288/5,
                            '200':96/7,'22-2':16,
                            '31-1':32/5,
                            '400':24/35}



h_nm_ell_n_coeff={}

h_nm_ell_n_coeff['00']={'000':17/21,
                            '11-1':1,
                            '200':4/21}

h_nm_ell_n_coeff['01']={'000':1}

h_nm_ell_n_coeff['10']={'000':13/21,
                            '11-1':1,
                            '200':8/21}

h_nm_ell_n_coeff['11']={'000':1,
                            '11-1':1}

h_nm_ell_n_coeff['20']={'000':-1/3,
                            '200':1/3}

h_nm_ell_n_coeff['30']={'000':5/3,
                            '11-1':2,
                            '200':1/3}



h_s_nm_ell_n_coeff={}

h_s_nm_ell_n_coeff['00']={'000':8/315,
                            '11-1':4/15,
                            '200':254/441,
                            '31-1':2/5,
                            '400':16/245}

h_s_nm_ell_n_coeff['01']={'000':4/45,
                            '200':8/63,
                            '400':8/35}

h_s_nm_ell_n_coeff['02']={'200':2/3}

h_s_nm_ell_n_coeff['10']={'000':16/315,
                            '11-1':4/15,
                            '200':214/441,
                            '31-1':2/5,
                            '400':32/245}

h_s_nm_ell_n_coeff['11']={'11-1':4/15,
                            '200':2/3,
                            '31-1':2/5}

h_s_nm_ell_n_coeff['20']={'000':2/45,
                            '200':-10/63,
                            '400':4/35}

h_s_nm_ell_n_coeff['30']={'000':2/45,
                            '11-1':8/15,
                            '200':74/63,
                            '31-1':4/5,
                            '400':4/35}


######
# notice: sigma4_2=sigma4_20*64/15
sigma4_20_ell_n_coeff={'000':47/60,'02-2':5/12,
                            '11-1':11/5,
                            '200':26/21,'22-2':1/3,
                            '31-1':4/5,
                            '400':8/35}


g_nm_ell_n_coeff={'00': [[2, -2, -1/6],
    [1, -1, -23/378],
    [3, -1, 19/63],
    [0, 0, 5/63],
    [2, 0, -23/378],
    [1, 1, -11/54],
    [3, 1, 1/9]],
    '01': [[2, -2, -1/6],
    [1, -1, 17/378],
    [3, -1, 17/63],
    [0, 0, 1/63],
    [2, 0, -55/378],
    [1, 1, -37/378],
    [3, 1, 5/63]],
    '10': [[0, -2, -1/18],
    [1, -1, -1/18],
    [0, 0, 2/63],
    [2, 0, 5/14],
    [1, 1, -25/126],
    [3, 1, -4/21],
    [0, 2, -1/18],
    [2, 2, 1/6]],
    '11': [[2, -2, -1/6],
    [1, -1, 19/126],
    [3, -1, 5/21],
    [0, 0, -1/21],
    [2, 0, -29/126],
    [1, 1, 1/126],
    [3, 1, 1/21]],
    '02': [[1, -1, 1/2],
    [3, -1, -1/2],
    [0, 0, -3/7],
    [2, 0, -1/7],
    [4, 0, 4/7],
    [1, 1, 1/2],
    [3, 1, -1/2]],
    '20': [[2, -2, -1],
    [1, -1, 5/14],
    [3, -1, 37/14],
    [0, 0, 3/7],
    [2, 0, -12/7],
    [4, 0, -12/7],
    [1, 1, -1/2],
    [3, 1, 3/2]],
    'sigma3_2': [[0, 0, 35/24],
    [2, 0, -5/2],
    [4, 0, 15/8],
    [1, 1, 5/6],
    [3, 1, -5/2],
    [0, 2, -5/12],
    [2, 2, 5/4]]}

def alpha_ell_ell(ell1,ell2):
    if ((ell1+1)//(ell2+1)) * ((ell1+ell2+1)%2) :
        return np.math.factorial(ell1)/(2**((ell1-ell2)/2)*np.math.factorial((ell1-ell2)//2)*np.prod(np.arange((ell1+ell2+1)%2,ell1+ell2+2,2)))
    else:
        return 0
###############################################################################################################
class kernal_FFTlog(object):
    def __init__(self,Plin_input,k_input):

        xi_ell_n_all={}
        self.P_ell_n={}
        for ell_n_coeff in g_nm_ell_n_coeff.values():
            # print(coeffs)
            for ell1,n_q,coeff in zip(*(np.array(ell_n_coeff).T)):
                n_q=np.int8(n_q)
                ell1=np.int8(ell1)
                for ell2 in range(np.int8(ell1)%2,np.int8(ell1)+1,2):
                    
                    ell_n=str(ell2)+str(n_q)

                    if ell_n in xi_ell_n_all.keys():continue

                    xi = mcfit.P2xi(k_input, l=ell2, lowring=True)
                    xi_ell_n=xi(k_input**n_q*Plin_input, extrap=False)
                    xi_ell_n_all[ell_n]=xi_ell_n

                    if ell_n=='31':qq=0.01
                    elif ell_n=='11':qq=0.5
                    elif ell_n=='02':qq=0.5
                    elif ell_n=='22':qq=0.001
                    elif ell_n=='20':qq=0.5
                    elif ell_n=='40':qq=0.5
                    else:qq=1.5
                    P=mcfit.xi2P(xi_ell_n[0], l=ell2, lowring=True,q=qq)
                    P_ell_n=P(xi_ell_n[1]/xi_ell_n[0]*(1.j)**ell2, extrap=False)
                    self.P_ell_n[ell_n]=InterpolatedUnivariateSpline(P_ell_n[0],P_ell_n[1]*(-1.j)**ell2*P_ell_n[0]**(-n_q)/(4*np.pi))(k_input)



        ell_n1_n2=[[0,0,0],[0,2,-2],[1,1,-1],[2,0,0],[2,2,-2],[3,1,-1],[4,0,0]]
        self.P_ell_n1_n2={}
        qq=0.4
        for NN in ell_n1_n2:
            r,xi0=xi_ell_n_all[str(NN[0])+str(NN[1])]
            xi1=xi_ell_n_all[str(NN[0])+str(NN[2])][1]
            xi_xi=xi0*xi1
            P_temp=mcfit.xi2P(r, l=0, lowring=True,q=qq)
            PP=P_temp(xi_xi, extrap=False)
            self.P_ell_n1_n2[str(NN[0])+str(NN[1])+str(NN[2])]=InterpolatedUnivariateSpline(PP[0],PP[1])(k_input)
        self.k_input=k_input
    def J_nm(self,nm,k_output=None):
        J_nm0=np.zeros_like(self.k_input)
        ell_n_coeff=g_nm_ell_n_coeff[nm]
        for ell1,n_q,coeff in zip(*(np.array(ell_n_coeff).T)):
            n_q=np.int8(n_q)
            ell1=np.int8(ell1)
            for ell2 in range(np.int8(ell1)%2,np.int8(ell1)+1,2):
                ell_n=str(ell2)+str(n_q)
                J_nm0=J_nm0+self.P_ell_n[ell_n]*(alpha_ell_ell(ell1,ell2)*(2*ell2+1)*coeff)
        if k_output is None:
            return [self.k_input,J_nm0]
        else:
            J_nm_func=InterpolatedUnivariateSpline(self.k_input,J_nm0)
            return [k_output,J_nm_func(k_output)]

    def I_nm(self,nm,k_output=None):
        I_nm0=np.zeros_like(self.k_input)
        for ell_n1_n2 in f_nm_ell_n_coeff[nm]:
            I_nm0+=f_nm_ell_n_coeff[nm][ell_n1_n2]*self.P_ell_n1_n2[ell_n1_n2]
        if k_output is None:
            return [self.k_input,I_nm0]
        else:
            I_nm_func=InterpolatedUnivariateSpline(self.k_input,I_nm0)
            return [k_output,I_nm_func(k_output)]
    def K_nm(self,nm,k_output=None):
        K_nm0=np.zeros_like(self.k_input)
        for ell_n1_n2 in h_nm_ell_n_coeff[nm]:
            K_nm0+=h_nm_ell_n_coeff[nm][ell_n1_n2]*self.P_ell_n1_n2[ell_n1_n2]
        if k_output is None:
            return [self.k_input,K_nm0]
        else:
            K_nm_func=InterpolatedUnivariateSpline(self.k_input,K_nm0)
            return [k_output,K_nm_func(k_output)]
    def K_s_nm(self,nm,k_output=None):
        K_s_nm0=np.zeros_like(self.k_input)
        for ell_n1_n2 in h_s_nm_ell_n_coeff[nm]:
            K_s_nm0+=h_s_nm_ell_n_coeff[nm][ell_n1_n2]*self.P_ell_n1_n2[ell_n1_n2]
        if k_output is None:
            return [self.k_input,K_s_nm0]
        else:
            K_s_nm_func=InterpolatedUnivariateSpline(self.k_input,K_s_nm0)
            return [k_output,K_s_nm_func(k_output)]
    def sigma4_2(self,k_min=1e-3,k_max=50):
        index=(self.k_input>=k_min)&(self.k_input<k_max)
        k=self.k_input[index]
        kernel=np.zeros_like(k)
        for ell_n1_n2 in sigma4_20_ell_n_coeff:
            kernel+=sigma4_20_ell_n_coeff[ell_n1_n2]*self.P_ell_n1_n2[ell_n1_n2][index]
        kernel=kernel*64/15/k**2
        return 1/(24*np.pi**2)*np.sum((kernel[:-1]+kernel[1:])*(k[1:]-k[:-1]))/2
    
###############################################################################################################
###############################################################################################################
# AP effect
def P_AP_effect(k,mu,alpha_p,alpha_v,r_s_ratio):
    # r_s_ratio=r_s_fid/r_s
    k,mu=np.meshgrid(k,mu,sparse=True)
    F=alpha_p/alpha_v
    k_th_ap=k/alpha_v*(1+mu**2*(1/F**2-1))**(1/2)
    mu_ap=mu/F*(1+mu**2*(1/F**2-1))**(-1/2)
    return k_th_ap,mu_ap,r_s_ratio**3/(alpha_v**2*alpha_p)

###############################################################################################################
###############################################################################################################

class Plk_theory(object):
    def __init__(self,Plin_func,k_output,k_min=1e-5,k_max=1e4,k_bin=2048,k_value_max=100,sigma_v_2_lin=False):

        k=np.logspace(np.log10(k_min),np.log10(k_max),k_bin)
        if k_output is None:
            k_output=k
        Plin=np.zeros_like(k)
        Plin[k<=k_value_max]=Plin_func(k[k<=k_value_max])

        kernal_FFTlog_func=kernal_FFTlog(Plin_input=Plin,k_input=k)
        self.J_nm={}
        for nm in g_nm_ell_n_coeff.keys():
            if nm=='sigma3_2': 
                kk,sigma3_2=kernal_FFTlog_func.J_nm(nm,k_output)
                self.sigma3_2=(kk**2*sigma3_2)
            else:
                self.J_nm[nm]=kernal_FFTlog_func.J_nm(nm,k_output)[1]
        self.I_nm={}
        for nm in f_nm_ell_n_coeff.keys():
            self.I_nm[nm]=kernal_FFTlog_func.I_nm(nm,k_output)[1]
        self.K_nm={}
        for nm in h_nm_ell_n_coeff.keys():
            self.K_nm[nm]=kernal_FFTlog_func.K_nm(nm,k_output)[1]
        self.K_s_nm={}
        for nm in h_s_nm_ell_n_coeff.keys():
            self.K_s_nm[nm]=kernal_FFTlog_func.K_s_nm(nm,k_output)[1]
        self.sigma4_2=kernal_FFTlog_func.sigma4_2(k_max=k_value_max)
        q=np.logspace(np.log10(k_min),np.log10(k_value_max),2048)
        kernel=q**2*Plin_func(q)**2
        K_norm=1/(2*np.pi**2)*np.sum((kernel[1:]+kernel[:-1])*(q[1:]-q[:-1]))/2

        self.K_nm['01']=self.K_nm['01']-K_norm
        self.K_s_nm['01']=self.K_s_nm['01']-4/9*K_norm
        self.K_s_nm['02']=self.K_s_nm['02']-2/3*K_norm
        
        if sigma_v_2_lin:
            k_temp=k[k<=k_value_max]
            kernal=Plin[k<=k_value_max]
            self.sigma_v_2_lin_0=1/6/np.pi**2*np.sum((kernal[1:]+kernal[:-1])*(k_temp[1:]-k_temp[:-1]))/2

        self.Plin=Plin_func(k_output)
        self.k=k_output
        
        

    def P_nm(self,f,D,b1,b2,sigma_v_2,bs=None,b3nl=None,sigma_v_1_2=0,sigma_v_2_2=0,sigma4_2=None):
        
        if bs is None:bs=-4/7*(b1-1)
        if b3nl is None:b3nl=32/315*(b1-1)
        if sigma4_2 is not None:self.sigma4_2=sigma4_2

        if sigma_v_2 == 'lin':
            sigma_v_2=D**2*self.sigma_v_2_lin_0
            self.sigma_v_2=sigma_v_2

        self.P_00=b1**2*D**2*(self.Plin+2*D**2*(self.I_nm['00']+3*self.k**2*self.Plin*self.J_nm['00']))+2*b1*D**4*(b2*self.K_nm['00']+bs*self.K_s_nm['00']+b3nl*self.sigma3_2*self.Plin)+D**4*(1/2*b2**2*self.K_nm['01']+1/2*bs**2*self.K_s_nm['01']+b2*bs*self.K_s_nm['02'])

        self.P_01=f*b1*D**2*(self.Plin+2*D**2*(self.I_nm['01']+b1*self.I_nm['10']+3*self.k**2*self.Plin*(self.J_nm['01']+b1*self.J_nm['10']))-b2*D**2*self.K_nm['11']-bs*D**2*self.K_s_nm['11'])-f*D**4*(b2*self.K_nm['10']+bs*self.K_s_nm['10']+b3nl*self.sigma3_2*self.Plin)

        self.P_02_mu0=f**2*b1*D**4*(self.I_nm['02']+2*self.k**2*self.Plin*self.J_nm['02'])-f**2*self.k**2*(sigma_v_2+sigma_v_1_2/f**2)*self.P_00+f**2*D**4*(b2*self.K_nm['20']+bs*self.K_s_nm['20'])

        self.P_02_mu2=f**2*b1*D**4*(self.I_nm['20']+2*self.k**2*self.Plin*self.J_nm['20'])+f**2*D**4*(b2*self.K_nm['30']+bs*self.K_s_nm['30'])

        self.P_03=-f**2*self.k**2*(sigma_v_2+sigma_v_2_2/f**2)*self.P_01

        self.P_04_mu0=-1/2*f**4*b1*self.k**2*(sigma_v_2+sigma_v_1_2/f**2)*D**4*(self.I_nm['02']+2*self.k**2*self.Plin*self.J_nm['02'])+1/4*f**4*b1**2*self.k**4*self.P_00*((sigma_v_2+sigma_v_1_2/f**2)**2+D**4*self.sigma4_2)

        self.P_04_mu2=-1/2*f**4*b1*self.k**2*(sigma_v_2+sigma_v_1_2/f**2)*D**4*(self.I_nm['20']+2*self.k**2*self.Plin*self.J_nm['20'])

        self.P_11_mu0=f**2*D**2*b1**2*D**2*self.I_nm['31']

        self.P_11_mu2=f**2*D**2*((self.Plin+D**2*(2*self.I_nm['11']+4*b1*self.I_nm['22']+b1**2*self.I_nm['13']+6*self.k**2*self.Plin*(self.J_nm['11']+2*b1*self.J_nm['10']))))

        self.P_12_mu0=f**3*D**4*(self.I_nm['12']-b1*self.I_nm['03']+2*self.k**2*self.Plin*self.J_nm['02'])-f**2*self.k**2*(sigma_v_2+sigma_v_1_2/f**2)*self.P_01+2*f**3*self.k**2*D**4*(sigma_v_2+sigma_v_1_2/f**2)*(self.I_nm['01']+self.I_nm['10']+3*self.k**2*self.Plin*(self.J_nm['01']+self.J_nm['10']))

        self.P_12_mu2=f**3*D**4*(self.I_nm['21']-b1*self.I_nm['30']+2*self.k**2*self.Plin*self.J_nm['20'])

        self.P_13_mu0=-f**2*self.k**2*f**2*D**2*((sigma_v_2+sigma_v_1_2/f**2)*b1**2*D**2*self.I_nm['31'])

        self.P_13_mu2=-f**2*self.k**2*f**2*D**2*((sigma_v_2+sigma_v_2_2/f**2)*(self.Plin+D**2*(2*self.I_nm['11']+4*b1*self.I_nm['22']+b1**2*self.I_nm['13']+6*self.k**2*self.Plin*(self.J_nm['11']+2*b1*self.J_nm['10']))))

        self.P_22_mu0=1/4*f**4*D**4*self.I_nm['23']+f**4*self.k**4*(sigma_v_2+sigma_v_1_2/f**2)**2*self.P_00-f**2*self.k**2*(sigma_v_2+sigma_v_1_2/f**2)*(2*self.P_02_mu0-f**2*D**4*(b2*self.K_nm['20']+bs*self.K_s_nm['20']))

        self.P_22_mu2=1/4*f**4*D**4*2*self.I_nm['32']-f**2*self.k**2*(sigma_v_2+sigma_v_1_2/f**2)*(2*self.P_02_mu2-f**2*D**4*(b2*self.K_nm['30']+bs*self.K_s_nm['30']))

        self.P_22_mu4=1/4*f**4*D**4*self.I_nm['33']

    def P_dd_ell(self,ells):
        if 'P_dd_mu_n' not in self.__dict__:
            self.P_dd_mu_n=mu_ns_ells_integrate(np.arange(0,9,2),ells)
        return self.P_dd_mu_n[0]*self.P_00 + self.P_22_mu4*self.P_dd_mu_n[8]/4 + self.P_dd_mu_n[6]*(self.P_04_mu2 + self.P_12_mu2 + self.P_13_mu2 + self.P_22_mu2/4) + self.P_dd_mu_n[4]*(self.P_02_mu2 + self.P_03 + self.P_04_mu0 + self.P_11_mu2 + self.P_12_mu0 + self.P_13_mu0 + self.P_22_mu0/4) + self.P_dd_mu_n[2]*(2*self.P_01 + self.P_02_mu0 + self.P_11_mu0)
    def P_pp_ell(self,a,H,ells):
        if 'P_pp_mu_n' not in self.__dict__:
            self.P_pp_mu_n=mu_ns_ells_integrate(np.arange(0,9,2),ells)
        return H**2*a**2*(self.P_pp_mu_n[0]*self.P_11_mu0 + self.P_22_mu4*self.P_pp_mu_n[6] + self.P_pp_mu_n[4]*(2*self.P_12_mu2 + 3*self.P_13_mu2 + self.P_22_mu2) + self.P_pp_mu_n[2]*(self.P_11_mu2 + 2*self.P_12_mu0 + 3*self.P_13_mu0 + self.P_22_mu0))/self.k**2
    def iP_dp_ell(self,a,H,ells):
        if 'iP_dp_mu_n' not in self.__dict__:
            self.iP_dp_mu_n=mu_ns_ells_integrate(np.arange(1,9,2),ells)
        return -(a*H/self.k)*(self.P_22_mu4*self.iP_dp_mu_n[7]/2 + self.iP_dp_mu_n[5]*(2*self.P_04_mu2 + 3/2*self.P_12_mu2 + 2*self.P_13_mu2 + self.P_22_mu2/2) + self.iP_dp_mu_n[3]*(self.P_02_mu2 + 3*self.P_03/2 + 2*self.P_04_mu0 + self.P_11_mu2 + 3/2*self.P_12_mu0 + 2*self.P_13_mu0 + self.P_22_mu0/2) + self.iP_dp_mu_n[1]*(self.P_01 + self.P_02_mu0 + self.P_11_mu0))
    
    def P_dd_interpolate(self,kind=3,ext=1):
        P_mu0=self.P_00
        P_mu8=self.P_22_mu4/4
        P_mu6=self.P_04_mu2 + self.P_12_mu2 + self.P_13_mu2 + self.P_22_mu2/4
        P_mu4=self.P_02_mu2 + self.P_03 + self.P_04_mu0 + self.P_11_mu2 + self.P_12_mu0 + self.P_13_mu0 + self.P_22_mu0/4
        P_mu2=2*self.P_01 + self.P_02_mu0 + self.P_11_mu0
        self.P_dd_mu0=InterpolatedUnivariateSpline(self.k,P_mu0,k=kind,ext=ext)
        self.P_dd_mu2=InterpolatedUnivariateSpline(self.k,P_mu2,k=kind,ext=ext)
        self.P_dd_mu4=InterpolatedUnivariateSpline(self.k,P_mu4,k=kind,ext=ext)
        self.P_dd_mu6=InterpolatedUnivariateSpline(self.k,P_mu6,k=kind,ext=ext)
        self.P_dd_mu8=InterpolatedUnivariateSpline(self.k,P_mu8,k=kind,ext=ext)

    def P_pp_interpolate(self,a,H,kind=3,ext=1):
        P_mu0=(a*H/self.k)**2*self.P_11_mu0
        P_mu6=(a*H/self.k)**2*self.P_22_mu4
        P_mu4=((a*H/self.k)**2*(2*self.P_12_mu2 + 3*self.P_13_mu2 + self.P_22_mu2))
        P_mu2=((a*H/self.k)**2*(self.P_11_mu2 + 2*self.P_12_mu0 + 3*self.P_13_mu0 + self.P_22_mu0))
        self.P_pp_mu0=InterpolatedUnivariateSpline(self.k,P_mu0,k=kind,ext=ext)
        self.P_pp_mu2=InterpolatedUnivariateSpline(self.k,P_mu2,k=kind,ext=ext)
        self.P_pp_mu4=InterpolatedUnivariateSpline(self.k,P_mu4,k=kind,ext=ext)
        self.P_pp_mu6=InterpolatedUnivariateSpline(self.k,P_mu6,k=kind,ext=ext)

    def iP_dp_interpolate(self,a,H,kind=3,ext=1):
        P_mu7=(-a*H/self.k)*self.P_22_mu4/2
        P_mu5=(-a*H/self.k)*(2*self.P_04_mu2 + 3/2*self.P_12_mu2 + 2*self.P_13_mu2 + self.P_22_mu2/2)
        P_mu3=(-a*H/self.k)*(self.P_02_mu2 + 3*self.P_03/2 + 2*self.P_04_mu0 + self.P_11_mu2 + 3/2*self.P_12_mu0 + 2*self.P_13_mu0 + self.P_22_mu0/2)
        P_mu1=(-a*H/self.k)*(self.P_01 + self.P_02_mu0 + self.P_11_mu0)
        self.iP_dp_mu1=InterpolatedUnivariateSpline(self.k,P_mu1,k=kind,ext=ext)
        self.iP_dp_mu3=InterpolatedUnivariateSpline(self.k,P_mu3,k=kind,ext=ext)
        self.iP_dp_mu5=InterpolatedUnivariateSpline(self.k,P_mu5,k=kind,ext=ext)
        self.iP_dp_mu7=InterpolatedUnivariateSpline(self.k,P_mu7,k=kind,ext=ext)

    def P_dd_ell_AP_effect(self,ells,k,mu,alpha_p,alpha_v,r_s_ratio):
        k_ap,mu_ap,V_correct=P_AP_effect(k,mu,alpha_p,alpha_v,r_s_ratio)
        Pkmu=self.P_dd_mu0(k_ap)+self.P_dd_mu2(k_ap)*mu_ap**2+self.P_dd_mu4(k_ap)*mu_ap**4+self.P_dd_mu6(k_ap)*mu_ap**6+self.P_dd_mu8(k_ap)*mu_ap**8
        return V_correct*LT(Pkmu,mu,ells)
    def P_pp_ell_AP_effect(self,ells,k,mu,alpha_p,alpha_v,r_s_ratio):
        k_ap,mu_ap,V_correct=P_AP_effect(k,mu,alpha_p,alpha_v,r_s_ratio)
        Pkmu=self.P_pp_mu0(k_ap)+self.P_pp_mu2(k_ap)*mu_ap**2+self.P_pp_mu4(k_ap)*mu_ap**4+self.P_pp_mu6(k_ap)*mu_ap**6
        return V_correct*LT(Pkmu,mu,ells)
    def iP_dp_ell_AP_effect(self,ells,k,mu,alpha_p,alpha_v,r_s_ratio):
        k_ap,mu_ap,V_correct=P_AP_effect(k,mu,alpha_p,alpha_v,r_s_ratio)
        Pkmu=self.iP_dp_mu1(k_ap)*mu_ap+self.iP_dp_mu3(k_ap)*mu_ap**3+self.iP_dp_mu5(k_ap)*mu_ap**5+self.iP_dp_mu7(k_ap)*mu_ap**7
        return V_correct*LT(Pkmu,mu,ells)
    
    def P_nm_derivative(self,f,D,b1,b2,sigma_v_2,bs=None,b3nl=None,sigma_v_1_2=0,sigma_v_2_2=0,sigma4_2=None):
        
        if bs is None:bs=-4/7*(b1-1)
        if b3nl is None:b3nl=32/315*(b1-1)
        if sigma4_2 is not None:self.sigma4_2=sigma4_2

        if sigma_v_2 == 'lin':
            sigma_v_2=D**2*self.sigma_v_2_lin_0
            self.sigma_v_2=sigma_v_2
        self.P_00_derivative= 0
        self.P_01_derivative= -D**4*(self.K_nm['10']*b2 + self.K_s_nm['10']*bs + self.Plin*b3nl*self.sigma3_2) + D**2*b1*(-D**2*self.K_nm['11']*b2 - D**2*self.K_s_nm['11']*bs + 2*D**2*(self.I_nm['01'] + self.I_nm['10']*b1 + 3*self.Plin*self.k**2*(self.J_nm['01'] + self.J_nm['10']*b1)) + self.Plin)
        self.P_02_mu0_derivative= -2*(sigma_v_2+sigma_v_1_2/f**2)*f*self.k**2*(2*D**4*b1*(self.K_nm['00']*b2 + self.K_s_nm['00']*bs + self.Plin*b3nl*self.sigma3_2) + D**4*(0.5*self.K_nm['01']*b2**2 + 0.5*self.K_s_nm['01']*bs**2 + self.K_s_nm['02']*b2*bs) + D**2*b1**2*(2*D**2*(self.I_nm['00'] + 3*self.J_nm['00']*self.Plin*self.k**2) + self.Plin)) + 2*D**4*b1*f*(self.I_nm['02'] + 2*self.J_nm['02']*self.Plin*self.k**2) + 2*D**4*f*(self.K_nm['20']*b2 + self.K_s_nm['20']*bs)
        self.P_02_mu2_derivative= 2*D**4*b1*f*(self.I_nm['20'] + 2*self.J_nm['20']*self.Plin*self.k**2) + 2*D**4*f*(self.K_nm['30']*b2 + self.K_s_nm['30']*bs)
        self.P_03_derivative= -(sigma_v_2+sigma_v_2_2/f**2)*f**2*self.k**2*(-D**4*(self.K_nm['10']*b2 + self.K_s_nm['10']*bs + self.Plin*b3nl*self.sigma3_2) + D**2*b1*(-D**2*self.K_nm['11']*b2 - D**2*self.K_s_nm['11']*bs + 2*D**2*(self.I_nm['01'] + self.I_nm['10']*b1 + 3*self.Plin*self.k**2*(self.J_nm['01'] + self.J_nm['10']*b1)) + self.Plin)) - 2*(sigma_v_2+sigma_v_2_2/f**2)*f*self.k**2*(-D**4*f*(self.K_nm['10']*b2 + self.K_s_nm['10']*bs + self.Plin*b3nl*self.sigma3_2) + D**2*b1*f*(-D**2*self.K_nm['11']*b2 - D**2*self.K_s_nm['11']*bs + 2*D**2*(self.I_nm['01'] + self.I_nm['10']*b1 + 3*self.Plin*self.k**2*(self.J_nm['01'] + self.J_nm['10']*b1)) + self.Plin))
        self.P_04_mu0_derivative= -2.0*(sigma_v_2+sigma_v_1_2/f**2)*D**4*b1*f**3*self.k**2*(self.I_nm['02'] + 2*self.J_nm['02']*self.Plin*self.k**2) + b1**2*f**3*self.k**4*((sigma_v_2+sigma_v_1_2/f**2)**2 + D**4*self.sigma4_2)*(2*D**4*b1*(self.K_nm['00']*b2 + self.K_s_nm['00']*bs + self.Plin*b3nl*self.sigma3_2) + D**4*(0.5*self.K_nm['01']*b2**2 + 0.5*self.K_s_nm['01']*bs**2 + self.K_s_nm['02']*b2*bs) + D**2*b1**2*(2*D**2*(self.I_nm['00'] + 3*self.J_nm['00']*self.Plin*self.k**2) + self.Plin))
        self.P_04_mu2_derivative= -2.0*(sigma_v_2+sigma_v_1_2/f**2)*D**4*b1*f**3*self.k**2*(self.I_nm['20'] + 2*self.J_nm['20']*self.Plin*self.k**2)
        self.P_11_mu0_derivative= 2*D**4*self.I_nm['31']*b1**2*f
        self.P_11_mu2_derivative= 2*D**2*f*(D**2*(2*self.I_nm['11'] + self.I_nm['13']*b1**2 + 4*self.I_nm['22']*b1 + 6*self.Plin*self.k**2*(2*self.J_nm['10']*b1 + self.J_nm['11'])) + self.Plin)
        self.P_12_mu0_derivative= 6*(sigma_v_2+sigma_v_1_2/f**2)*D**4*f**2*self.k**2*(self.I_nm['01'] + self.I_nm['10'] + 3*self.Plin*self.k**2*(self.J_nm['01'] + self.J_nm['10'])) - (sigma_v_2+sigma_v_1_2/f**2)*f**2*self.k**2*(-D**4*(self.K_nm['10']*b2 + self.K_s_nm['10']*bs + self.Plin*b3nl*self.sigma3_2) + D**2*b1*(-D**2*self.K_nm['11']*b2 - D**2*self.K_s_nm['11']*bs + 2*D**2*(self.I_nm['01'] + self.I_nm['10']*b1 + 3*self.Plin*self.k**2*(self.J_nm['01'] + self.J_nm['10']*b1)) + self.Plin)) - 2*(sigma_v_2+sigma_v_1_2/f**2)*f*self.k**2*(-D**4*f*(self.K_nm['10']*b2 + self.K_s_nm['10']*bs + self.Plin*b3nl*self.sigma3_2) + D**2*b1*f*(-D**2*self.K_nm['11']*b2 - D**2*self.K_s_nm['11']*bs + 2*D**2*(self.I_nm['01'] + self.I_nm['10']*b1 + 3*self.Plin*self.k**2*(self.J_nm['01'] + self.J_nm['10']*b1)) + self.Plin)) + 3*D**4*f**2*(-self.I_nm['03']*b1 + self.I_nm['12'] + 2*self.J_nm['02']*self.Plin*self.k**2)
        self.P_12_mu2_derivative= 3*D**4*f**2*(self.I_nm['21'] - self.I_nm['30']*b1 + 2*self.J_nm['20']*self.Plin*self.k**2)
        self.P_13_mu0_derivative= -4*(sigma_v_2+sigma_v_1_2/f**2)*D**4*self.I_nm['31']*b1**2*f**3*self.k**2
        self.P_13_mu2_derivative= -4*(sigma_v_2+sigma_v_2_2/f**2)*D**2*f**3*self.k**2*(D**2*(2*self.I_nm['11'] + self.I_nm['13']*b1**2 + 4*self.I_nm['22']*b1 + 6*self.Plin*self.k**2*(2*self.J_nm['10']*b1 + self.J_nm['11'])) + self.Plin)
        self.P_22_mu0_derivative= 4*(sigma_v_2+sigma_v_1_2/f**2)**2*f**3*self.k**4*(2*D**4*b1*(self.K_nm['00']*b2 + self.K_s_nm['00']*bs + self.Plin*b3nl*self.sigma3_2) + D**4*(0.5*self.K_nm['01']*b2**2 + 0.5*self.K_s_nm['01']*bs**2 + self.K_s_nm['02']*b2*bs) + D**2*b1**2*(2*D**2*(self.I_nm['00'] + 3*self.J_nm['00']*self.Plin*self.k**2) + self.Plin)) - (sigma_v_2+sigma_v_1_2/f**2)*f**2*self.k**2*(-4*(sigma_v_2+sigma_v_1_2/f**2)*f*self.k**2*(2*D**4*b1*(self.K_nm['00']*b2 + self.K_s_nm['00']*bs + self.Plin*b3nl*self.sigma3_2) + D**4*(0.5*self.K_nm['01']*b2**2 + 0.5*self.K_s_nm['01']*bs**2 + self.K_s_nm['02']*b2*bs) + D**2*b1**2*(2*D**2*(self.I_nm['00'] + 3*self.J_nm['00']*self.Plin*self.k**2) + self.Plin)) + 4*D**4*b1*f*(self.I_nm['02'] + 2*self.J_nm['02']*self.Plin*self.k**2) + 2*D**4*f*(self.K_nm['20']*b2 + self.K_s_nm['20']*bs)) - 2*(sigma_v_2+sigma_v_1_2/f**2)*f*self.k**2*(-2*(sigma_v_2+sigma_v_1_2/f**2)*f**2*self.k**2*(2*D**4*b1*(self.K_nm['00']*b2 + self.K_s_nm['00']*bs + self.Plin*b3nl*self.sigma3_2) + D**4*(0.5*self.K_nm['01']*b2**2 + 0.5*self.K_s_nm['01']*bs**2 + self.K_s_nm['02']*b2*bs) + D**2*b1**2*(2*D**2*(self.I_nm['00'] + 3*self.J_nm['00']*self.Plin*self.k**2) + self.Plin)) + 2*D**4*b1*f**2*(self.I_nm['02'] + 2*self.J_nm['02']*self.Plin*self.k**2) + D**4*f**2*(self.K_nm['20']*b2 + self.K_s_nm['20']*bs)) + D**4*self.I_nm['23']*f**3
        self.P_22_mu2_derivative= -(sigma_v_2+sigma_v_1_2/f**2)*f**2*self.k**2*(4*D**4*b1*f*(self.I_nm['20'] + 2*self.J_nm['20']*self.Plin*self.k**2) + 2*D**4*f*(self.K_nm['30']*b2 + self.K_s_nm['30']*bs)) - 2*(sigma_v_2+sigma_v_1_2/f**2)*f*self.k**2*(2*D**4*b1*f**2*(self.I_nm['20'] + 2*self.J_nm['20']*self.Plin*self.k**2) + D**4*f**2*(self.K_nm['30']*b2 + self.K_s_nm['30']*bs)) + 2.0*D**4*self.I_nm['32']*f**3
        self.P_22_mu4_derivative= D**4*self.I_nm['33']*f**3
    def P_pv_ell(self,a,H,f,ells):
        if 'P_pv_mu_n' not in self.__dict__:
            self.P_pv_mu_n=mu_ns_ells_integrate(np.arange(1,9,2),ells)
        return a*H*f/self.k*(self.P_22_mu4_derivative*self.P_pv_mu_n[7]/4 + self.P_pv_mu_n[5]*(self.P_04_mu2_derivative + self.P_12_mu2_derivative + self.P_13_mu2_derivative + self.P_22_mu2_derivative/4) + self.P_pv_mu_n[3]*(self.P_02_mu2_derivative + self.P_03_derivative + self.P_04_mu0_derivative + self.P_11_mu2_derivative + self.P_12_mu0_derivative + self.P_13_mu0_derivative + self.P_22_mu0_derivative/4) + self.P_pv_mu_n[1]*(2*self.P_01_derivative + self.P_02_mu0_derivative + self.P_11_mu0_derivative))
    def P_pv_interpolate(self,a,H,f,kind=3,ext=1):
        # P_mu0=self.P_00
        P_mu7=self.P_22_mu4_derivative/4
        P_mu5=self.P_04_mu2_derivative + self.P_12_mu2_derivative + self.P_13_mu2_derivative + self.P_22_mu2_derivative/4
        P_mu3=self.P_02_mu2_derivative + self.P_03_derivative + self.P_04_mu0_derivative + self.P_11_mu2_derivative + self.P_12_mu0_derivative + self.P_13_mu0_derivative + self.P_22_mu0_derivative/4
        P_mu1=2*self.P_01_derivative + self.P_02_mu0_derivative + self.P_11_mu0_derivative
        # self.P_pv_mu0=InterpolatedUnivariateSpline(self.k,P_mu0,k=kind,ext=ext)
        self.P_pv_mu1=InterpolatedUnivariateSpline(self.k,a*H*f/self.k*P_mu1,k=kind,ext=ext)
        self.P_pv_mu3=InterpolatedUnivariateSpline(self.k,a*H*f/self.k*P_mu3,k=kind,ext=ext)
        self.P_pv_mu5=InterpolatedUnivariateSpline(self.k,a*H*f/self.k*P_mu5,k=kind,ext=ext)
        self.P_pv_mu7=InterpolatedUnivariateSpline(self.k,a*H*f/self.k*P_mu7,k=kind,ext=ext)
    def P_pv_ell_AP_effect(self,ells,k,mu,alpha_p,alpha_v,r_s_ratio):
        k_ap,mu_ap,V_correct=P_AP_effect(k,mu,alpha_p,alpha_v,r_s_ratio)
        Pkmu=self.P_pv_mu1(k_ap)*mu_ap+self.P_pv_mu3(k_ap)*mu_ap**3+self.P_pv_mu5(k_ap)*mu_ap**5+self.P_pv_mu7(k_ap)*mu_ap**7
        return V_correct*LT(Pkmu,mu,ells)

