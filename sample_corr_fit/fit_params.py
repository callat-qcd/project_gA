params = dict()
params['tau'] = 1
params['bs'] = False

''' a09 ensembles '''
params['a09m400'] = dict()
params['a09m400']['seed'] = 'a09m400'
params['a09m400']['Nbs']  = 5000
params['a09m400']['t_min_max'] = {
    'proton':[8,18],
    'gA'    :[6,15],
    'gV'    :[7,15]
}
params['a09m400']['plt_range'] = {
    'proton':[6,20,0.5,0.6],
    'gA'    :[2.5,17.5,1.1,1.5],
    'gV'    :[4,17,1.015,1.035]
}
params['a09m400']['fit_ini'] = {
    'E_0'   :0.53411,
    'dE_10' :0.37459,
    'zs_0'  :7.2e-4,
    'zp_0'  :2.6e-3,
    'zs_1'  :8.7e-4,
    'zp_1'  :5.0e-3,
    'gA_00' :1.24,
    'gA_11' :1.09,
    'gA_10' :-.17,
    'gV_00' :1.02,
    'gV_11' :1.02,
    'gV_10' :-0.18,
    'dAss_0':-1.2e-6,
    'dAps_0':-5.3e-6,
    'dAss_1':-1.2e-7,
    'dAps_1':-2.0e-6,
    'dVss_0':9.0e-7,
    'dVps_0':3.9e-6,
    'dVss_1':-7.1e-11,
    'dVps_1':4.6e-7,
}

params['a09m310'] = dict()
params['a09m310']['seed'] = 'a09m310'
params['a09m310']['Nbs']  = 5000
params['a09m310']['t_min_max'] = {
    'proton':[9,16],
    'gA'    :[3,12],
    'gV'    :[7,17]
}
params['a09m310']['plt_range'] = {
    'proton':[7,18,0.47,0.55],
    'gA'    :[0,15,1.1,1.5],
    'gV'    :[5,20,1.015,1.035]
}
params['a09m310']['fit_ini'] = {
    'E_0'   :.49241,
    'dE_10' :.408,
    'zs_0'  :2.2e-5,
    'zp_0'  :2.3e-3,
    'zs_1'  :1.3e-5,
    'zp_1'  :3.1e-3,
    'gA_00' :1.27,
    'gV_00' :1.02,
    'gA_11' :1.55,
    'gA_10' :-.01,
    'dAss_0':-4.7e-10,
    'dAps_0':-6.2e-8,
    'dAss_1':-4.2e-10,
    'dAps_1':1.5e-8,
    'gV_11' :1.06,
    'gV_10' :-0.066,
    'dVss_0':3.9e-10,
    'dVps_0':3.1e-8,
    'dVss_1':2.6e-11,
    'dVps_1':-4.1e-10,
}
