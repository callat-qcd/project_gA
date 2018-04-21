params = dict()
params['tau'] = 1
params['a09m310'] = dict()
params['a09m310']['seed'] = 'a09m310'
params['bs'] = False
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
