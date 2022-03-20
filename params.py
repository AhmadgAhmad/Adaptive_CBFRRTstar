import ConfigParser
import os

# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class Params(object):
    __instance = None
    def __new__(cls,configFile=None):
        if Params.__instance is None:
            Params.__instance = object.__new__(cls)



            config = ConfigParser.ConfigParser()
            config.read(configFile)

            # Define Simulation params
            Params.__instance.max_timesteps  = config.getint('sim','max_timesteps')
            Params.__instance.plot_sim       = config.getboolean('sim','plot_sim')
            Params.__instance.live_plots     = config.getboolean('sim','live_plots')
            Params.__instance.plot_cbf       = config.getboolean('sim','plot_cbf')
            Params.__instance.plot_constrs   = config.getboolean('sim','plot_constrs')
            Params.__instance.plot_clf       = config.getboolean('sim','plot_clf')
            Params.__instance.plot_delta     = config.getboolean('sim','plot_delta')
            Params.__instance.p_cbf          = config.getfloat('sim','p_cbf')
            
            # Decentralized True will make CBF ij and ji
            # Assigns priority to agents for better deconfliction
            Params.__instance.decentralized  = config.getboolean('sim','decentralized')

            # CLF Fields
            Params.__instance.epsilon = config.getfloat('clf','epsilon')
            Params.__instance.p = config.getfloat('clf','p')
            Params.__instance.gamma = config.getfloat('clf','gamma')

            # Dynamics Params
            Params.__instance.step_size = config.getfloat('dynamics','step_size') # seconds

            # HOCBF parameters
            Params.__instance.alpha_linear_p1 = config.getfloat('HOCBF_param','alpha_linear_p1')
            Params.__instance.alpha_linear_p2 = config.getfloat('HOCBF_param','alpha_linear_p2')
            Params.__instance.hocbf_ufn_en = config.getboolean('HOCBF_param', 'hocbf_ufn_en')

            # Unicycle Dynamics Params
            Params.__instance.v_upper_bound = config.getfloat('unicycle','v_upper_bound')
            Params.__instance.w_upper_bound = config.getfloat('unicycle','w_upper_bound')
            Params.__instance.vel_penalty = config.getfloat('unicycle','vel_penalty')
            Params.__instance.steer_penalty = config.getfloat('unicycle','steer_penalty')
            Params.__instance.l = config.getfloat('unicycle','l')
            Params.__instance.vInt_enabled = config.getboolean('unicycle', 'vInt_enabled')
            Params.__instance.wInt_enabled = config.getboolean('unicycle', 'wInt_enabled')

            # Unicycle extended Dynamics
            Params.__instance.we_upper_bound = config.getfloat('unicycle_extended', 'we_upper_bound')
            Params.__instance.mu_upper_bound = config.getfloat('unicycle_extended', 'mu_upper_bound')
            Params.__instance.acc_penalty = config.getfloat('unicycle_extended', 'acc_penalty')
            Params.__instance.steer_penalty = config.getfloat('unicycle_extended', 'steer_penalty')
            Params.__instance.wExt_enabled = config.getboolean('unicycle_extended', 'wExt_enabled')
            Params.__instance.muExt_enabled = config.getboolean('unicycle_extended', 'muExt_enabled')

            #Unicycle dynamics with actuators velocities
            Params.__instance.sl_upper_bound = config.getfloat('unicycle_actuators', 'sl_upper_bound')
            Params.__instance.sr_upper_bound = config.getfloat('unicycle_actuators', 'sr_upper_bound')
            Params.__instance.s_penalty = config.getfloat('unicycle_actuators', 's_penalty')
            Params.__instance.k = config.getfloat('unicycle_actuators', 'k')
            Params.__instance.R = config.getfloat('unicycle_actuators', 'R')

            # Single Integrator Dynamics
            Params.__instance.max_speed = config.getfloat('single_int','max_speed')

            # Double Integrator Dynamics
            Params.__instance.max_accel = config.getfloat('double_int','max_accel')

            #The mission space parameters:
            Params.__instance.length = config.getfloat('mission_space_dim', 'length')
            Params.__instance.width = config.getfloat('mission_space_dim', 'width')

            #CBF-RRTstr paramaters:
            Params.__instance.edge_length = config.getfloat('CBF_RRTstr', 'edge_length')
            Params.__instance.gamma = config.getfloat('CBF_RRTstr', 'gamma')
            Params.__instance.CBF_RRTstrEnable = config.getboolean('CBF_RRTstr', 'CBF_RRTstrEnable')
            Params.__instance.treeSample_enabled = config.getboolean('CBF_RRTstr', 'treeSample_enabled')

            #CE parameters:
            Params.__instance.rho = config.getfloat('CE', 'rho')
            Params.__instance.rce = config.getfloat('CE', 'rce')
            Params.__instance.md = config.getfloat('CE', 'md')
            Params.__instance.kgmm = config.getint('CE', 'kgmm')
            Params.__instance.kSamples = config.getfloat('CE', 'kSamples')
            Params.__instance.plot_pdf_kde = config.getboolean('CE', 'plot_pdf_kde')
            Params.__instance.plot_pdf_gmm = config.getboolean('CE', 'plot_pdf_gmm')
            Params.__instance.kde_enabled = config.getboolean('CE', 'kde_enabled')
            Params.__instance.CE_enabled = config.getboolean('CE', 'CE_enabled')
            Params.__instance.wSamples_enabled = config.getboolean('CE', 'wSamples_enabled')
            Params.__instance.plot_preElite = config.getboolean('CE', 'plot_preElite')
            Params.__instance.ordDict_enabled = config.getboolean('CE', 'ordDict_enabled')
            Params.__instance.kdT_enabled = config.getboolean('CE', 'kdT_enabled')
            Params.__instance.initTree_flag = config.getboolean('CE', 'initTree_flag')
            Params.__instance.dataGen_enabled = config.getboolean('CE', 'dataGen_enabled')
            Params.__instance.save_costData = config.getboolean('CE', 'save_costData')
            Params.__instance.save_tree = config.getboolean('CE', 'save_tree')
            Params.__instance.save_adapDist = config.getboolean('CE', 'save_adapDist')
            Params.__instance.plt_ex2g_Es = config.getboolean('CE', 'plt_ex2g_Es')


            #CBF-RRT parameters:
            Params.__instance.CBF_RRT_enabled = config.getboolean('CBF_RRT', 'CBF_RRT_enabled')

            #RRT* parameters: 
            Params.__instance.RRTstr_enabled = config.getboolean('RRTstr', 'RRTstr_enabled')
            
            #CBF-CLF paramters :
            Params.__instance.CBF_CLF_enabled = config.getboolean('CBF_CLF', 'CBF_CLF_enabled')
            


            #Debugging logs; 
            Params.__instance.debug_flag = config.getboolean('DEBUG_LOG', 'debug_flag')


        return Params.__instance
    