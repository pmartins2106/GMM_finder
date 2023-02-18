# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:50:49 2023

@author: pmartins
"""

# from scipy import stats 
# from six.moves import urllib




# def acronyms_list():
    # """
    # Scraped distribution equations from the SciPy website. Duplicates are 
    # removed.
    
    # """
    
    # acronyms = [
    #     'LSpI',
    #     'LCaI',
    #     'LMx(Sp>Ca)I',
    #     'LMx(Sp<Ca)I',
    #     'LMx(Sp=Ca)I',
    #     'HSpI',
    #     'HMx(Sp>Ca)I',
    #     'HCaI',
    #     'HMx(Sp<Ca)I',
    #     'HMx(Sp=Ca)I',
    #     'HMxD(I/A)',
    #     'HCaA',
    #     'HMx(Sp>Ca)A',
    #     'HMxD(A/I)',
    #     'HSpA',
    #     'HMx(Sp<Ca)A',
    #     'HMx(Sp=Ca)I'
    #      ]
         
    # return acronyms



def names_list():
    """
    Scraped proper names.Duplicates removed.
    """


    names = [
            'Linear specific inhibition',
            'Linear catalytic inhibition',
            'Linear mixed, predominantly specific inhibition',
            'Linear mixed, predominantly catalytic inhibition',
            'Linear mixed, balanced inhibition',
            'Hyperbolic specific inhibition',
            'Hyperbolic mixed, predominantly specific inhibition',
            'Hyperbolic catalytic inhibition',
            'Hyperbolic mixed, predominantly catalytic inhibition',
            'Hyperbolic mixed, balanced inhibition',
            'Hyperbolic mixed, dual modification (inhibition -> activation)',
            'Hyperbolic catalytic activation',
            'Hyperbolic mixed, predominantly specific activation',
            'Hyperbolic mixed, dual modification (activation -> inhibition)',
            'Hyperbolic specific activation',
            'Hyperbolic mixed, predominantly catalytic activation',
            'Hyperbolic mixed, balanced activation'
                ]
    return names

def gmm_list_names():
    """
    """
    
    names =  ['LSpI',
    'LCaI',
    'LMx(Sp>Ca)I',
    'LMx(Sp<Ca)I',
    'LMx(Sp=Ca)I',
    'HSpI',
    'HMx(Sp>Ca)I',
    'HCaI',
    'HMx(Sp<Ca)I',
    'HMx(Sp=Ca)I',
    'HMxD(I/A)',
    'HCaA',
    'HMx(Sp>Ca)A',
    'HMxD(A/I)',
    'HSpA',
    'HMx(Sp<Ca)A',
    'HMx(Sp=Ca)I']
   
    return names


def gmm_urls():
    """ 
     """
    
    
    all_url_links = ['https://www.enzyme-modifier.ch/lspi-new/',
                    'https://www.enzyme-modifier.ch/lcai-new/',
                    'https://www.enzyme-modifier.ch/lmxspcai-new/',
                    'https://www.enzyme-modifier.ch/lmxspcai/',
                    'https://www.enzyme-modifier.ch/lmxsp-cai-new/',
                    'https://www.enzyme-modifier.ch/hspi-new/',
                    'https://www.enzyme-modifier.ch/hmxspcai-new/',
                    'https://www.enzyme-modifier.ch/hcai-new/',
                    'https://www.enzyme-modifier.ch/hmxsp-cai-new/',
                    'https://www.enzyme-modifier.ch/hmxsp-cai/',
                    'https://www.enzyme-modifier.ch/hmxdi-a-new/',
                    'https://www.enzyme-modifier.ch/hcaa-new/',
                    'https://www.enzyme-modifier.ch/hmxspcaa-new/',
                    'https://www.enzyme-modifier.ch/hmxda-i-new/',
                    'https://www.enzyme-modifier.ch/hspa-new/',
                    'https://www.enzyme-modifier.ch/hmxsp-caa-new/',
                    'https://www.enzyme-modifier.ch/hmxsp-caa-new-2/'
]
   
    return all_url_links



# def scipy_distribution_names_and_docstrings():
#     """
#     Compile a list of distributions with their names and names
#     of how to access docstrings.
    
#     However, this function returns some distributions multiple times
#     because they are repeated with the different parameter setup,
#     mostly with negative/positive parameter values.

#     Wanting to have only one function, I removed duplicates from 
#     the list that I'm using in web-app -- uncommented lines.

#     Returns
#     -------
#     access_docstrings : In the form ['stats.alpha.__doc__', 
#                                      ' ... ', ... ]
#     distribution_names : In the form ['alpha', 
#                                       '....', ...]

#     """
    
#     access_docstrings = []
#     distribution_names = []
#     for i, name in enumerate(sorted(stats._distr_params.distcont)):
#         docstrings = 'stats.'+str(name[0])+('.__doc__')
#         access_docstrings.append(docstrings)
#         distribution_names.append(name[0])
    
#     return access_docstrings, distribution_names


# def get_scipy_distribution_parameters():
#     """
#     Modified from: https://stackoverflow.com/questions/37559470/what-do-all-the-distributions-available-in-scipy-stats-look-like

#     Returns
#     -------
#     names, all_params_names, all_params

#     """
    
#     names, all_params_names, all_params = [], [], []

#     # Get names and parameters of each scipy distribution 
#     # from stats._distr_params.distcont 
#     for name, params in sorted(stats._distr_params.distcont):
        
#         names.append(name)
        
#         # Add loc and scale to the parameters as they are not listed
#         loc, scale = 0.00, 1.00
#         params = list(params) + [loc, scale]
#         all_params.append(params)
        
#         # Get distribution information
#         dist = getattr(stats, name)
        
#         # Join parameters of each distribution with loc and scale
#         p_names = ['loc', 'scale']
#         if dist.shapes:
#             p_names = [sh.strip() for sh in dist.shapes.split(',')] + ['loc', 'scale']
#             all_params_names.append(p_names)
#         else:
#             all_params_names.append(['loc', 'scale'])

#     return names, all_params_names, all_params



# This script goes throught the SciPy website and extracts the equations
# for each continuous distribution.
# Note: idx 87 - reciprocal - distribution doesn't have a webpage in 
# SciPy 1.6.1; So the equation is obtained from 1.4 version

# When using these equations make sure to check them, since I compiled a 
# list of them and noticed a number of errors, either due to multiple
# equations on the website, or due to missing equations

# These are crosschecked with the output from the function below this one
# And final equations are majority from the 
# function:extract_equations_from_docstrings()
# def scrape():
#     """ Get all url's to scipy sites; scrape equations"""
    
#     all_urls = []
#     for i, name in enumerate(sorted(stats._distr_params.distcont)):
#         # Skipping this because its reciprocal - doesn't have a page here
#         if i == 87:
#             pass
#         else:
#             add = 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.'+str(name[0])+'.html#scipy.stats.'+str(name[0])
#             all_urls.append(add)
    
    
#     distribution = []
#     for idx, i in enumerate(all_urls):
    
#         if idx == 87:
#             print(' \[f(x, a, b) = \frac{1}{ \left(x*log( \frac{b}{a} \right) }')
        
#         else:
#             html_doc = urllib.request.urlopen(i).read()
#             soup = BeautifulSoup(html_doc, 'html.parser')
      
    
#             divTag = soup.find_all("div", class_="math notranslate nohighlight")
    
    
#             for tag in divTag:
#                 distribution.append(tag.text)

# def extract_equations_from_docstrings():
#     """
#     First part gets all docstrings, and stores them in [all_names].
#     docstrings are listed in l which are then accessed (see below)
#     and from each, equation is extracted.
#     This requires later manual check, due to complexity of the 
#     structure around the equations.
#     """
#     all_names = []
#     for i, name in enumerate(sorted(stats._distr_params.distcont)):
#         add = 'stats.'+str(name[0])+('.__doc__')
#         all_names.append(add)
        
        
#     l = [stats.alpha.__doc__,
#      stats.anglit.__doc__,
#      stats.arcsine.__doc__,
#      stats.argus.__doc__,
#      stats.beta.__doc__,
#      stats.betaprime.__doc__,
#      stats.bradford.__doc__,
#      stats.burr.__doc__,
#      stats.burr12.__doc__,
#      stats.cauchy.__doc__,
#      stats.chi.__doc__,
#      stats.chi2.__doc__,
#      stats.cosine.__doc__,
#      stats.crystalball.__doc__,
#      stats.dgamma.__doc__,
#      stats.dweibull.__doc__,
#      stats.erlang.__doc__,
#      stats.expon.__doc__,
#      stats.exponnorm.__doc__,
#      stats.exponpow.__doc__,
#      stats.exponweib.__doc__,
#      stats.f.__doc__,
#      stats.fatiguelife.__doc__,
#      stats.fisk.__doc__,
#      stats.foldcauchy.__doc__,
#      stats.foldnorm.__doc__,
#      stats.gamma.__doc__,
#      stats.gausshyper.__doc__,
#      stats.genexpon.__doc__,
#      stats.genextreme.__doc__,
#      stats.gengamma.__doc__,
#      stats.gengamma.__doc__,
#      stats.genhalflogistic.__doc__,
#      stats.geninvgauss.__doc__,
#      stats.genlogistic.__doc__,
#      stats.gennorm.__doc__,
#      stats.genpareto.__doc__,
#      stats.gilbrat.__doc__,
#      stats.gompertz.__doc__,
#      stats.gumbel_l.__doc__,
#      stats.gumbel_r.__doc__,
#      stats.halfcauchy.__doc__,
#      stats.halfgennorm.__doc__,
#      stats.halflogistic.__doc__,
#      stats.halfnorm.__doc__,
#      stats.hypsecant.__doc__,
#      stats.invgamma.__doc__,
#      stats.invgauss.__doc__,
#      stats.invweibull.__doc__,
#      stats.johnsonsb.__doc__,
#      stats.johnsonsu.__doc__,
#      stats.kappa3.__doc__,
#      stats.kappa4.__doc__,
#      stats.kappa4.__doc__,
#      stats.kappa4.__doc__,
#      stats.kappa4.__doc__,
#      stats.ksone.__doc__,
#      stats.kstwo.__doc__,
#      stats.kstwobign.__doc__,
#      stats.laplace.__doc__,
#      stats.laplace_asymmetric.__doc__,
#      stats.levy.__doc__,
#      stats.levy_l.__doc__,
#      stats.levy_stable.__doc__,
#      stats.loggamma.__doc__,
#      stats.logistic.__doc__,
#      stats.loglaplace.__doc__,
#      stats.lognorm.__doc__,
#      stats.loguniform.__doc__,
#      stats.lomax.__doc__,
#      stats.maxwell.__doc__,
#      stats.mielke.__doc__,
#      stats.moyal.__doc__,
#      stats.nakagami.__doc__,
#      stats.ncf.__doc__,
#      stats.nct.__doc__,
#      stats.ncx2.__doc__,
#      stats.norm.__doc__,
#      stats.norminvgauss.__doc__,
#      stats.pareto.__doc__,
#      stats.pearson3.__doc__,
#      stats.powerlaw.__doc__,
#      stats.powerlognorm.__doc__,
#      stats.powernorm.__doc__,
#      stats.rayleigh.__doc__,
#      stats.rdist.__doc__,
#      stats.recipinvgauss.__doc__,
#      stats.reciprocal.__doc__,
#      stats.rice.__doc__,
#      stats.semicircular.__doc__,
#      stats.skewnorm.__doc__,
#      stats.t.__doc__,
#      stats.trapezoid.__doc__,
#      stats.triang.__doc__,
#      stats.truncexpon.__doc__,
#      stats.truncnorm.__doc__,
#      stats.truncnorm.__doc__,
#      stats.tukeylambda.__doc__,
#      stats.uniform.__doc__,
#      stats.vonmises.__doc__,
#      stats.vonmises_line.__doc__,
#      stats.wald.__doc__,
#      stats.weibull_max.__doc__,
#      stats.weibull_min.__doc__,
#      stats.wrapcauchy.__doc__]
    
    
#     # Get equations, tracing places after math
#     # The end of equation is not marked in any special way, so
#     # get more lines just in sace some equations are longer 
#     eq = []
#     reg = []
#     for j in range(len(sorted(l))):
#         for i, line in enumerate(l[j].split('\n')):
#             if 'math::' in line:
#                 eq.append(l[j].split('\n')[i+2:i+4])
#                 reg.append(l[j].split('\n')[i+2:i+6])
     
#     # Join each extraction together, this will remove the extra
#     # blank lines from the equation
#     class OurList(list): 
#         def join(self, s):
#             return s.join(self)
    
#     joined_eq = []
#     for i in eq:
#         li = OurList(i)
#         li = li.join('')
#         joined_eq.append(li)
    
#     #print(joined_eq)
#     return

# Creating dictionaries
def creating_dictionaries():
    """
    Final results from the helper functions are various dictionaries that
    I use in the code.
    For every dictionary, there is an example how it looks like.
    """
    
    # # Dictionary containing distribution name and how to access docstrings    
    # access_docstrings, distribution_names = scipy_distribution_names_and_docstrings()
    # name_docstring_dict = {distribution: access_docstrings[i] for i, distribution in enumerate(distribution_names)}
    
    """
    {'alpha': 'stats.alpha.__doc__',
     'anglit': 'stats.anglit.__doc__',
     ...
    """
    
    # # Dictionary containing distribution name and its PDF    
    # eq = acronyms_list()
    # name_eq_dict = {distribution: eq[i] for i, distribution in enumerate(gmm_list_names())}
    
    """
    {'alpha': 'f(x, a) = \\frac{1}{x^2 \\Phi(a) \\sqrt{2\\pi}} * \\exp(-\\frac{1}{2} (a-1/x)^2)',
     'anglit': 'f(x) = \\sin(2x + \\pi/2) = \\cos(2x)',
     ...
    """
    
    # Dictionary containing distribution name and its proper name    
    proper_names = names_list()
    url_names = gmm_urls()
    name_proper_dict = {mech: proper_names[i] for i, mech in enumerate(gmm_list_names())}
    name_url_dict = {mech: url_names[i] for i, mech in enumerate(gmm_list_names())}
    
    """
    {'alpha': 'Alpha distribution',
     'anglit': 'Anglit distribution',
     ...
    """
    
    # # Nested dictionary containing distribution name and its paramaters&values  
    # names, all_params_names, all_params = get_scipy_distribution_parameters()
    # all_dist_params_dict = {function: {param:  f"{all_params[i][j]:.2f}"  for j, param in enumerate(all_params_names[i])} for i, function in enumerate(names)}

    # """
    #     {'alpha': {'a': 3.57,
    #                'loc': 0.0,
    #                'scale': 1.0},
    #      'anglit': {'loc': 0.0,
    #                 'scale': 1.0},
    #      etc...
    # """

    # Names and url-s
    # name_url_dict = {distribution: [gmm_urls()[i], names_list()[i]] for i, distribution in enumerate(gmm_list_names())}

    """
    {'alpha': ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.alpha.html#scipy.stats.alpha',
               'Alpha distribution'],
     'anglit': ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anglit.html#scipy.stats.anglit',
                'Anglit distribution'],
     ...
    """


    return name_proper_dict, name_url_dict #name_docstring_dict, name_eq_dict, all_dist_params_dict, 