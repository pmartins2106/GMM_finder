# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:41:04 2023

@author: pmartins
"""

# Package imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
# curve-fit() function imported from scipy
from scipy.optimize import curve_fit
import math
# List of GMMS
from gmm_list import gmm_list_names,creating_dictionaries
import time
import base64
import collections


def page_find():
    """
    The find page in this app is made with Streamlit for fitting a GMM
    a GMM mechanism to the User imported data.
    """
    name_proper_dict, name_url_dict = creating_dictionaries()
       
    st.sidebar.info("""
                Import enyme kinetics data and discover the mechanism of enzyme 
                inhibiton/activation.
                """)
                
    # Add a bit empy space before showing About
    st.sidebar.text("")
    
    # Input enzyme concentration
    Enzyme = st.sidebar.number_input('Enzyme concentration [E]', format="%2.2e", value = 1e0,min_value = 1e-20)
    # Ryn full analysis or step-by-step analysis
    st.sidebar.markdown("**Type of Analysis:**")
    analysis_mode = st.sidebar.radio('', ('Run All', 'Step-by-Step'))   

    st.sidebar.text("")
    st.sidebar.text("")

    st.markdown("<h1 style='text-align: center;'> Find the General Modifier Mechanism </h1>", 
                unsafe_allow_html=True)

    # Using cache to load the data only once 
    # @st.cache_data
    def load_csv():
        """ Get the loaded .csv into Pandas dataframe. """
        
        # df_load = pd.read_csv(input, sep=',' , engine='python',
        #                       nrows=25, skiprows=1, encoding='utf-8')
        df_load = pd.read_excel(input, nrows=25, skiprows=5,  engine="odf")
        return df_load
    

    # Streamlit - upload a file
    input = st.file_uploader('')
 
    # if 'run_example' not in st.session_state:
    st.session_state.run_example = False
        
    # run example option
    st.session_state.run_example = st.checkbox('Run a prefilled example') 
        
    # Ask for upload if not yet or run example.       
    if input is None:
        st.write('Upload your data, or:')
        # Get the template  
        download_sample = st.checkbox("Download template")
      
    try:
        if download_sample:
            with open("datasets//GMM_Finder_template.ods", "rb") as fp:
                st.download_button(
                label="Download",
                data=fp,
                file_name="GMM_Finder_template.ods",
                mime="application/vnd.ms-excel"
                )
            st.markdown("""**fill the template; save as comma Separated (.csv) file; upload the .csv file*""")
           
            # if run button is pressed
        if st.session_state.run_example:
            input = "datasets//GMM_Finder_example.ods"
            # Pass to a function above so we can use st.cache
            df = load_csv()
            # Replace inf/-inf with NaN and remove NaN if present
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            df = df.replace([np.inf, -np.inf], np.nan).dropna() #[data_col]
            #convert to [v]/[E]
            cols = df.columns[1:]
            df[cols] = df[cols] / Enzyme
               
            st.info('Uploaded Data')
            st.dataframe(df)
         
    except:
        # If the user imports file - parse it
       if input:
           with st.spinner('Loading data...'):
                    # Pass to a function above so we can use st.cache
                    df = load_csv()
                    # Replace inf/-inf with NaN and remove NaN if present
                    df = df.replace([np.inf, -np.inf], np.nan).dropna() #[data_col]
                    st.info('Uploaded Data')
                    st.dataframe(df)
                    # csv_downloader(results_to_dataframe(df, results))
                
               
    # def plot(df, data_stat):
    #     """ 
    #     Histogram of the input data. Contains also information about the 
    #     Figure style, depending on the active Mode.
    #     """
        
    #     if analysis_mode == 'Light Mode':
    #         hist_edge_color = 'black'
    #         hist_color= 'white'
    #         quant_color = 'black'
    #         median_color = 'black'
    #         pdf_color = '#08519c'
    #         cdf_color = 'black'
    #         plt.style.use('classic')
    #         plt.rcParams['figure.facecolor'] = 'white'


    #     if analysis_mode == 'Dark Mode':
    #         hist_edge_color = 'black'
    #         hist_color= 'white'
    #         median_color = 'magenta'
    #         quant_color = 'white'
    #         pdf_color = '#fec44f'
    #         cdf_color = 'white'
    #         plt.style.use('dark_background')
    #         plt.rcParams['figure.facecolor'] = 'black'

    #     fig, ax = plt.subplots(1,1)
    ## ....           
    #     return fig
    
    #plot experimental and fitted data
    def plot_fit(p_x, p_xfit, p_y, p_yfit, i, colors):
        # fig_fit1 = plt.figure()
        plt.style.use("dark_background")
        plt.scatter(p_x, p_y, color=colors[i], label=df.columns[i+1])
        plt.plot(p_xfit, p_yfit, color=colors[i])
        plt.xlabel('[S]')
        plt.ylabel('[v] / [E]')
        # return fig_fit1
        

    #Fit Michaelis-Menten equations to data
    def MMeq(x, p_kcat, p_Km):
        y = p_kcat*x/(p_Km+x)
        return y
    
    @st.cache_data()
    def fit_data(df):
        """ 
        Fit data:
        """
        
        fig_fit = plt.figure()
        # If the distribution(s) are selected in the selectbox
        # if data_fit_bt:
            
        # NS = df.shape[0] #number of substrate concentrations
        NX = df.shape[1]-1 #number of modifier concentrations
        
        # Check for nan/inf and remove them
        ## Get histogram of the data and histogram parameters
        num_bins = round(math.sqrt(len(df)))
        hist, bin_edges = np.histogram(df, num_bins, density=True)
        central_values = np.diff(bin_edges)*0.5 + bin_edges[:-1]

        results = {}
        xdata = df.iloc[:, 0]
        fit_x = np.linspace(0,max(xdata),200)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for modifier in range(NX):
            # Go through each [X]
            ydata = df.iloc[:, modifier+1]
            parameters, covariance = curve_fit(MMeq, xdata, ydata)
            # standard error https://education.molssi.org/python-data-analysis/03-data-fitting/index.htmlhttps://education.molssi.org/python-data-analysis/03-data-fitting/index.html
            SE = np.sqrt(np.diag(covariance))
            kcat = parameters[0]
            SE_kcat = SE[0]
            Km = parameters[1]
            SE_Km = SE[1]
            fit_y = MMeq(fit_x, kcat, Km)
            plot_fit(xdata, fit_x, ydata, fit_y, modifier, colors)
            # Parse fit results 
            results[modifier] = [df.columns[modifier+1], kcat, 
                                 SE_kcat, Km, SE_Km]
            plt.legend(title='[X]')
        col = df.columns[1:].values
        results = pd.DataFrame(results,
                               index = ['[X]','kcat', 'SE kcat', 'Km', 'SE Km'])
        results = results.transpose()
        return results, colors, fig_fit
    
    
    #Fit fingerprint equations to data
    def fun_kcat(x, p_k2, p_Kca, p_K1):
        """
        Kca = Kx*alpha
        K1 = beta/Kca
        """
        y = p_k2 * (1 + x*p_K1) / (1 + x/p_Kca)
        return y
    
    def fun_Km(x, p_Km, p_Kx, p_Kca):
        """
        Kca = Kx*alpha
        """
        y = p_Km * (1 + x/p_Kx) / (1 + x/p_Kca)
        return y
    
    @st.cache_data()
    def fit_fingerprints(df):
        """ 
        """
        xdata = df.iloc[:, 0].astype(float)
        # kcat vs. [X]
        ydata0 = df.iloc[:,1]
        #index of minimum [X]
        imin = np.argmin(xdata)
        # initial guess of k2, Kca and K1
        ig = np.asarray([1,ydata0[imin],np.mean(xdata)])
        parameters0, covariance0 = curve_fit(fun_kcat, xdata, ydata0, p0=ig, maxfev=10000)
        k2 = parameters0[0]
        Kca = parameters0[1]
        K1 = parameters0[2]
        
        # Kmvs. [X]
        ydata1 = df.iloc[:,3]
        # initial guess of Km, Kx and Ka
        ig = np.asarray([ydata1[imin],np.mean(xdata),np.mean(xdata)])
        parameters1, covariance1 = curve_fit(fun_Km, xdata, ydata1, ig, maxfev=10000)
        Km = parameters1[0]
        Kx = parameters1[1]
        Kca2 = parameters1[2]
        
        fig_fgpts, axs = plt.subplots(2)
        fig_fgpts.suptitle('Dependencies of the apparent parameters on the modifier concentration')
        plt.style.use("dark_background")
 
        x_fit = np.linspace(0,max(xdata),200)
        
        axs[0].scatter(xdata, ydata0)
        fit_y = fun_kcat(x_fit, k2, Kca, K1)
        axs[0].plot(x_fit, fit_y)
        axs[0].set(ylabel = 'kcat_app')
          
        axs[1].scatter(xdata, ydata1)
        fit_y = fun_Km(x_fit, Km, Kx, Kca2)
        axs[1].plot(x_fit, fit_y)
        plt.xlabel('[X]')
        plt.ylabel('Km_app')
        
        return k2, Km, Kx, fig_fgpts
    
    
    # Global fit
    @st.cache_data()
    def fun_eq_global(XDATA,alpha,beta,Kx):
        """
        # Global Fitting with fixed alpha and beta
        Based on https://stackoverflow.com/questions/28372597/python-curve-fit-with-multiple-independent-variables
        """
        S = XDATA[0:NS]
        X = XDATA[NS:NS+NX]
        # k2 = 50
        # Km = 85
        vE = np.zeros((len(S),len(X)))
        
        for i in range(NS):
           vE [:][i] = k2 * (1 + beta * X / alpha / Kx) * S[i] / (Km * (1 + X / Kx) + S[i] * ( 1+ X / Kx / alpha))
        
        vE = vE.flatten()
        return vE
    
    def flatten(nested_l):
        
        """
        Flatten the list and take care if there are tuples in the list.
        Arguments can be multiple (a, b, c...). This function is from:
        https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
        """
        if isinstance(nested_l, collections.Iterable):
            return [a for i in nested_l for a in flatten(i)]
        else:
            return [nested_l]
        

    def results_to_dataframe(df, results):
        """ 
        This function takes the results from the fitting and parses it to 
        produce variables that will be storred into PandasDataframe for
        the easy overview.
        """
        
        # Pasing dictionary with best fit results
        fit_dict_res = fit_data(df)
        
        df_dist = []
        df_params = []
        df_sse = []
        for distribution, result in fit_dict_res.items():
            sse = result[0]
            best_params = result[1:4] 

            flat_list = list(flatten(best_params))
            
            param_names = (distribution.shapes + ',loc,scale').split(',') if distribution.shapes else ['loc', 'scale']
            param_str = ', '.join([f'{k} = {round(v,2)}' for k,v 
                                   in zip(param_names, flat_list)])
            
            #title = f'{distribution.name}: {param_str}'
            
            df_dist.append(f'{distribution.name}')
            df_params.append(f'{param_str}')
            df_sse.append(round(sse, 4))

        fit_results = pd.DataFrame(
                {'Distribution': df_dist,
                 'Fit Parameters': df_params,
                 'SSE': df_sse}
                )

        return fit_results 
    
    def produce_output_for_code_download_parameters(df, results):
        """
        Parse the best fit function and parameters to generate python
        code for User. Works fine for all current forms of the 
        continuous functions (with various numbers of shape parameters).
        """
        
        # Need to start loop as I want to separate first item
        i = -1
        # Pasing dictionary with best fit results
        fit_dict_res = fit_data(df)
        for distribution, result in fit_dict_res.items():
            i += 1
            if i == 0:
                # Need to add to to check if there are shapes or not
                if distribution.shapes is not None:
                    shapes = distribution.shapes+str(',')
                else:
                    shapes = ""
                #print(shapes)
            else:
                pass
        df_results = results_to_dataframe(df, results)
        
        best_dist = df_results['Distribution'][0]
        
        fit_params_all = df_results['Fit Parameters']
          
        # Get scale
        best_scale = fit_params_all[0].split(", ")[-1]
        # Get loc 
        best_loc = fit_params_all[0].split(", ")[-2]
        # Get all arguments
        args = fit_params_all[0].split(",")[0:-2]
        # String manipulation to matches desired form for the code generation
        args = str([i for i in args]).strip(" [] ").strip("'").replace("'", '').replace(" ", '').replace(",",'\n')

        return shapes, best_dist, best_scale, best_loc, args, fit_params_all[0]
    
    def get_code():
        """ Prints out the python formatted code"""
        
        # st.code(f"{generate_fit_code}")    

    def py_file_downloader(py_file_text):
        """
        Strings <-> bytes conversions and creating a link which will
        download generated python script.
        """

        # Add a timestamp to the name of the saved file
        time_stamp = time.strftime("%Y%m%d_%H%M%S")

        # Base64 takes care of porting info to the data
        b64 = base64.b64encode(py_file_text.encode()).decode()
        
        # Saved file will have distribution name and the timestamp
        # code_file = f"{best_dist}_{time_stamp}.py"
        #st.markdown(f'** Download Python File **: \
        #            <a href="data:file/txt;base64,{b64}" \
        #                download="{code_file}">Click Here</a>', 
        #                unsafe_allow_html=True)
        st.download_button(
            label = 'Download .py file',
            data = f'{py_file_text}',
            # file_name = f'{code_file}',
            mime = 'application/octet-stream')

    def csv_downloader(data):
        """
        Strings <-> bytes conversions and creating a link which will
        download generated csv file with the DataFrame that contains fitting
        results.
        """
        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        
        csvfile = data.to_csv()
        
        b64 = base64.b64encode(csvfile.encode()).decode()
        
        new_filename = f"fit_results{time_stamp}.csv"
        
        #href = f'** Download DataFrame as .csv: ** \
        #    <a href="data:file/csv;base64,{b64}" \
        #    download="{new_filename}">Click Here</a>'
        #st.markdown(href, unsafe_allow_html=True)
        
        st.download_button(
            label='Download DataFrame as .csv',
            data=f'{csvfile}',
            file_name=f'fit_results_{time_stamp}.csv',
            key='download-csv')
        
    def selector(p_alpha,p_beta,p_Km,p_Kx):
        """
        Decision tree to select the GMM mechanism from the fitted parameters.
        """
        
        SF = 0.05  # Sensitivity factor
        
        # Evaluating the Specific/Catalytic/Mixed nature of the modifier (alpha)
        alpha_round = round(p_alpha * (10**2)) * (10**-2)
        if (alpha_round < 1+SF) and (alpha_round > 1-SF):  # alpha = 1
            flag_alpha = 'Balanced'
        elif alpha_round >= 1+SF:  # alpha > 1
            flag_alpha = 'Specific'
        elif alpha_round <= 1-SF:  # alpha < 1
            flag_alpha = 'Catalytic'
        
        # Evaluating alpha & beta to determine modifying mechanism
        beta_round = round(p_beta * (10**2)) * (10**-2)
        
        if beta_round <= 0.05: # beta = 0
            if (alpha_round < 1+SF) and (alpha_round > 1-SF):  # alpha = 1
                Mechanism = 'LMx(Sp=Ca)I'
            elif alpha_round >= 1+SF:  # alpha > 1
                if alpha_round > 20:  # alpha --> +Inf
                    Mechanism = 'LSpI'
                else:
                    Mechanism = 'LMx(Sp>Ca)I'
            elif (alpha_round <= 1-SF) and (alpha_round >= SF*2):  # alpha < 1
                Mechanism = 'LMx(Sp<Ca)I'
            # elif alpha_round > 20:  # alpha --> +Inf
            #     Mechanism = 'LSpI'
            elif alpha_round < SF*2 and p_Kx > p_Km:  # alpha --> 0, Kx > Km in the absence of modifier
                Mechanism = 'LCaI'   
            elif (beta_round < 1+SF) and (beta_round > 1-SF):
                if alpha_round >= 1+SF:  # alpha > 1
                    Mechanism = 'HSpI'
                elif alpha_round <= 1-SF:  # alpha < 1
                    Mechanism = 'HSpA'
            else: # Unsuccessful determination
                Mechanism ='Unable to successfully determine mechanism.'
        elif (beta_round <= 1-SF) and (beta_round > SF):     # beta < 1
            if alpha_round >= 1+SF:  # alpha > 1
                Mechanism = 'HMx(Sp>Ca)I'
            elif alpha_round <= 1-SF:  # alpha < 1
                if (abs(alpha_round - beta_round) < 0.3):  # alpha = beta
                    Mechanism = 'HCaI'
                elif alpha_round > beta_round:  # alpha > beta
                    Mechanism = 'HMx(Sp<Ca)I'
                elif beta_round > alpha_round:  # beta > alpha
                    Mechanism = 'HMxD(A/I)'
            elif (alpha_round < 1+SF) and (alpha_round > 1-SF):  # alpha = 1
                Mechanism = 'HMx(Sp=Ca)I'
        elif beta_round >= 1 + SF:    # beta > 1
            if alpha_round <= 1 - SF: # alpha < 1
                if (abs(alpha_round - beta_round) < 0.3): # alpha = beta
                    Mechanism = 'HCaA'
                # elif alpha_round > beta_round: # alpha > beta
                #     Mechanism = 'HMxD(I/A)'
                elif beta_round > alpha_round: # beta > alpha
                    Mechanism = 'HMx(Sp>Ca)A'
            elif alpha_round >= 1 + SF: # alpha > 1
                if (abs(alpha_round - beta_round) < 0.3): # alpha = beta
                    Mechanism = 'HCaA'
                elif alpha_round < beta_round: # alpha < beta
                    Mechanism = 'HMx(Sp<Ca)A'
                elif alpha_round > beta_round: # alpha > beta
                    Mechanism = 'HMxD(I/A)'
            elif (alpha_round < 1 + SF) and (alpha_round > 1 - SF): # alpha = 1
                Mechanism = 'HMx(Sp<Ca)A'
        else: # Unsuccessful determination
            Mechanism = 'Unable to successfully determine mechanism.'
        return Mechanism

    # Key to run all if this option is selected
    key_run = True if analysis_mode == 'Run All' else False
    # Display resulsts    
    if input:
        # st.write("Determine apparent kcat and Km values:")
        data_fit_chk = st.checkbox("Determine apparent values of kcat and Km", value=key_run)
        if data_fit_chk:
            with st.spinner("Fitting... Please wait a moment."):
                results, colors, fig = fit_data(df)
                st.pyplot(fig)
                st.info('Fitted parameters and values of standard\
                        error (SE)')
                st.dataframe(results)
                
                data_comp_chk = st.checkbox("Compute derived parameters", value=key_run)
                if data_comp_chk:
                    results = results.assign(inv_cat = 1 / results.kcat,
                                             kcat_Km = results.kcat / results.Km,
                                             Km_kcat = results.Km / results.kcat)
                    st.dataframe(results)
                    
                    k2, Km, Kx_e, fig_fgpts = fit_fingerprints(results)
                    st.pyplot(fig_fgpts)
                    
                    # st.info('First estimates of parameters k2 and Km')
                    # results_fgrprt_0 = pd.DataFrame([ [k2, Km ] ],
                    #                               index = ['Values'],
                    #                               columns = ['k2','Km'])
                    # st.dataframe(results_fgrprt_0)
                    
                    final_polish = st.checkbox("Final polish and GMM identification", value=key_run)
                    if final_polish:
                       
                        # Global Fitting with fixed alpha and beta
                        y = (df.iloc[:,1:].to_numpy())
                        y = y.flatten()
                        # y = np.array(y)
                        S_exp = df.iloc[:,0].astype(float) #.to_numpy()
                        X_exp = results.iloc[:, 0].astype(float)
                        NS = len(S_exp)
                        NX = len(X_exp)
                        XDATA = np.concatenate((S_exp, X_exp), axis=0)
                       
                        # initial guesses for alpha, beta, Kx
                        p0 = 1, 1, Kx_e
                        # y1 = fun_eq_global(XDATA,2.4,0.35,39.77)
                        # st.write(y1)
                        # XDATA=np.column_stack([S,X])
                        
                        popt, pcov = curve_fit(fun_eq_global,XDATA, y, p0, maxfev=10000)
                        alpha = popt[0]
                        beta= popt[1]
                        Kx = popt[2]
                        
                        st.info('Global Fit')     
                        # plot global fit
                        fig_fit2 = plt.figure()
                        xdata = S_exp
                        fit_x = np.linspace(0,max(xdata),200)
                        for modifier in range(NX):
                            # Go through each [X]
                            ydata = df.iloc[:, modifier+1]
                            Kca = Kx*alpha
                            kcat =  fun_kcat(X_exp[modifier], k2, Kca, beta/Kca)
                            Km = fun_Km(X_exp[modifier], Km, Kx, Kca)
                            fit_y = MMeq(fit_x, kcat, Km)
                       
                            plot_fit(xdata, fit_x, ydata, fit_y, modifier, colors)
                            plt.legend(title='[X]')
                        plt.xlabel('[S]')
                        plt.ylabel('[v] / [E]')
                        st.pyplot(fig)
                        results_fgrprt = pd.DataFrame([ [alpha, beta, k2, Km, Kx ]],
                                                      index = ['Values'],
                                                      columns = ['alpha','beta','k2','Km', 'Kx'])
                        st.dataframe( results_fgrprt)
                        
                        # selector(p_alpha, p_beta, p_Km, p_Kx)                      
                        Mechanism = selector(alpha, beta, Km, Kx )
                        if Mechanism != 'Unable to successfully determine mechanism.':
                            st.success('Success!')
                            # Goodness of  https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
                            y1 = fun_eq_global(XDATA,alpha, beta, Kx)
                            residuals = y - y1
                            ss_res = np.sum(residuals**2)
                            ss_tot = np.sum((y-np.mean(y))**2)
                            r_squared = 1 - (ss_res / ss_tot)  
                            st.write(':green[The kinetic mechanism of this modifier is:]  ', name_proper_dict[Mechanism])
                            st.write(':green[Acronym:]', Mechanism)
                            st.write(':green[Goodness of fit (r^4):]','%.5f' % r_squared)
                            st.write(':green[For more information click:]  ', name_url_dict[Mechanism])
                           
                        else:
                            st.warning(Mechanism)
                       

        
        
#                 generate_fit_code = f"""
# # -*- coding: utf-8 -*-
# # Generated using Distribution Analyser:
# # https://github.com/rdzudzar/DistributionAnalyser
# # {time.strftime("%Y%m%d_%H%M%S")}
# # ---

# import matplotlib.pyplot as plt #v3.2.2
# import numpy as np #v1.18.5
# from scipy.stats import {best_dist} #v1.6.1
# import math

# # Set random seed
# np.random.seed(1)

# # Function parameters
# {best_scale}
# {best_loc}
# {args}              
     
# # Generate evenly spaced numbers over a specified interval
# size = 400
# x = np.linspace({best_dist}.ppf(0.001, {shapes} loc=loc, scale=scale ), 
#                 {best_dist}.ppf(0.999, {shapes} loc=loc, scale=scale ),
#                 size)

# # Freeze the distribution
# rv = {best_dist}({shapes} loc=loc, scale=scale)

# # Generate random numbers
# r = {best_dist}.rvs({shapes} loc=loc, scale=scale, size=size)

# # Make a plot
# fig, ax = plt.subplots(1, 1)

# # Plot PDF, CDF and SF
# ax.plot(x, rv.pdf(x), linestyle='-', color='#3182bd', lw=3, label='PDF')
# ax.plot(x, rv.cdf(x), linestyle='-', color='k', lw=3, label='CDF')
# ax.plot(x, rv.sf(x), linestyle='-', color='#df65b0', lw=3, label='SF')

# ###### User Data #######


# ## You can plot your data on the results uncommenting following code lines
# ## If You are using Pandas:
# # import pandas as pd
# # import math
# #
# ## Import data
# # df = pd.read_csv('datafile_name.csv')
# # df = df['Column_name']


# ## Your data instead of 'df'; can be df['Column_Name']


# ## Plot histogram of your data
# #ax.hist(df, density=True, bins=20, 
# #        edgecolor='black', 
# #        fill = False, 
# #        linewidth=1, alpha=1, label='Sample distribution')


# ###### End of User inpu #######

# # Set legend
# ax.legend(bbox_to_anchor=(0,1.1,1,0.2), 
#              loc="center", 
#              borderaxespad=0, ncol=3)

# # Set Figure aestetics
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_title(f'{best_dist}: {fit_params_all}')

# ax.set_xlabel('X value')
# ax.set_ylabel('Density')
   
# plt.show()
#     """
#                 # Press the button to get the python code and 
#                 #   download hyperlink option
#                 if st.checkbox('Generate Python Code'):

#                     st.info("""
#                          **Python script** with best fit
#                          distribution & parameters.
#                         """)
#                     get_code()
#                     py_file_downloader(f"{generate_fit_code}")

                

    # return