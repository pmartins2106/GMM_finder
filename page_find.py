# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:41:04 2023

@author: pmartins
"""

# Package imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cmasher as cmr
import numpy as np
# import seaborn as sns #check if needed
#from scipy import stats
import scipy.stats
# curve-fit() function imported from scipy
from scipy.optimize import curve_fit
import math

from bokeh.plotting import figure
from bokeh.models import Legend
#from bokeh.io import curdoc

# Helper function imports
# These are pre-computed so that they don't slow down the App
# from helper_functions import distr_selectbox_names,creating_dictionaries

import time
import base64
import collections


def page_find():
    """
    The find page in this app is made with Streamlit for fitting a GMM
    a GMM mechanism to the User imported data.
    """
    # name_docstring_dict, name_eq_dict, name_proper_dict, \
    #     all_dist_params_dict, name_url_dict = creating_dictionaries()
    
    st.sidebar.info("""
                Import enyme kinetics data and discover the mechanism of enzyme 
                inhibiton/activation.
                """)
                
    # Add a bit empy space before showing About
    st.sidebar.text("")
    
    # Input enzyme concentration
    Enzyme = st.sidebar.number_input('Enzyme concentration [E]', format="%2.2e", value = 1e0,min_value = 1e-20)
    
    st.sidebar.markdown("**Type of Analysis:**")
    analysis_mode = st.sidebar.radio('', ('Run All', 'Step-by-Step'))   

    st.sidebar.text("")
    st.sidebar.text("")

    st.markdown("<h1 style='text-align: center;'> Find the General Modifier Mechanism </h1>", 
                unsafe_allow_html=True)

    
    # Using cache as we perform only once 'loading' of the data
    @st.cache_data
    def load_csv():
        """ Get the loaded .csv into Pandas dataframe. """
        
        df_load = pd.read_csv(input, sep=',' , engine='python',
                             nrows=25, skiprows=1, encoding='utf-8')
        return df_load
   
    # Streamlit - upload a file
    input = st.file_uploader('')
    
 
    if 'run_example' not in st.session_state:
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
            st.markdown(""" [Download spreadsheet](https://github.com/rdzudzar/DistributionAnalyser/blob/main/sample_data/sample_da.csv)""")            
            st.markdown("""**fill the template; save as comma Separated (.csv) file; upload the .csv file*""")
            # if run button is pressed
        if st.session_state.run_example:
            input = "datasets//GMM_Finder_example.csv"
            # Pass to a function above so we can use st.cache
            df = load_csv()
            # Replace inf/-inf with NaN and remove NaN if present
            df = df.replace([np.inf, -np.inf], np.nan).dropna() #[data_col]
            #convert to [v]/[E]
            cols = df.columns[1:]
            df[cols] = df[cols] / Enzyme
               
            st.info('Uploaded Data')
            st.dataframe(df)
            # csv_downloader(results_to_dataframe(df, results))
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
                
               
    def plot(df, data_stat):
        """ 
        Histogram of the input data. Contains also information about the 
        Figure style, depending on the active Mode.
        """
        
        if analysis_mode == 'Light Mode':
            hist_edge_color = 'black'
            hist_color= 'white'
            quant_color = 'black'
            median_color = 'black'
            pdf_color = '#08519c'
            cdf_color = 'black'
            plt.style.use('classic')
            plt.rcParams['figure.facecolor'] = 'white'


        if analysis_mode == 'Dark Mode':
            hist_edge_color = 'black'
            hist_color= 'white'
            median_color = 'magenta'
            quant_color = 'white'
            pdf_color = '#fec44f'
            cdf_color = 'white'
            plt.style.use('dark_background')
            plt.rcParams['figure.facecolor'] = 'black'

        fig, ax = plt.subplots(1,1)
        
        # Plot hist
        ax.hist(df, bins=round(math.sqrt(len(df))), 
                density=True, color=hist_color, 
                ec=hist_edge_color, alpha=0.3)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.tick_params(top=False, bottom=True, left=True, right=False,
                labelleft=True, labelbottom=True)

        # ax.set_xlabel(f'{data_col}')
        ax.set_ylabel('Density')
        
        # If user selects data_stat
        if data_stat:
            # Hist contains tuple: n bins, (n+1) bin boundaries
            hist = np.histogram(df, bins=round(math.sqrt(len(df))))
            #Generates a distribution given by a histogram.
            hist_dist = scipy.stats.rv_histogram(hist)
            x_plot = np.linspace(min(df), max(df), len(df))

    
            q = [0.05, 0.25, 0.50, 0.75, 0.95]
            n = ['5th','25th','50th','75th','95th']
            quantiles = df.quantile(q)
            q_max = hist_dist.cdf(quantiles)

            
            for i, qu in enumerate(quantiles):
                ax.plot(qu, q_max[i], alpha=0.5, color=quant_color,
                        markersize=10, marker='D')
                ax.text(qu, q_max[i]+(q_max[i]/10), f'{n[i]}', ha='center',
                        alpha=0.5)
            ax.scatter([], [], alpha=0.5, color=quant_color, marker='D', 
                       label='Percentiles')
            # The pdf is defined as a stepwise function from the provided histogram.
            # The cdf is a linear interpolation of the pdf.
            ax.plot(x_plot, hist_dist.pdf(x_plot), linewidth=2,
                    color=pdf_color, label='PDF')
            ax.plot(x_plot, hist_dist.cdf(x_plot), linewidth=2,
                    color=cdf_color, label='CDF')
            
            
            ax.vlines(np.mean(df), ymin=0, ymax=hist_dist.cdf(np.mean(df)),
                      color='red', linestyle='--', linewidth=2,
                      label=f'Mean {round(np.mean(df),2)}')
            ax.vlines(np.median(df), ymin=0, ymax=hist_dist.cdf(np.median(df)),
                      color=median_color, linestyle='--', linewidth=2,
                      label=f'Median {round(np.median(df),2)}')            
       
            
            leg = plt.legend(loc=0)
            leg.get_frame().set_edgecolor("#525252")

        return fig
    
    def bokeh_set_plot_properties(analysis_mode, n):
        """
        Constructs a list of properties that will be assigned to a Bokeh
        figure depending whether it is in the Light or Dark Mode.

        Parameters
        ----------
        analysis_mode : string; plot 'Dark Mode' or 'Light Mode'

        Returns
        -------
        p : Bokeh Figure
        colors_cmr : Colors from the colormap to be assigned to lines

        """
                
        p = figure(height=450, width=700)
        
        p.add_layout(Legend(), 'right')
        p.legend.title = '15 Best Fits and their SSE'
        p.legend.background_fill_alpha = 1
        p.legend.label_text_font_size = '11pt'
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        # p.xaxis.axis_label = f'{data_col}'
        p.yaxis.axis_label = 'Density' 


        if analysis_mode == 'Dark Mode':
            text_color = 'white'
            back_color = 'black'
            legend_color = 'yellow'
            
            # It will get n colors from cmasher rainforest map
            # if n>15, it will take 15; otherwise n will be the 
            # lengthe of the chosed distributions (n defined line 685)
            colors_cmr = cmr.take_cmap_colors('cmr.rainforest_r', 
                                      n, cmap_range=(0.1, 0.7), 
                                     return_fmt='hex')     
        
        if analysis_mode == 'Light Mode':
            text_color = 'black'
            back_color = 'white'
            legend_color = 'blue'   
        
            colors_cmr = cmr.take_cmap_colors('cmr.rainforest', 
                                      n, cmap_range=(0.2, 0.9), 
                                     return_fmt='hex')
        
        p.legend.title_text_color = text_color
        p.yaxis.major_label_text_color = text_color
        p.xaxis.axis_label_text_color = text_color
        p.xaxis.major_label_text_color = text_color
        p.yaxis.axis_label_text_color = text_color
        p.xaxis.major_tick_line_color = text_color
        p.yaxis.major_tick_line_color = text_color
        p.xaxis.minor_tick_line_color = text_color
        p.yaxis.minor_tick_line_color = text_color
        p.xaxis.axis_line_color = text_color
        p.yaxis.axis_line_color = text_color
        
        p.border_fill_color = back_color
        p.background_fill_color = back_color
        p.legend.background_fill_color = back_color
        p.legend.label_text_color = legend_color
        p.title.text_color = legend_color
        p.outline_line_color = back_color
        
        return p, colors_cmr
       
      
    def bokeh_pdf_plot_results(df, results, n):
        """
        Process results and plot them on the Bokeh Figure. User can interact
        with the legend (clicking on the items will enhance lines on Figure)

        Parameters
        ----------
        df : input data
        results : nested list (contains tuples) with the data from the 
                fitting (contains [sse, arg, loc, scale])
        n : integer; First n best fit PDFs to show on the Figure.
        analysis_mode : string; 'Dark Mode' or 'Light Mode' (connected with radio
                                                         button)

        Returns
        -------
        p : Returns Bokeh interactive figure (data histogram+best fit PDFs)

        """
                 
        # Pasing dictionary with best fit results
        fit_dict_res = fit_data(df)
        hist, edges = np.histogram(df, density=True, 
                                   bins=round(math.sqrt(len(df))))
        
    
        # Obtain Figure mode from the function:  bokeh_set_plot_properties
        p, colors_cmr = bokeh_set_plot_properties(analysis_mode, n)
        
        # Bokeh histogram
        p.quad(top=hist, bottom=0, left=edges[:-1], 
               right=edges[1:], line_color="black",
               line_width = 0.3,
               fill_color='white', fill_alpha = 0.3)
        
        # Plot each fitted distribution
        i = -1
        for distribution, result in fit_dict_res.items():
            i += 1
            
            sse = round(result[0],2)
            arg = result[1]
            loc = result[2]
            scale = result[3]

            best_params = result[1:4] 
            flat_list = list(flatten(best_params)) 
            param_names = (distribution.shapes + ', loc, scale').split(', ') if distribution.shapes else ['loc', 'scale']
            param_str = ', '.join([f'{k} = {round(v,2)}' for k,v 
                                   in zip(param_names, flat_list)])

            # Generate evenly spaced numbers over a specified interval
            # Make pdf/cdf with the parameters of fitted functions
            x_plot = np.linspace(min(df), max(df), 400)
            y_plot = distribution.pdf(x_plot, loc=loc, scale=scale, *arg)

            # The best fit distribution will be with i=0
            if i == 0:
                # Bokeh line plot with interactive legend
                line = p.line(x_plot, y_plot, line_width=5,
                       line_color = colors_cmr[0],
                       legend_label=str(distribution.name) + ": " + str(sse)
                       )
                line.visible = True
                p.legend.click_policy = "hide"
                p.title.text = f'Best fit {distribution.name}: {param_str}'
                
                
                if distribution.name in name_eq_dict.keys():
                        
                    scipy_link = f'[{name_url_dict[distribution.name][1]}]({name_url_dict[distribution.name][0]})'    
                    
                    #st.markdown(f"""You can read more about best fit distribution:                        
                    #    [**{name_url_dict[distribution.name][1]}**]
                    #        ({name_url_dict[distribution.name][0]})
                    #    """)

                    st.markdown(f'You can read more about best fit distribution: {scipy_link}')

            # Next 15 best fits; 15 is arbitrary taken.
            elif (i>0) and (i < 15):
                lines = p.line(x_plot, y_plot, line_width=2.5,
                                line_dash="10 2",
                       line_color = colors_cmr[i],
                       legend_label =str(distribution.name) + ": " + str(sse)
                        )
                lines.visible = False
                p.legend.click_policy = "hide"

            else:
                pass
                   
        return p  


    def bokeh_cdf_plot_results(df, results, n):
        """
        Process results and plot them on the Bokeh Figure. User can interact
        with the legend (clicking on the items will enhance lines on Figure)

        Parameters
        ----------
        df : input data
        results : nested list (contains tuples) with the data from the 
                fitting (contains [sse, arg, loc, scale])
        n : integer; First n best fit CDFs to show on the Figure.
        analysis_mode : string; 'Dark Mode' or 'Light Mode' (connected with radio
                                                         button)

        Returns
        -------
        p : Returns Bokeh interactive figure (data hostogram+best fit CDFs)

        """
        
        # Hist contains tuple: n bins, (n+1) bin boundaries
        hist_data = np.histogram(df, bins=round(math.sqrt(len(df))))
        #Generates a distribution given by a histogram.
        hist_dist_data = scipy.stats.rv_histogram(hist_data)
        x_plot_data = np.linspace(min(df), max(df), 400)
          
    
        # Pasing dictionary with best fit results
        fit_dict_res = fit_data(df)
        
        hist, edges = np.histogram(df, density=True, 
                                   bins=round(math.sqrt(len(df))))
        
    
        # Obtain Figure mode from the function:  bokeh_set_plot_properties
        p, colors_cmr = bokeh_set_plot_properties(analysis_mode, n)
        
        # Bokeh histogram
        p.quad(top=hist, bottom=0, left=edges[:-1], 
               right=edges[1:], line_color="black",
               line_width = 0.3,
               fill_color='white', fill_alpha = 0.3)
        
        p.line(x_plot_data, hist_dist_data.cdf(x_plot_data), 
                          line_color='red', legend_label='CDF sample data',
                          line_width=3)  
        p.legend.click_policy = "hide"

        # Plot each fitted distribution
        i = -1
        for distribution, result in fit_dict_res.items():
            i += 1
            
            sse = round(result[0],2)
            arg = result[1]
            loc = result[2]
            scale = result[3]

            best_params = result[1:4] 
            flat_list = list(flatten(best_params)) 
            param_names = (distribution.shapes + ', loc, scale').split(', ') if distribution.shapes else ['loc', 'scale']
            param_str = ', '.join([f'{k} = {round(v,2)}' for k,v 
                                   in zip(param_names, flat_list)])

            # Generate evenly spaced numbers over a specified interval
            # Make pdf/cdf with the parameters of fitted functions
            x_plot = np.linspace(min(df), max(df), len(df))

            y_plot = distribution.cdf(x_plot, loc=loc, scale=scale, *arg)
                
                
            # The best fit distribution will be with i=0
            if i == 0:
                # Bokeh line plot with interactive legend
                line = p.line(x_plot, y_plot, line_width=5,
                       line_color = colors_cmr[0],
                       legend_label=str(distribution.name) + ": " + str(sse)
                       )
                line.visible = True
                p.legend.click_policy = "hide"
                p.title.text = f'Best fit {distribution.name}: {param_str}'

            # Next 15 best fits; 15 is arbitrary taken.
            elif (i>0) and (i < 15):
                lines = p.line(x_plot, y_plot, line_width=2.5,
                                line_dash="10 2",
                       line_color = colors_cmr[i],
                       legend_label =str(distribution.name) + ": " + str(sse)
                        )
                lines.visible = False
                p.legend.click_policy = "hide"

            else:
                pass
                   
        return p

    def MMeq(x, kcat, Km):
        y = kcat*x/(Km+x)
        return y
    
    @st.cache_data()
    def fit_data(df):
        """ 
        Modified from: https://stackoverflow.com/questions/6620471/fitting\
            -empirical-distribution-to-theoretical-ones-with-scipy-python 
        
        This function is performed with @cache - storing results in the local
        cache; read more: https://docs.streamlit.io/en/stable/caching.html
        """
        
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
        fig_fit = plt.figure()
        xdata = df.iloc[:, 0]
        x_fit = np.linspace(0,max(xdata),50)
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
            
            plt.style.use("dark_background")
            plt.scatter(xdata, ydata, color=colors[modifier], label=df.columns[modifier+1])
            fit_y = MMeq(x_fit, kcat, Km)
            plt.plot(x_fit, fit_y, color=colors[modifier])
            plt.xlabel('[S]')
            plt.ylabel('[v] / [E]')
            
            # Parse fit results 
            results[modifier] = [df.columns[modifier+1], kcat, 
                                 SE_kcat, Km, SE_Km]
            plt.legend(title='[X]')
        col = df.columns[1:].values
        results = pd.DataFrame(results,
                               index = ['[X]','kcat', 'SE kcat', 'Km', 'SE Km'])
        results = results.transpose()
        return results, fig_fit
    
    def fun_kcat(x, b, k2, Kca):
        y = k2 * (1 + b*x/Kca) / (1 + x/Kca)
        return y
    
    def fun_Km(x, a, Km, Kx):
        y = Km * (1 + x/Kx) / (1 + x/a/Kx)
        return y
    
    @st.cache_data()
    def fit_fingerprints(df):
        """ 
        Descriprion here
        This function is performed with @cache - storing results in the local
        cache; read more: https://docs.streamlit.io/en/stable/caching.html
        """
        xdata = df.iloc[:, 0].astype(float)
        # kcat vs. [X]
        ydata0 = df.iloc[:,1]
        parameters0, covariance0 = curve_fit(fun_kcat, xdata, ydata0)
        beta = parameters0[0]
        
        # Kmvs. [X]
        ydata1 = df.iloc[:,3]
        parameters1, covariance1 = curve_fit(fun_Km, xdata, ydata1)
        alpha = parameters1[0]
        Km = parameters1[1]
        Kx = parameters1[2]
        
        fig_fgpts, axs = plt.subplots(2)
        fig_fgpts.suptitle('Dependencies of the apparent parameters on the modifier concentration')
        plt.style.use("dark_background")
 
        x_fit = np.linspace(0,max(xdata),50)
        
        axs[0].scatter(xdata, ydata0)
 
        fit_y = fun_kcat(x_fit, beta, parameters0[1], parameters0[2])
        axs[0].plot(x_fit, fit_y)
        axs[0].set(ylabel = 'kcat_app')
          
        axs[1].scatter(xdata, ydata1)
        fit_y = fun_Km(x_fit, alpha, Km, parameters1[2])
        axs[1].plot(x_fit, fit_y)
        plt.xlabel('[X]')
        plt.ylabel('Km_app')
        
        
        results_fgrprt = pd.DataFrame([ [alpha, beta, Km, Kx ] ],
                                      index = ['Values'],
                                      columns = ['alpha','beta', 'Km', 'Kx'])        
        
        return results_fgrprt, fig_fgpts


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

    # Distribution names
    dis = distr_selectbox_names()

    # Checks steps by steps to ensure the flow of the data input,
    # fitting and the display of the results.

    key_run = True if analysis_mode == 'Run All' else False
        
    if input:
        # st.write("Determine apparent kcat and Km values:")
        data_fit_chk = st.checkbox("Determine apparent values of kcat and Km", value=key_run)
        if data_fit_chk:
            with st.spinner("Fitting... Please wait a moment."):
                results, fig = fit_data(df)
                
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
                    
                    results_fgrprt, fig_fgpts = fit_fingerprints(results)
                    
                    st.pyplot(fig_fgpts)
                    
                    st.info('First estimates of parameters alpha, beta, Km and Kx')
                    st.dataframe(results_fgrprt)
                    
                    final_polish = st.checkbox("Final polish and find GMM", value=key_run)
                    if final_polish:
                        st.info('Final estimates')
                        st.dataframe(results_fgrprt)
                        
                        Mechanism = selector(results_fgrprt.iat[0,0],results_fgrprt.iat[0,1], 
                                             results_fgrprt.iat[0,2], results_fgrprt.iat[0,3])
                        if Mechanism != 'Unable to successfully determine mechanism.':
                            st.success('Success!')
                            st.write(':blue[The kinetic mechanism of this modifier is:]  ', Mechanism)
                        else:
                            st.warning(Mechanism)
                       

        
        # # Add an option to have a 'Select All' distribution
        # dis_with_all =[]
        # dis_with_all = dis[:]
        # dis_with_all.append('All_distributions')

        # chosen_distr = st.multiselect('Choose distributions to fit', 
        #                               dis_with_all)
        # # Remove problematic distributions
        # if 'All_distributions' in chosen_distr:
        #     dis.remove('levy_stable')
        #     dis.remove('kstwo')
        #     chosen_distr = dis
          
        # # Give warnings if User selects problematic distributions
        # if chosen_distr:
        #     if 'kstwo' in chosen_distr:
        #         st.warning("User, be aware that **kstwo**\
        #                    distribution has some issues and will not compute.")
            
        #     if 'levy_stable' in chosen_distr:
        #         st.warning("User, be aware that **levy_stable**\
        #                    distribution will take several minutes to compute.")
            
        #     if chosen_distr == dis:
        #         st.warning(" You have selected **All distributions**, due to \
        #                    the slow computation of the *levy_stable* \
        #                     (and errors with *kstwo*), \
        #                     these distributions are removed\
        #                     from the list of 'All_distributions' ")
            
        #     st.write("Do you want to fit the selected distribution(s) \
        #              to your data?")
            
        #     # Checking length of selected distributions, for a number colors
        #     # that will be taken from colormap. As plot gets messy with more
        #     # than 15, I limit to that; if smaller, use that number
        #     if len(chosen_distr) > 15:
        #         n = 15
        #     else:
        #         n = len(chosen_distr)
                
                               
            # After fitting, checkbox apears and when clicked: user get 
            # options which results they want to see, as they are several
            # fit_confirmation =  st.checkbox("Yes, please.", value=False)
            # if fit_confirmation:
            #     st.write('Results are ready, select what you wish to see:')   

            #     if st.checkbox('Interactive Figures'):
            #         st.info('Interactive Figure: click on the legend to \
            #             enhance selected fit.')
            #         # p1 =  bokeh_pdf_plot_results(df, results, n) #p2
            #         # st.bokeh_chart(p1)
            #         # st.info('Interactive Figure: Comparing CDFs')
            #         # p2 =  bokeh_cdf_plot_results(df, results, n)
            #         # st.bokeh_chart(p2)
                
            #     if st.checkbox('Table'):
            #         st.info('DataFrame: all fitted distributions\
            #             and their SSE (sum of squared estimate of errors).')
            #         st.dataframe(results_to_dataframe(df, results))
            #         csv_downloader(results_to_dataframe(df, results))
                    

            #     shapes, best_dist, best_scale, best_loc, args, fit_params_all \
            #         = produce_output_for_code_download_parameters(df, results)

                # Fitting outputs are parsed to the f string below
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