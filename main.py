# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:37:35 2023

@author: pmartins 
based on https://github.com/rdzudzar/DistributionAnalyser/blob/main/main.py
Thank you Robert!
"""

# Awesome Streamlit
import streamlit as st

# Add pages -- see those files for deatils within
from page_explore import page_explore
from page_find import page_find
from page_introduction import page_introduction

# Use random seed
import numpy as np
np.random.seed(1)


# Set the default elements on the sidebar
st.set_page_config(page_title='GMM Finder')

logo, name = st.sidebar.columns(2)
# with logo:
#     image = 'https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/logo_da.png?token=AIAWV2ZRCFKYM42DVFTD3OLAN3CQK'
#     st.image(image, use_column_width=True)
with name:
    st.markdown("<h1 style='text-align: left; color: grey;'> \
                GMM Finder </h1>", unsafe_allow_html=True)

st.sidebar.write(" ")


def main():
    """
    Register pages to Explore and Fit:
        page_introduction - contains page with images and brief explanations
        page_find - contains various functions that allows user to upload
                    data as a .csv file, fit parameters and find the GMM.
        page_explore - contains links to other resources
    """

    pages = {
        "Introduction": page_introduction,
        "Find Mechanism": page_find,
        "Other Resources": page_explore,
    }

    st.sidebar.title("Main options")

    # Radio buttons to select desired option
    page = st.sidebar.radio("Select:", tuple(pages.keys()))
                                
    # Display the selected page with the session state
    pages[page]()

    # Write About
    st.sidebar.header("About")
    st.sidebar.warning(
            """
            The GMM Finder app is running in a beta version.
            For more information please contact:
            pmartins@ibmc.up.pt
            """
    )


if __name__ == "__main__":
    main()