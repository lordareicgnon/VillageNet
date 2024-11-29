import streamlit as st
import numpy as np
import sklearn
from download_button_file import download_button
import importlib
import walk_likelihood as wl
import VillageNet as VN
from sklearn.datasets import load_digits
import pandas as pd


run=0
st.write("""
# VillageNet Clustering Ⓒ \n
2023 Aditya Ballal, Gregory DePaul, Esha Datta & Leighton T. Izu
""")

#more_information = st.checkbox("More Information", False)

with st.expander("More Information"):
    st.write("""
    We present ”VillageNet Clustering,” a novel unsupervised clustering algorithm
    designed for effectively clustering complex manifold data.
    """)

#with st.sidebar:
#    with st.expander("More information on parameters"):
#        st.write("This is more information on parameters")

#ranonce=0
#runmapperplus = st.checkbox("Run MapperPlus", False)
runmapperplus=1
#with st.expander("Run MapperPlus"):
if runmapperplus:
    st.markdown("## Upload data for Clustering")
    see_results=0
    uploaded_file='False'
    with st.expander("ℹ️ More information"):
        st.write("Upload data in CSV format.")
    #with st.sidebar:
    Sample_data = st.checkbox(
        "Use Sample Data", False, help="Pen Digits Dataset")


    if not Sample_data:
        uploaded_file = st.file_uploader("Upload CSV", type=".csv")
        #st.write(uploaded_file)
        if uploaded_file:
            
            head=st.checkbox("Contains headers", False)
            if head:
                df=pd.read_csv(uploaded_file)
            else:
                df=pd.read_csv(uploaded_file,header=None)
    
            st.write('### Data Uploaded')
    
            st.write(df)
            data=np.array(df)
            file_name=uploaded_file.name

    else:
        #from sklearn.datasets import load_wine
        file_name='PenDigits.csv'
        #data = load_wine()['data']
        data = load_digits()['data']
    if uploaded_file or Sample_data:
        normalize = st.checkbox(
            "Normalize Data", False, help="Normalize Data using standard method")
        X=data
    #submit=False
    #with st.form("parameters"):
    if uploaded_file or Sample_data:
        st.markdown("## Hyper Parameters")

        cols = st.columns((1, 1))
        villages = cols[0].number_input('Number of villages',min_value=2, max_value=data.shape[0],step=1,value=np.minimum(200,X.shape[0]))
        neighbors = cols[1].number_input('Number of nearest neighbors',min_value=1, max_value=data.shape[0],step=1,value=np.minimum(20,X.shape[0]))
        #new_method=st.checkbox('Use new method',False)
        with st.form(key="my_form"):

            run=st.form_submit_button(label="Cluster")
if run:
    model=VN.VillageNet(villages=villages,neighbors=neighbors,normalize=normalize)
    model.fit(X)
    U=np.zeros((X.shape[0],max(model.comm_id)+1))
    U[range(X.shape[0]),model.comm_id]=1
    for i in range(max(model.comm_id)+1):

        with st.expander("Cluster "+str(i)):            
            st.write(str(list(np.array(range(X.shape[0]))[U[:,i]==1]))[1:-1])


    disjoint_str=''
    for i in range(X.shape[0]-1):
        disjoint_str+=str(model.comm_id[i])+'\n'
    disjoint_str+=str(model.comm_id[X.shape[0]-1])
    download_button( disjoint_str,'disjoint_clusters_'+file_name,'Final Clusters')
