import streamlit as st
import numpy as np
import sklearn
from download_button_file import download_button
import importlib
import walk_likelihood as wl
import VillageNet as VN
from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt

def KP_Survival(date,event_yes):
    N=len(date)
    date[np.isnan(date)]=0
    event_yes[np.isnan(event_yes)]=0
    inds=np.argsort(date)
    tms=[0]
    S=[1]
    s=1
    N_pop=len(date)
    for i in inds:
        if event_yes[i]>0:
            tms.append(date[i])
            S.append(s)
            s=s*(N_pop-1)/N_pop
            tms.append(date[i])
            S.append(s)
        N_pop=N_pop-1    
    
    return [tms,S]    


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


            datacols = st.columns((1, 1, 1))
            transpose=datacols[0].checkbox("Transpose Data", False)
            head=datacols[1].checkbox("Contains Headers", False)
            ids=datacols[2].checkbox("Contains Indices", False)
            
            

            #if head:
            #    df=pd.read_csv(uploaded_file)
            #else:
            #    df=pd.read_csv(uploaded_file,header=None)
            df=pd.read_csv(uploaded_file,header=None)

            if transpose:
                df=df.T
            if head:
                df = df.rename(columns=df.iloc[0]).drop(df.index[0])
            if ids:
                df=df.set_index(df.columns.tolist()[0])

            #head=st.checkbox("Contains headers", False)
            #if head:
            #    df=pd.read_csv(uploaded_file)
            #else:
            #    df=pd.read_csv(uploaded_file,header=None)
    
            st.write('### Data Uploaded')
    
            st.write(df)
            data=np.array(df)
            file_name=uploaded_file.name
            data=data.astype(float)
            st.write(data.dtype)
            
            

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
    KM_file = st.file_uploader("Kapplan Meier Analysis: Upload a 2 column file in CSV format with column 1 as time & column 2 as event", type=".csv")
    if KM_file:
        datacols2 = st.columns((1, 1, 1))
        transpose2=datacols2[0].checkbox("Transpose KM Data", False)
        head2=datacols2[1].checkbox("Contains KM Headers", False)
        ids2=datacols2[2].checkbox("Contains KM Indices", False)
            
        df2=pd.read_csv(KM_file,header=None)

        if transpose2:
            df2=df2.T
        if head2:
            df2 = df2.rename(columns=df2.iloc[0]).drop(df2.index[0])
        if ids2:
            df2=df2.set_index(df2.columns.tolist()[0])
        st.write('### Kapplan Meier Data Uploaded')
    
        st.write(df2)
        KMdata=np.array(df2)
        KMdata=KMdata.astype(float)
    if uploaded_file or Sample_data:
        st.markdown("## Hyper Parameters")

        cols = st.columns((1, 1))
        villages = cols[0].number_input('Number of villages',min_value=2, max_value=data.shape[0],step=1,value=np.minimum(200,X.shape[0]))
        neighbors = cols[1].number_input('Number of nearest neighbors',min_value=1, max_value=data.shape[0],step=1,value=np.minimum(20,X.shape[0]))
        num_comms=st.checkbox("Set number of clusters", False)
        if num_comms:
            cols2 = st.columns(1)
            comms = cols2[0].number_input('Number of clusters',min_value=2, max_value=data.shape[0],step=1,value=np.minimum(3,X.shape[0]))
        else:
            comms=None
        #new_method=st.checkbox('Use new method',False)
        with st.form(key="my_form"):

            run=st.form_submit_button(label="Cluster")
if run:
    model=VN.VillageNet(villages=villages,neighbors=neighbors,normalize=normalize)
    model.fit(X,comms=comms)
    U=np.zeros((X.shape[0],max(model.comm_id)+1))
    U[range(X.shape[0]),model.comm_id]=1
    for i in range(max(model.comm_id)+1):
        arr=np.array(range(X.shape[0]))[U[:,i]==1]
        strng=str(arr[0])
        for j in arr[1:]:
            strng=strng+', '+str(j)
        with st.expander("Cluster "+str(i)):            
            #st.write(str(list(np.array(range(X.shape[0]))[U[:,i]==1]))[1:-1])
            st.write(strng)


    disjoint_str=''
    for i in range(X.shape[0]-1):
        disjoint_str+=str(model.comm_id[i])+'\n'
    disjoint_str+=str(model.comm_id[X.shape[0]-1])
    download_button( disjoint_str,'disjoint_clusters_'+file_name,'Final Clusters')
    
    
    if KM_file:
        run2=st.form_submit_button(label="Plot Kaplan Meier Curves")
        if run2:
            Ss=[]
            tms=[]
            fig, ax = plt.subplots()
            for i in range(max(model.comm_id)+1):
                [a,b]=KP_Survival(KMdata[model.comm_id==i,0],KMdata[model.comm_id==i,1])
                print(len(a))
                Ss.append(b)
                tms.append(a)
                ax.plot(tms[i],Ss[i],label="Community "+str(i))
                ax.legend()
                
            st.pyplot(fig)

