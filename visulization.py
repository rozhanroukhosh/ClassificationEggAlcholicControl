import numpy as np
import pandas as pd 
import random
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot


init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

seed = 23
random.seed = seed


def choose_random(maching_condition, df, random_id=random.randint(0,100)):
    """
        summary: Function merged data frame - one data frame for randomly selected subject from control group and 
        one data frame for randomly selected subject from alcoholic group
        params: 
            maching_condition: string -> shows the maching condition that will be used as s1 match , s2 match
            df: the data frame passed for processing
            random_id: the randome id of the subject we choose
           one data frame for randomly selected subject from alcoholic group
        return:
            a new Dataframe concated for alcholic choosen id and control id
    """
   
    # random choose the name_id of subject from alcoholic/control group
   
    control_name = df['name'][(df['subject identifier'] == 'c') & 
                                  (df['matching condition'] == maching_condition)].unique()[random_id]
    # get min trial numbers for each group
    control_trial_number = df['trial number'][(df['name'] == control_name) \
                                              & (df['matching condition'] == maching_condition)].min()
    # filter the EEG DF  
    control_df = df[(df['name'] == control_name) & (df['trial number'] == control_trial_number)]
    


   
    alcoholic_name = df['name'][(df['subject identifier'] == 'a') & 
                                    (df['matching condition'] == maching_condition)].unique()[random_id]
    alcoholic_trial_number = df['trial number'][(df['name'] == alcoholic_name)\
                                                & (df['matching condition'] == maching_condition)].min()
    
    alcoholic_df = df[(df['name'] == alcoholic_name) & (df['trial number'] == alcoholic_trial_number)]
    
    
    return pd.concat([alcoholic_df, control_df], axis=0)



def plot_sensors_correlation(df, title,threshold, list_of_pairs):
    """
    summary: Funtion plots the the correlation plots between sensor positions for each group
    params:
        df: dataframe of the data
        title: plot title
        threshold: the value for saying this two region of the brain are correlated
        list_of_pairs: list of pairs of brain region
    """

    a_corr= pd.pivot_table(df[df['subject identifier'] == 'a'], \
                                          values='sensor value', index='sample num', columns='sensor position').corr()

    c_corr = pd.pivot_table(df[df['subject identifier'] == 'c'], \
                                          values='sensor value', index='sample num', columns='sensor position').corr()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.set_title('Control group', fontsize=14)
    mask = np.zeros_like(c_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(c_corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    
    ax = fig.add_subplot(122)
    ax.set_title('Alcoholic group', fontsize=14)
    mask = np.zeros_like(a_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    #todo work on the color mapping
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(a_corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.01, cbar_kws={"shrink": .5})
    
    
    
    plt.suptitle('Correlation between Sensor Positions for ' \
                 + df['matching condition'].unique()[0] + ' stimulus', fontsize=16)
    plt.show()
    
    
def get_correlated_pairs(stimulus, threshold, group, df,list_of_pairs):
    """
    summary: Funtion returns the df which holds pairs of channel with high correlation for stimulus,
             group and threshold provided
    params:
        stimulus: string, the stimulus condition
        df: dataframe of the data
        group: string, the group title 
        threshold: the value for saying this two region of the brain are correlated
        list_of_pairs: list of pairs of brain region
    """
 
    corr_pairs = {}
    unique_trial_numbers = df['trial number'][(df['subject identifier'] == group) \
                                            & (df['matching condition'] == stimulus)].unique()
    # create dictionary where keys are the pairs and values are the amount of high correlation pair
    for i in range(len(list_of_pairs)):
        temp_corr_pair = dict(zip(list_of_pairs[i], [0]))
        corr_pairs.update(temp_corr_pair)

    for trial_number in unique_trial_numbers:    
        corr = pd.pivot_table(df[(df['subject identifier'] == group) \
                                        & (df['trial number'] == trial_number)], 
                                        values='sensor value', index='sample num',\
                                        columns='sensor position').corr()

        # by setting the j we are going just through values below the main diagonal
        j = 0 
        for column in corr.columns:
            j += 1
            for i in range(j, len(corr)):
                if ((corr[column][i] >= threshold) & (column != corr.index[i])): 
                        corr_pairs[column + '-' + corr.index[i]] += 1

    corr_count = pd.DataFrame(corr_pairs, index=['count']) \
                .T.reset_index(drop=False) \
                .rename(columns={'index': 'channel pair'})
    
    corr_count['group'] = group
    corr_count['stimulus'] = stimulus
    return(corr_count)



def compare_corr_pairs(stimulus, corr_pairs_df):
    """
    summary: Function creates bar chart with the ratio of correlated pairs for both groups
    
    params:
        stimulus: string, the stimulus condition
        corr_pairs_df: correted pairs list
    """
    control = corr_pairs_df[(corr_pairs_df['group'] == 'c') \
                                   & (corr_pairs_df['stimulus'] == stimulus)]
    alcoholic = corr_pairs_df[(corr_pairs_df['group'] == 'a') \
                                     & (corr_pairs_df['stimulus'] == stimulus)]
    top_control_pairs = control.sort_values('count', ascending=False)['channel pair'][:20]
    top_alcoholic_pairs = alcoholic.sort_values('count', ascending=False)['channel pair'][:20]
    
    
    merged_df = pd.DataFrame({'channel pair': \
                              np.unique(np.concatenate((top_control_pairs,top_alcoholic_pairs), axis=0))})
    merged_df = merged_df.merge(control[['channel pair', 'count', 'trials count']],
                                on='channel pair', how='left') \
                                .rename(columns={'count':'count control', 'trials count': 'trials count c'})
    merged_df = merged_df.merge(alcoholic[['channel pair', 'count', 'trials count']],
                                on='channel pair', how='left') \
                                .rename(columns={'count':'count alcoholic', 'trials count': 'trials count a'})

    alcholic_bar = go.Bar(x=merged_df['channel pair'],
                    y=(merged_df['count alcoholic']/merged_df['trials count a']),
                    text=merged_df['count alcoholic'],
                    name='Alcoholic Group',
                    marker=dict(color='rgb(10,200,45)'))

    control_bar = go.Bar(x=merged_df['channel pair'],
                    y=(merged_df['count control']/merged_df['trials count c']),
                    text=merged_df['count control'],
                    name='Control Group',
                    marker=dict(color='rgb(200,30,45)'))

    layout = go.Layout(title='Amount of Correlated Pairs for the whole Data Set (' + stimulus + ' stimulus)',
                       xaxis=dict(title='Channel Pairs'),
                       yaxis=dict(title='Ratio of Trail count by count of group'),
                       barmode='group')

    data = [alcholic_bar, control_bar]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
    
def sample_coor(df, list_of_pairs, S1obj, match, notMatch):
    """
    sumary : get plots for one random object

    Params:

    df : TYPE
        DESCRIPTION.
    list_of_pairs : list
        list_of_pairs: list of pairs of brain region
    S1obj : dataframe
        a randomly selected subject with the response to s1obj.
    match : dataframe
        a randomly selected subject with the response to match.
    notMatch : dataframe
        a randomly selected subject with the response to notMatch.

    Returns
    -------
    None.

    """
    
    plot_sensors_correlation(S1obj, 's1 obj stimuls', 0.98 , list_of_pairs)
    plot_sensors_correlation(match, 's1s2 match', 0.98, list_of_pairs)
    plot_sensors_correlation(notMatch, 's1s2 nomatch', 0.98, list_of_pairs)
    
    sns.violinplot(x="subject identifier",y="sensor value", hue="subject identifier", data=match,color='green') \
    .set(title='match sensor value'); 
    plt.show()
    sns.violinplot(x="subject identifier",y="sensor value", hue="subject identifier", data=notMatch,color='blue') \
    .set(title='not match sensor value'); 
    plt.show()
    sns.violinplot(x="subject identifier",y="sensor value", hue="subject identifier", data=S1obj,color='blue')\
    .set(title='s1 obj sensor value'); 
    plt.show()
    

def general_coor(corr_pairs_df):
    """
    summary: get the  correclation for  all of the  dataset 
    Params:
    corr_pairs_df : dataftame
        a dataframe all of the correlated pairs of channls.

    Returns
    -------
    None.

    """
    compare_corr_pairs('S1 obj', corr_pairs_df)
    compare_corr_pairs('S2 match', corr_pairs_df)
    compare_corr_pairs('S2 nomatch', corr_pairs_df)
    

def plot(EEG_data):
    """
    summary: in this function all the necessary list and dataframe are created for vasulization 
    Parameters
    ----------
    EEG_data : dataframe
        the whole dataset.

    Returns
    -------
    None.

    """
    df = EEG_data.toPandas()
    S1obj = choose_random('S1 obj', df, 1)
    match = choose_random('S2 match', df, 1)
    notMatch = choose_random('S2 nomatch', df, 1)
    # create the corr list of possible channel pairs
    sample_corr_df = pd.pivot_table(S1obj, values='sensor value', index='sample num', columns='sensor position').corr()
    # print("sample_c",sample_corr_df) 
    list_of_pairs = []
    j = 0
    for column in sample_corr_df.columns:
        j += 1
        for i in range(j, len(sample_corr_df)):
            #if not the same column
            if column != sample_corr_df.index[i]:
                temp_pair = [column + '-' + sample_corr_df.index[i]]
                list_of_pairs.append(temp_pair)
                
    # print("list_of_pairs",list_of_pairs)           
    corr_pairs_df = pd.DataFrame({})

    stimuli_list = ['S1 obj', 'S2 match', 'S2 nomatch']
    ## create df that holds information of total trial amount for each subject by stimulus
    size_df = df.groupby(['subject identifier', 'matching condition']) \
             [['trial number']].nunique().reset_index(drop=False).rename(columns={'trial number':'trials count'})

    for stimulus in stimuli_list:
        corr_pairs_df = pd.concat([corr_pairs_df, get_correlated_pairs(stimulus=stimulus, \
                                     threshold=.9, group='c', df=df,list_of_pairs=list_of_pairs)], axis = 0)
        
        corr_pairs_df = pd.concat([corr_pairs_df,get_correlated_pairs(stimulus=stimulus, \
                                    threshold=.9, group='a', df=df,list_of_pairs = list_of_pairs)], axis = 0)
        
    corr_pairs_df = corr_pairs_df.merge(size_df, left_on=['group', 'stimulus'], \
                                        right_on=['subject identifier', 'matching condition'], how='left')
    
    # print("list_of_pairs",corr_pairs_df) 
    sample_coor(df, list_of_pairs, S1obj, match, notMatch)
    general_coor(corr_pairs_df)
    
    
   
    


def plot_confusion(cf_matrix):
    """
    summary: Display the visualization of the Confusion Matrix.
    params: cf_matrix numpy array of the confusion Matrix
    
    """
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix for Test Dataset\n\n');
    ax.set_xlabel('\nPredicted Group ')
    ax.set_ylabel('Actual Group ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Alcholic','Control'])
    ax.yaxis.set_ticklabels(['Alcholic','Control'])

   
    plt.show()
    

# def get_correlated_pairs_sample(df, group, threshold, list_of_pairs):
#     """
#         summary: Function merged data frame - one data frame for randomly selected subject from control group and
#         one data frame for randomly selected subject from alcoholic group
#         params:
#             maching_condition: string -> shows the maching condition that will be used as s1 match , s2 match
#             df: the data frame passed for processing
#             random_id: the randome id of the subject we choose
#            one data frame for randomly selected subject from alcoholic group
#         return:
#             a new Dataframe concated for alcholic choosen id and control id
#     """

#     # create dictionary where keys are the pairs and values are the amount of high correlation pair
#     corr_pairs_dict = {}
#     for i in range(len(list_of_pairs)):
#         temp_corr_pair = dict(zip(list_of_pairs[i], [0]))
#         corr_pairs_dict.update(temp_corr_pair)

#     j = 0
#     for column in df.columns:
#         j += 1
#         for i in range(j, len(df)):
#             if ((df[column][i] >= threshold) & (column != df.index[i])):
#                 corr_pairs_dict[column + '-' + df.index[i]] += 1

#     corr_count = pd.DataFrame(corr_pairs_dict, index=['count'])\
#                             .T.reset_index(drop=False).rename(columns={'index': 'channel pair'})
#     print('Channel pairs that have correlation value >= ' + str(threshold) + ' (' + group + ' group):')
#     print(corr_count['channel pair'][corr_count['count'] > 0].tolist())


