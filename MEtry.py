import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from collections import defaultdict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

import cobra
from cobra.flux_analysis import sample,pfba,flux_variability_analysis
from cobra.flux_analysis.sampling import OptGPSampler, ACHRSampler
from cobra import Reaction, Metabolite, Model
from cobra.flux_analysis.loopless import add_loopless, loopless_solution

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler,StandardScaler,Imputer,FunctionTransformer
from sklearn.model_selection import train_test_split, validation_curve,GridSearchCV,learning_curve
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor

iML=cobra.io.load_matlab_model('iML1515.mat')
iML.reactions.EX_o2_e.bounds=(-18,1000.) #fix oxygen bounds
iML.reactions.EX_glc__D_e.bounds= (0.0,1000) # remove glucose feed
# making gene dict (common name to b name)
gene_dict_vals=pd.ExcelFile('gene_dict.xlsx').parse('gene_dict')
# gene_upper=[x.upper() for x in gene_dict_vals.old_name]
gene_dict=dict(zip(gene_dict_vals.old_name,gene_dict_vals.b_name))

#build GPR dictionary from iML1515 
iML_gene_list=[x.id for x in iML.genes]
GPR_dict=defaultdict(list)
for o_gene, n_gene in gene_dict.items():
    if n_gene in iML_gene_list:
        temp_gene=iML.genes.get_by_id(n_gene)
        rxn_list=[]
        for reaction in temp_gene.reactions:
            temp_dict={}
            temp_dict['mets']=[x.id for x in reaction.metabolites]
            temp_dict['mets_coefs']=[x for x in reaction.get_coefficients(reaction.metabolites)]
            temp_dict['lower_bound']=reaction.lower_bound
            temp_dict['upper_bound']=reaction.upper_bound
            temp_dict['id']=reaction.id
            temp_dict['name']=reaction.name
            temp_dict['subsystem']=reaction.subsystem
            temp_dict['gpr']=reaction.gene_reaction_rule
            rxn_list.append(temp_dict)
            
        GPR_dict[n_gene]=rxn_list
        
    else:
        GPR_dict[n_gene]=['NA']
        
# carbon source dictionary
carbon_sources_list=['None','EX_glc__D_e','EX_sucr_e','EX_fru_e','EX_sucr_e',
   'EX_pyr_e', 'EX_ac_e','EX_lac-D_e','EX_succ_e','EX_glyc_e','EX_gal_e',
   'EX_cellb_e','EX_xyl__D_e','EX_cellb_e','EX_sucr_e','EX_mnl_e','EX_glcn_e',
    'EX_arab__L_e','EX_ppa_e','EX_man_e','EX_xyl__D_e','EX_arab__L_e']

carbon_sources_key=[str(x) for x in range(22)]
cs_dict=dict(zip(carbon_sources_key,carbon_sources_list))

# product dictionary
product_key=['fatty acid', 'fatty alcohol', 'fatty acid (anteiso)', 'lipid',
       'Isobutanol', 'butanol', 'l-lactate', 'fatty alcohol (odd)',
       'succinate', 'xylitol', 'd-lactate', 'l-malate', 'butyrate',
       'hexanoate', '3-Methyl-1-butanol ', 'pyruvate', 'l-alanine', 'BDO',
       'ethanol', 'PHB', 'PHA', '1-propanol', 'methylketone', 'acetate','hydrogen',
       'l-phenylalanine','lycopene','13PDO','acrylic_acid','mesaconate','glycolate','ethylene_glycol']

product_type=[1,1,1,1,
              1,1,0,1,
              0,1,0,0,1,
              0,1,0,0,1,
              0,1,1,1,1,0,0,
              0,1,0,1,1,0,1]

product_list=['ACCOAC','ACCOAC','ACCOAC','ACCOAC',
              'OP4ENH','PDH','EX_lac__L_e','ACCOAC',
              'EX_succ_e','EX_xyl__D_e','EX_lac__D_e','EX_mal__L_e','PDH',
              'EX_hxa_e','PDH','EX_pyr_e','EX_ala__L_e','PDH',
              'EX_etoh_e','PDH','PDH','PDH','PDH','EX_ac_e','EX_h2_e',
              'EX_phe__L_e','ACCOAC','13PPDH2','PCNO','ICDHyr','EX_glyclt_e','2DDARAA']

prod_dict=dict(zip(product_key,product_list))
prod_type_dict=dict(zip(product_key,product_type))

#oxygen dictionary
oxy_dict={'1': -18.0, '2': 0.0,'3': -9.0,'nan': -18.0}

biorxn='BIOMASS_Ec_iML1515_core_75p37M'
glucose='EX_glc__D_e'

def add_genetic_info(temp_model,safe_model,genes_conv,GPR_dict,product,biorxn,max_grw50):
    success_genes=0
    for gene,mod_type in genes_conv.items():
        if  gene!='none':
            # check for special cases (TODO)
            # check if gene is present in GPR dictionary
            if not(GPR_dict[gene]==['NA']):
        #         print(mod_type)
                success_genes+=1
                new_rxns=[x['id'] for x in GPR_dict[gene]]
                for rxn_dict in GPR_dict[gene]:
                    if not(rxn_dict['id'] in temp_model.reactions):
                        new_reaction=Reaction(rxn_dict['id'])
                        new_reaction.name=rxn_dict['name']
                        new_reaction.subsystem=rxn_dict['subsystem']
                        new_reaction.lower_bound=rxn_dict['lower_bound']
                        new_reaction.upper_bound=rxn_dict['upper_bound']
                        new_reaction.add_metabolites(zip(rxn_dict['mets'],rxn_dict['mets_coefs']))
                        new_reaction.gene_reaction_rule=rxn_dict['gpr']
                        temp_model.add_reactions([new_reaction])
                #TODO: make allowance for irreversible and reversible reactions (not sure if this is an issue)
        #         temp_model.reactions.get_by_id(biorxn).bounds=(max_grw50,max_grw50)
                if mod_type=='1':
                    # we have an overexpression
                    rxn_vals={}
                    for rxn in new_rxns:
                        with temp_model as temp_temp_model:
                            temp_temp_model.reactions.get_by_id(biorxn).bounds=(max_grw50,max_grw50)
                            temp_temp_model.objective=rxn
                            #check for successful simulation
                            temp_temp_sol=temp_temp_model.optimize()
                            if temp_temp_sol.status!='optimal':
                                sim_grw_flag=0

                            obj_val=temp_temp_sol.objective_value* 0.1 # using 10% of actual value
                            rxn_vals[rxn]=obj_val
                    #update reaction bounds
                    for rxn in new_rxns:
                        temp_model.reactions.get_by_id(rxn).lower_bound=rxn_vals[rxn]
                else: 
                    #we have a knockout
                    for rxn in new_rxns:
                        temp_model.reactions.get_by_id(rxn).bounds=(0.0,0.0)
            
                
    # test if adding the modified genes messed things up
#     test_model=temp_model.copy()
    temp_model.objective=biorxn
    temp_sol1=temp_model.optimize()
    temp_model.objective=product
    temp_sol2=temp_model.optimize()
    
    if (temp_sol1.status!='optimal') or (temp_sol2.status!='optimal'): 
        sim_grw_flag=0
    else:
        sim_grw_flag=1
#         safe_model=temp_model.copy()
        safe_model=temp_model
                        
    return temp_model,safe_model,sim_grw_flag,success_genes


def plot_learning_curve(estimator, title, X, y, ylim=None,xlim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 20)):
    from sklearn.model_selection import learning_curve
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_validation_curve(estimator, title, X, y,param_range,param_name, ylim=None, cv=None):
  
    train_scores, test_scores = validation_curve(estimator, X, y,param_name=param_name,\
                                                param_range=param_range, cv=cv)  
    #plot validation curve
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    lw = 2

    plt.semilogx(param_range, train_scores_mean, label='Training score',
                color='darkorange', lw=lw)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color='darkorange', lw=lw)

    plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
                color='navy', lw=lw)

    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color='navy', lw=lw)

    plt.legend(loc='best')
    return plt

def add_env_info(temp_model,safe_model,product,biorxn,oxy,oxy_dict,cs_dict,cs1,cs2,cs3):
    
    temp_model.reactions.EX_o2_e.bounds=(oxy_dict[oxy],1000) # oxygen
    #carbon sources
    if (cs1=='0') or (cs1=='nan'):
        cs1='1'
    for cs in [cs1,cs2,cs3]:
        if cs!='0':
            temp_model.reactions.get_by_id(cs_dict[cs]).bounds=(-10,1000)
    # checking if everything is ok
#     test_model=temp_model.copy()
    temp_model.objective=biorxn
    temp_sol1=temp_model.optimize()
    temp_model.objective=product
    temp_sol2=temp_model.optimize()
    
    if (temp_sol1.status!='optimal') or (temp_sol2.status!='optimal'): 
        sim_grw_flag=0
    else:
        sim_grw_flag=1
#         safe_model=temp_model.copy()
        safe_model=temp_model
    
 
    return temp_model,safe_model,sim_grw_flag
        