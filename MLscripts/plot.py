import matplotlib.pyplot as plt
import numpy as np

feature_names = ['DER_mass_MMC',
 'DER_mass_transverse_met_lep',
 'DER_mass_vis',
 'DER_pt_h',
 'DER_deltaeta_jet_jet',
 'DER_mass_jet_jet',
 'DER_prodeta_jet_jet',
 'DER_deltar_tau_lep',
 'DER_pt_tot',
 'DER_sum_pt',
 'DER_pt_ratio_lep_tau',
 'DER_met_phi_centrality',
 'DER_lep_eta_centrality',
 'PRI_tau_pt',
 'PRI_tau_eta',
 'PRI_tau_phi',
 'PRI_lep_pt',
 'PRI_lep_eta',
 'PRI_lep_phi',
 'PRI_met',
 'PRI_met_phi',
 'PRI_met_sumet',
 'PRI_jet_num',
 'PRI_jet_leading_pt',
 'PRI_jet_leading_eta',
 'PRI_jet_leading_phi',
 'PRI_jet_subleading_pt',
 'PRI_jet_subleading_eta',
 'PRI_jet_subleading_phi',
 'PRI_jet_all_pt']


# label: 1 -> positive, -1 -> negative, other -> all
def plot_features(y, tx, label=2):
    positive_percentage = y[y==1].shape[0] / y.shape[0]
    negative_percentage = y[y==-1].shape[0] / y.shape[0]
    print("Positive y: " + str(positive_percentage*100) + "%")
    print("Negative y: " + str(negative_percentage*100) + "%")
    
    if label == 1:
        color = 'blue'
    elif label == -1:
        color = 'red'
    else:
        color = 'green'
        
    n_cols = 2
    n_rows = tx.shape[1] // n_cols
    fig, axes = plt.subplots(n_rows, n_cols)
    
    plt.subplots_adjust(hspace=0.9)
    
    fig.set_size_inches(15, 30)
    
    for x in range(tx.shape[1]):
        feature_col = tx[:, x]
        
        current_ax = axes[x//n_cols][x % n_cols]
        
        if label != 1 and label != -1:
            labels = np.where(~np.isnan(feature_col))
        else:    
            labels = np.where((~np.isnan(feature_col)) & (y==label))
            
        current_ax.scatter(labels, feature_col[labels], color=color, s=4)
        current_ax.set_title(feature_names[x] + "(" + str(x) + ")")
    
# label: 1 -> positive, -1 -> negative, other -> all
def plot_feature_histograms(y, tx, label=2):
    positive_percentage = y[y==1].shape[0] / y.shape[0]
    negative_percentage = y[y==-1].shape[0] / y.shape[0]
    print("Positive y (blue): " + str(positive_percentage*100) + "%")
    print("Negative y (red): " + str(negative_percentage*100) + "%")
    
    if label == 1:
        color = 'blue'
    elif label == -1:
        color = 'red'
    else:
        color = 'green'
        
    n_cols = 2
    n_rows = tx.shape[1] // n_cols
    fig, axes = plt.subplots(n_rows, n_cols)
    
    plt.subplots_adjust(hspace=0.9)
    
    fig.set_size_inches(15, 30)
    
    for x in range(tx.shape[1]):
        
        feature_col = tx[:, x]
        current_ax = axes[x//n_cols][x % n_cols]
        
        if label != 1 and label != -1:
            labels = np.where((~np.isnan(feature_col)) & (y==1))
            current_ax.hist(x=feature_col[labels], histtype='step', bins=100, color='blue', alpha=0.7, rwidth=0.85)
#             current_ax.hist(x=feature_col[labels], bins='auto', color='blue', alpha=0.7, rwidth=0.85)
            labels = np.where((~np.isnan(feature_col)) & (y==-1))
            color = 'red'
        else:
            labels = np.where((~np.isnan(feature_col)) & (y==label))

        current_ax.hist(x=feature_col[labels], histtype='step', bins=100, color=color, alpha=0.7, rwidth=0.85)
        current_ax.set_title(feature_names[x] + " (" + str(x) + ")")