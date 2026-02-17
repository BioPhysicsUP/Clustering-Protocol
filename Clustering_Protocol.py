# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:03:04 2025

@author: Michael Lovemore, University of Pretoria, Biophysics Research Group -- Clustering Protocl 
for Single Molecule Spectroscopy Data
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd   
from   tabulate import tabulate
from   pandasgui import show
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment


class SMS_clustering_protocol:
    """ 
    This function is a clustering algorithm used to cluster and sort clusters based on 
    SMS lifetime and intensity data. It can be extended, but for the current version, it assumes a
    pandas dataframe is used with the following columns, 'int' for intensity, 'particle' for 
    porticle labels which are labelled as "Particle 1", ... , 'av_tau' for lifetimes,
    'start' and 'end' for start and end times, and 'dwell' for dwell times. For the intensity trace
    it also assumes a csv used for the raw data and column names need to be adjusted accordingly.
    """
    
           
    def __init__(self, dataframe):     
        """
        Initialize dataframe to be used throughout the class.
        """
        self.dataframe = dataframe   

    def _draw_cluster_ellipse(self, mean, cov, ax, color, alpha=0.6, conf=0.95):
            if cov.shape == (2, 2):
                # Compute eigenvalues and eigenvectors
                vals, vecs = np.linalg.eigh(cov)  # Use eigh for symmetric matrices
                order = vals.argsort()[::-1]  # descending order
                vals = vals[order]
                vecs = vecs[:, order]
    
                # Angle of ellipse (in degrees)
                angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    
                # Width and height = 2 * sqrt(eigenvalues) scaled by chi2
                width, height = 2 * np.sqrt(vals * chi2.ppf(conf, df=2))
            else:
                angle = 0
                width, height = 2 * np.sqrt(cov * chi2.ppf(conf, df=2))
            
            ell = Ellipse(
                    xy=mean,
                    width=width,
                    height=height,
                    angle=angle,
                    edgecolor=color,
                    facecolor='none',
                    lw=2,
                    alpha=alpha
                    )
            ax.add_patch(ell)


    def contour_plot(self, fontsize=30, xminn=0, xlim=4, ylim=4.5, yminn=0, levels=20, 
                     numaxlabl=5, xdata = 'int', ydata = 'av_tau', xlbl = 'Intensity (kcounts/s)', ylbl = 'Lifetime (ns)'):
        """ 
        This function returns a contour plot of the lifetime-intensity distribution.
        """
        dataframe = self.dataframe
        int_data  = dataframe[xdata]# / 1000
        tau_data  = dataframe[ydata]
        
        
        
        plt.figure(dpi=1000, figsize=(10, 6))
        ax        = plt.gca()
        
        # Plot KDE
        sns.kdeplot(
            x=int_data,
            y=tau_data,
            cmap='turbo',
            bw_adjust=0.6,
            fill=True,
            alpha=0.75,
            common_grid=True,
            levels=levels,
            ax=ax
            )
        
        #plt.scatter(int_data, tau_data, s=100, alpha=0.7, color = 'red')

        contour_levels = np.linspace(0, 1, levels)
        vmin           = contour_levels.min()
        vmax           = contour_levels.max()

        # Colorbar based on colormap and normalization
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm   = mpl.cm.ScalarMappable(cmap='turbo', norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.ax.tick_params(labelsize=fontsize)

        # Format colorbar ticks
        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

        # Plot formatting
        ax.patch.set_facecolor('#36215e')
        ax.set_xlabel(xlbl, fontsize=fontsize)
        ax.set_ylabel(ylbl, fontsize=fontsize)
        ax.set_xlim(xminn, xlim)
        ax.set_ylim(yminn, ylim)
        ax.locator_params(axis='x', nbins=numaxlabl)
        ax.locator_params(axis='y', nbins=numaxlabl)
        ax.tick_params(labelsize=fontsize)
        plt.tight_layout()
        plt.show()

    
    def int_trace_plot(self, particle_id, time_res, fontsize = 30, numaxlabl = 6, xlim=25, ylim=3, 
                       xmin=0, ymin=0):
        """
        This function accepts particle_id and time_res as an input, and generates the resolved
        intensity trace, as well as the raw intensity trace
        """
        times, intensities = [], []
        particle_df        = self.dataframe[self.dataframe["particle"] == particle_id]
        
        for _, row in particle_df.iterrows():
            t = np.arange(row["start"], row["end"], time_res)
            i = np.full_like(t, row["int"], dtype=float)
            times.append(t)
            intensities.append(i)
        
        df     = pd.read_csv("C:/Users/Mikey/Downloads/LHCII Traces/" + particle_id 
                             + " trace (ROI).csv") 
        x      = df['Bin Time (s)'].iloc[1:]   # column B, from row 2 onwards
        y      = df['Bin Int (counts/100ms)'].iloc[1:]   # column C, from row 2 onwards
        plt.figure(dpi=1000, figsize=(20, 6)) #make x 18 for longer looking trace
        plt.step(x, y/100, color = 'grey')            
        t_vals, i_vals = np.concatenate(times), np.concatenate(intensities)
        i_vals = i_vals
        
        plt.step(t_vals, i_vals, label = f"Particle {particle_id}", linewidth = 2, color = 'green')
        plt.xlabel("Time (s)",  fontsize=fontsize)
        plt.ylabel("Intensity (kcounts/s)", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)  # X-axis tick labels
        plt.yticks(fontsize=fontsize)  # Y-axis tick labels
        plt.xlim(xmin, xlim)
        plt.ylim(ymin, ylim)
        plt.show()
    


    def find_nr_of_clusters(self, max_cluster=10, conf=0.95, fontsize=30, numaxlabl=10,
                            xlim=1, xmin=0, ymin=0, ylim=1, rel_drop_factor=0.5):
    
        X = self.dataframe[['int', 'av_tau']].values
        n_samples = X.shape[0]
    
        scores_dict = {'AIC': [], 'BIC': [], 'ICL': []}
        assignments_all, gmms_all = [], []
    
        # Fit GMMs
        for k in range(1, max_cluster + 1):
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=5, covariance_type='full')
            gmm.fit(X)
            clusters = gmm.predict(X)
    
            aic = gmm.aic(X)
            bic = gmm.bic(X)
            tau = gmm.predict_proba(X)
            icl = bic - np.sum(tau * np.log(tau + 1e-12))
    
            scores_dict['AIC'].append(aic)
            scores_dict['BIC'].append(bic)
            scores_dict['ICL'].append(icl)
    
            assignments_all.append(clusters)
            gmms_all.append(gmm)
    
        # Compute Mahalanobis-based tightness
        results = {'AIC': [], 'BIC': [], 'ICL': []}
        for idx, k in enumerate(range(1, max_cluster + 1)):
            clusters = assignments_all[idx]
            gmm = gmms_all[idx]
            means = gmm.means_
            covs = gmm.covariances_
    
            D2_all = np.zeros(n_samples)
            for i in range(k):
                idx_i = clusters == i
                if not np.any(idx_i):
                    continue
                Xk = X[idx_i]
                mu = means[i]
                cov = covs[i]
                inv_cov = np.linalg.inv(cov)
                diff = Xk - mu
                D2_all[idx_i] = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    
            chi2_theory = chi2.ppf(conf, df=2)
            tightness_list, frac_outside_list = [], []
            for i in range(k):
                D2k = D2_all[clusters == i]
                if len(D2k) == 0:
                    tightness_list.append(np.nan)
                    frac_outside_list.append(np.nan)
                    continue
                tightness_list.append(np.mean(D2k))
                frac_outside_list.append(np.mean(D2k > chi2_theory))
    
            avg_tightness = np.nanmean(tightness_list)
            frac_outside = np.nanmean(frac_outside_list)
    
            results['AIC'].append({'k': k, 'score': scores_dict['AIC'][idx],
                                   'avg_tightness': avg_tightness, 'frac_outside': frac_outside})
            results['BIC'].append({'k': k, 'score': scores_dict['BIC'][idx],
                                   'avg_tightness': avg_tightness, 'frac_outside': frac_outside})
            results['ICL'].append({'k': k, 'score': scores_dict['ICL'][idx],
                                   'avg_tightness': avg_tightness, 'frac_outside': frac_outside})
    
        # Candidate selection using normalized IC and statistical ΔIC threshold
        def candidate_clusters_stat(scores, rel_drop_factor=rel_drop_factor):
            scores = np.array(scores)
            # normalize
            scores_norm = scores / np.max(scores)
            ΔIC = np.diff(scores_norm)  # IC[k] - IC[k-1], negative is improvement
            std_drop = np.std(ΔIC)
            threshold = -rel_drop_factor * std_drop  # significant negative drops
            candidates = []
    
            # first cluster
            if ΔIC[0] < 0:
                candidates.append(2)
    
            # middle clusters
            for i in range(1, len(ΔIC)):
                # big negative improvement
                if ΔIC[i] < threshold:
                    candidates.append(i+2)
                # local minima: IC lower than neighbors
                if i < len(scores_norm)-1 and scores_norm[i] < scores_norm[i-1] and scores_norm[i] < scores_norm[i+1]:
                    candidates.append(i+1)
    
            return sorted(set(candidates))
    
        top_candidates = {}
        for metric in ['AIC','BIC','ICL']:
            scores = [r['score'] for r in results[metric]]
            candidate_k = candidate_clusters_stat(scores)
            # filter by fraction outside
            candidates = [r for r in results[metric] if r['k'] in candidate_k and r['frac_outside'] < 0.2]
            # rank by IC + tightness
            top_candidates[metric] = sorted(candidates, key=lambda r: (r['score'], r['avg_tightness']))
    
            print(f"\nCandidate clusters for {metric}:")
            for r in top_candidates[metric]:
                k = r['k']
                # compute backward delta for this cluster
                if k == 1:
                    delta_val = np.nan
                else:
                    delta_val = scores_dict[metric][k-2] - scores_dict[metric][k-1]
                print(f"k={k}, {metric}={r['score']:.3f}, Δ{metric}={delta_val:.3f}, "
                      f"Avg tightness={r['avg_tightness']:.3f}, Frac outside={r['frac_outside']:.4f}")

    
        # Plot IC curves with annotated candidates
        
        
        aic_scores = scores_dict['AIC']
        bic_scores = scores_dict['BIC']
        icl_scores = scores_dict['ICL']
        aic_norm = aic_scores / np.max(aic_scores)
        bic_norm = bic_scores / np.max(bic_scores)
        icl_norm = icl_scores / np.max(icl_scores)
        
        
        
        n_components_range = np.arange(1, max_cluster + 1)
        plt.figure(dpi=150, figsize=(10, 6))
        n_components_range = range(1, max_cluster + 1)
        plt.plot(n_components_range, aic_norm, 'ro--', markersize=8, label="AIC")
        plt.plot(n_components_range, bic_norm, 'go--', markersize=8, label="BIC")
        plt.plot(n_components_range, icl_norm, 'bo--', markersize=8, label="ICL")
        plt.xlabel('Number of clusters', fontsize=fontsize)
        plt.ylabel('Normalized IC', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.locator_params(axis='x', nbins=numaxlabl)
        plt.locator_params(axis='y', nbins=numaxlabl)
        plt.xlim(xmin, xlim)
        plt.ylim(ymin, ylim)
        plt.legend(fontsize=fontsize)
        plt.show()
    
        return top_candidates










    
        
    def clustering_the_data(self, cluster_nr, fontsize=30, xminn=0, xlim=4, ylim=4.5, yminn=0, 
                            levels=200, numaxlabl=5, conf = 0.95, xdata = 'int', ydata = 'av_tau', xlbl = 'Intensity (kcounts/s)', ylbl = 'Lifetime (ns)'):
        """
        This function takes "cluster_nr" as an input parameter (typically chosen from the BIC score)
        and clusters the data according using the designated number. This function also accepts
        "conf" as an input, which stipulates the confidence for which to plot probability mass
        ellipses about the centers of each cluster (black cross)
        """
        dataframe             = self.dataframe
        int_data              = dataframe[xdata]
        tau_data              = dataframe[ydata]
        dataframe['int_kcnt'] = int_data

        X        = dataframe[['int_kcnt', 'av_tau']]
        int_data = int_data 

        gmm = GaussianMixture(cluster_nr, random_state=42, n_init=5, covariance_type='full')
        gmm.fit(X)
        clusters = gmm.predict(X)
        dataframe['cluster'] = clusters
        
        X_np = X.values
        means = gmm.means_
        covs  = gmm.covariances_
        
        D2_all = np.zeros(len(X_np))

        for k in range(cluster_nr):
            idx = clusters == k
            if not np.any(idx):
                continue

            Xk = X_np[idx]
            mu = means[k]
            cov = covs[k]
            inv_cov = np.linalg.inv(cov)

            diff = Xk - mu
            D2 = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
            D2_all[idx] = D2
        
        chi2_theory = chi2.ppf(conf, df=2)
        
        cluster_tightness = []
        cluster_frac_outside = []

        for k in range(cluster_nr):
            D2k = D2_all[clusters == k]
            if len(D2k) == 0:
                cluster_tightness.append(np.nan)
                cluster_frac_outside.append(np.nan)
                continue

            cluster_tightness.append(np.mean(D2k))
            cluster_frac_outside.append(np.mean(D2k > chi2_theory))
            print('For conf =', conf, 'D2=', chi2_theory)
            print('For cluster k =', k+1)
            print('Cluster Tightness params = ', cluster_tightness[k])
            print('Frac outside cluster = ', cluster_frac_outside[k])
            
            
        
            
        print('Mean Tightness =', np.mean(np.array(cluster_tightness)))
        print('Mean Frac outside =', np.mean(np.array(cluster_frac_outside)))
            
        

        

        # Plot KDE with GMM cluster centers
        plt.figure(dpi=1000, figsize=(10, 6))
        ax = plt.gca()
        
        
        # KDE Plot
        sns.kdeplot(
            x=int_data,
            y=tau_data,
            cmap='turbo',
            bw_adjust=0.6,
            fill=True,
            alpha=0.75,
            common_grid=True,
            levels=levels,
            ax=ax
            )
        
        #Choose a colormap, e.g., 'tab10', 'viridis', 'plasma', etc.
        colormap = cm.get_cmap('Paired', cluster_nr)  
        colors = [colormap(i) for i in range(cluster_nr)]
        #plt.scatter(int_data, tau_data, s=100, alpha=0.7, color = 'red')
        # Plot cluster centers
        centers = gmm.means_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1, marker='X')

        # Draw ellipses for each cluster
        for i, (mean, cov) in enumerate(zip(means, covs)):
            color = colors[i % len(colors)]
            self._draw_cluster_ellipse(mean, cov, ax, color, alpha=1, conf = conf)

        # Fix colorbar (use full normalization range)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(cmap='turbo', norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
        cbar.ax.tick_params(labelsize=fontsize)

        ax.patch.set_facecolor('#36215e')
        plt.xlabel(xlbl, fontsize=fontsize)
        plt.ylabel(ylbl, fontsize=fontsize)
        plt.xlim(xminn, xlim)
        plt.ylim(yminn, ylim)
        plt.locator_params(axis='x', nbins=numaxlabl)
        plt.locator_params(axis='y', nbins=numaxlabl)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
        # Sort and label centers 
        int_centres = centers[:, 0] 
        tau_centers = centers[:, 1] 
        sorted_centers = sorted(zip(int_centres, tau_centers)) 
        int_cent_sort, tau_cent_sort = zip(*sorted_centers) 
        for i, (x, y) in enumerate(zip(int_cent_sort, tau_cent_sort)): 
            plt.text(x + 0.05, y + 0.05, f"{i+1}", color='black', fontsize=fontsize, 
                     fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        
        # plt.figure(dpi=1000, figsize=(10, 6))
        # plt.scatter(int_data, tau_data, s=30, alpha=0.7)
        # plt.xlabel("Spectral Peak Position (nm)")
        # plt.ylabel("FWHM (nm)")
        # plt.title("Simulated FWHM vs Spectral Peak")
        # for i, (mean, cov) in enumerate(zip(means, covs)):
        #     color = colors[i % len(colors)]
        #     self._draw_cluster_ellipse(mean, cov, ax, color, alpha=1, conf = conf)
        # for i, (x, y) in enumerate(zip(int_cent_sort, tau_cent_sort)): 
        #     plt.text(x + 0.05, y + 0.05, f"{i+1}", color='black', fontsize=fontsize, 
        #              fontweight='bold')
        # plt.show()
        

        self.centers  = centers
        self.clusters = clusters
        self.covs     = gmm.covariances_
        
        return (centers)
    
    def ground_truth_recovery(self, true_states, gmm_means, int_lbl = "mean_intensity", tau_lbl = "lifetime_mean" ):
        """ Both have array shape (n_states, 2) -> [int, tau] """
        
        true_centers = np.array([[v[int_lbl]/1000, v[tau_lbl]] 
                             for v in true_states.values()])

        # Compute distance matrix
        D = np.linalg.norm(true_centers[:, None, :] - gmm_means[None, :, :], axis=2)
    
        # Hungarian algorithm for optimal matching
        row_ind, col_ind = linear_sum_assignment(D)

        mean_int_error = np.abs(true_centers[row_ind, 0] - gmm_means[col_ind, 0])
        mean_tau_error = np.abs(true_centers[row_ind, 1] - gmm_means[col_ind, 1])
        rmse = np.sqrt(np.mean(D[row_ind, col_ind]**2))

        errors = {
            "mean_int_error": mean_int_error,
            "mean_tau_error": mean_tau_error,
            "rmse": rmse,
            "mapping": (row_ind, col_ind)
            }
        return errors
        
        
        
    def make_cluster_centre_df(self):
        """
        This function makes a dataframe from the cluster centres as it is a readily used 
        reference throughout the rest of the data analysis.
        """
        return pd.DataFrame(self.centers)
    
    def get_means_and_covs(self):
        """fff"""
        return self.centers, self.covs
    
    def sort_cluster_centres(self):
        """ 
        This function Takes the cluster-center coordinates and reorganizes them in order. This
        is neccesary as the order in which GMM cluster components are generated are typically
        not useable or relevant.
        """
        centers                       = self.centers
        int_centres                   = centers[:, 0]
        tau_centers                   = centers[:, 1]
        sorted_centers                = sorted(zip(int_centres, tau_centers))
        int_cent_sort, tau_cent_sort  = zip(*sorted_centers)
        int_cent_sort                 = list(int_cent_sort)
        tau_cent_sort                 = list(tau_cent_sort)
        self.sorted_int_centres       = int_cent_sort
        self.sorted_tau_centres       = tau_cent_sort
        return (int_cent_sort, tau_cent_sort)
    
    
    
    def get_ordered_clusters(self, int_centres):
        """ 
        This function allows input of "int_centres", typically acquired after running the 
        "sort_cluster_centres" function, and returns the cluster labels from GMM in order of 
        increasing intensites. The order is returned as a list (eg 0, 2, 1) which is to be used
        as an input to the "get_dwells_of_sorted_clusters" function.
        """
        dataframe            = self.dataframe 
        cluster_ordered      = []
        for i in range(len(int_centres)):
            mean_int         = dataframe[dataframe['cluster'] == i]['int'].mean()
            mean_tau         = dataframe[dataframe['cluster'] == i]['av_tau'].mean()
            print(f"cluster {i}: {mean_int}, {mean_tau}") 
            cluster_ordered.append((i, mean_int))
            
        cluster_ordered_sorted = sorted(cluster_ordered, key=lambda x: x[1])
        print(cluster_ordered_sorted)
        
        return(cluster_ordered_sorted)
    
    def get_dwells_of_sorted_clusters(self, dataframe, cluster_ordered_sorted):
        """
        This function takes "dataframe" and "cluster_ordered_sorted" as an input. To be accurate,
        this function inherits dataframe prof its parent function, but if a different dataframe is
        to be analyzed, then the "dataframe" input can be called. The output of the
        "get_ordered_clusters" function is typically used as the input for "cluster_ordered_sorted".
        This function returns the average and total dwell times for the different clusters, and the
        nr of data points in that cluster.
        """
        
        
        for index, (cluster_label, mean_int) in enumerate(cluster_ordered_sorted):
            print(f"S{index}_dwell_tot", dataframe[dataframe['cluster']  == 
                                                   cluster_label]['dwell'].sum())
            print(f"S{index}_dwell_mean", dataframe[dataframe['cluster'] == 
                                                    cluster_label]['dwell'].mean())
            print(f"Nr of Data Points in S{index}", len(dataframe[dataframe['cluster'] == 
                                                                  cluster_label]))
        
        
    def filtered_IQR_data_to_df(self, IQR_Coeff, centre_df ):
        """ 
        This function allows input of "IQR_Coeff" and "centre_df". It removes the outliers of the 
        post-clustered data via a modified IQR detection rule by eliminating data that lay 
        "IQR_Coeff" * IQR away from the median of "centre_df"
        """
        dataframe            = self.dataframe 
        distances            = cdist(dataframe[['int', 'av_tau']].values, centre_df.values)
        min_dist             = np.min(distances, axis=1)
        median_dist          = np.median(min_dist)
        q75, q25             = np.percentile(min_dist, [75, 25])
        iqr                  = q75 - q25
        threshold_distance   = median_dist + IQR_Coeff * iqr
        return (dataframe[min_dist <= threshold_distance])
    
    def rate_freq_dwelltime(self, full_df, lower_cluster, upper_cluster):    
        """ 
        This function takes "full_df" as an input for a desired dataframe, and allows input of
        two population labels, namely "lower_cluster" and "upper_cluster". It determines the
        switching rate, frequency and dwell times of the switches between the two input populations.
        """
        
        Q_to_U_swtch                 = 0
        U_to_Q_swtch                 = 0
        dwell_time_list              = full_df['dwell'].to_list()
                         
        for i in range(0, len(full_df)-1):
            if full_df['cluster'].iloc[i]==lower_cluster:
                if full_df['cluster'].iloc[i+1]==lower_cluster:
                    pass
                elif full_df['cluster'].iloc[i+1]==upper_cluster:
                    Q_to_U_swtch = Q_to_U_swtch + 1
            elif full_df['cluster'].iloc[i]==upper_cluster:
                if full_df['cluster'].iloc[i+1]==lower_cluster:
                    U_to_Q_swtch = U_to_Q_swtch + 1
                elif full_df['cluster'].iloc[i+1]==upper_cluster:
                    pass
                
        unquench_dwell_t = full_df[full_df['cluster'] == upper_cluster]['dwell'].sum()
        quench_dwell_t   = full_df[full_df['cluster'] == lower_cluster]['dwell'].sum()      
       
            
        k_UQ                         = U_to_Q_swtch/unquench_dwell_t
        
        k_QU                         = Q_to_U_swtch/quench_dwell_t
        
        switching_freq               = (Q_to_U_swtch + U_to_Q_swtch)/   \
                                       (quench_dwell_t + unquench_dwell_t)
                                       
                                       
        results_dict                 = {'kQU': [k_QU], 'kuq ': [k_UQ],
                                        'switching_freq': [switching_freq],
                                        'Q Dwell time': [quench_dwell_t],
                                        'U Dwell time': [unquench_dwell_t],
                                        'Nr of Q to U switches': [Q_to_U_swtch],
                                        'Nr of U to Q switches': [U_to_Q_swtch]
                                        }
        df = pd.DataFrame(results_dict)
        show(df)
        print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
    
    def weighted_int_and_tau(self, df):
        """ 
        This function allows input of "df", a pandas dataframe, and computes the weighted
        average intensity and lifetime, weighted by the dwell times.
        """
        weighted_avg_int    = (df['int'] * df['dwell']).sum() / df['dwell'].sum()
        weighted_avg_av_tau = (df['av_tau'] * df['dwell']).sum() / df['dwell'].sum()
        w                   = df['dwell']
        x                   = df['av_tau']
        x_w                 = (x * w).sum() / w.sum()
      
        int_var = ((df['dwell'] * (df['int'] - weighted_avg_int) ** 2).sum() / df['dwell'].sum())
        
        tau_var = ( (w * (x - x_w)**2).sum() ) / ( w.sum() - (w**2).sum()/w.sum() )
        weighted_std_int = int_var ** 0.5
        weighted_std_tau = tau_var ** 0.5

        print('weighted ave int =', weighted_avg_int, '±', weighted_std_int)
        print('weighted ave tau =', weighted_avg_av_tau, '±', weighted_std_tau)

        
    def determining_cluster_lifetimes(self, df, cluster_ordered_sorted):
        """ 
        This function allows input of the dataframe "df" and the sorted clusters in order attained
        from the previous functions, as an input parameter "cluster_ordered_sorted" and determines
        the lifetimes and errors for each cluster. 
        """
        for index, (cluster_label, mean_int) in enumerate(cluster_ordered_sorted):
            filter_df = df[df['cluster'] == cluster_label]['av_tau']
            print(f"Ave S{index}", filter_df.mean(), '+-', filter_df.std())
            
   