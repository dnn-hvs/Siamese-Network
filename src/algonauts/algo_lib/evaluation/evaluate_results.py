#!/usr/bin/env python
# This script computes the score for the comparison of Model RDM with
# the fMRI data
# Input
#   -target_rdm.mat is the file that contains EVC and IT fMRI RDM matrices.
#   -submit_rdm.mat is the file that is the model RDM to be compared against the fMRI data submitted
# Output
#   -EVC_corr and IT_corr is the correlation of model RDMs to EVC RDM and IT RDM respectively
#   -pval is the corresponding p-value showing the significance of the correlation
# Note: Remember to use the appropriate noise ceiling correlation values for the dataset you are testing
# e.g. nc118_EVC_R2 for the 118-image training set.

import os
import sys
import h5py
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
from scipy import io
from algo_lib.utils import utils
# from utils import utils
import algo_lib.utils.constants as constants
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


class Evaluate():
    def __init__(self, config):
        self.config = config
        self.keys = {
            "fmri": [
                # "EVC R2",
                "EVC % Noise Ceiling",
                #  "EVC significance",
                #  "IT R2",
                "IT % Noise Ceiling",
                #  "IT significance",
                #  "fMRI Avg R2",
                "fMRI Avg % Noise Ceiling"],
            "meg": [
                # "Early R2",
                "Early % Noise Ceiling",
                # "Early significance",
                # "Late R2",
                "Late % Noise Ceiling",
                # "Late significance",
                # "MEG Avg R2",
                "MEG Avg % Noise Ceiling"]
        }

    def sq(self, x):
        return squareform(x, force='tovector', checks=False)

    # defines the spearman correlation

    def spearman(self, model_rdm, rdms):
        model_rdm_sq = self.sq(model_rdm)
        return [stats.spearmanr(self.sq(rdm), model_rdm_sq)[0] for rdm in rdms]

    # computes spearman correlation (R) and R^2, and ttest for p-value.

    def fmri_rdm(self, model_rdm, fmri_rdms):
        corr = self.spearman(model_rdm, fmri_rdms)
        corr_squared = np.square(corr)
        return np.mean(corr_squared), stats.ttest_1samp(corr_squared, 0)[1]

    def meg_rdm(self, model_rdm, meg_rdms):
        corr = np.mean([self.spearman(model_rdm, rdms)
                        for rdms in meg_rdms], 1)
        corr_squared = np.square(corr)
        return np.mean(corr_squared), stats.ttest_1samp(corr_squared, 0)[1]

    def evaluate(self, submission, targets, target_names):
        if self.config["task"] == "fmri":
            out = {name: self.fmri_rdm(submission[name], targets[name])
                   for name in target_names}
        else:
            out = {name: self.meg_rdm(submission[name], targets[name])
                   for name in target_names}

        out['score'] = np.mean([x[0] for x in out.values()])
        return out

    # function that evaluates the RDM comparison.

    def write_excel(self, df, net_name):
        all_path = os.path.join(
            self.config["exp_id"], "results/all") if self.config["exp_id"] is not None else "results/all"
        best_path = os.path.join(
            self.config["exp_id"], "results/best") if self.config["exp_id"] is not None else "results/best"
        utils.makedirs(all_path)
        df.to_excel(os.path.join(all_path, net_name+".xlsx"))
        best_df = pd.DataFrame()
        # print(self.config["task"], df.columns, sep=" : ")
        for col in df.columns:
            row = df.loc[df[col].idxmax()]
            temp = pd.DataFrame(
                [row], index=["Best "+col+": "+net_name])
            temp["Network Name"] = net_name
            temp["Layer Name"] = row.name
            best_df = best_df.append(temp)
        utils.makedirs(best_path)
        best_df.to_excel(os.path.join(best_path, net_name+"_best.xlsx"))
        return best_df

    def write_final_results(self, df):
        path = os.path.join(
            self.config["exp_id"], "results/final") if self.config["exp_id"] is not None else "results/final"
        utils.makedirs(path)
        writer = pd.ExcelWriter(os.path.join(path,
                                             "Main_Results_"+self.config["task"]+".xlsx"))
        for index_substr in self.keys[self.config["task"]]:
            best = df.filter(like=index_substr, axis=0)
            best.to_excel(writer, sheet_name=index_substr)
        writer.save()
        writer.close()

    def test_submission(self, rdms):
        image_set_details = constants.EVALUATE_DICT[self.config["task"]
                                                    ][self.config["image_set"]]
        target_file = image_set_details['target_file']
        target_names = constants.EVALUATE_DICT[self.config["task"]
                                               ]["target_names"]
        nc_EVC_R2 = image_set_details['nc_EVC_R2'] if self.config["task"] == "fmri" else image_set_details['nc_early_R2']
        nc_IT_R2 = image_set_details['nc_IT_R2'] if self.config["task"] == "fmri" else image_set_details['nc_late_R2']
        nc_avg_R2 = image_set_details['nc_avg_R2']
        target = utils.load(target_file)
        df = pd.DataFrame()

        out = self.evaluate(rdms, target, target_names=target_names)
        evc_percentNC = ((out[target_names[0]][0])/nc_EVC_R2) *\
            100.  # evc percent of noise ceiling
        it_percentNC = ((out[target_names[1]][0])/nc_IT_R2) * \
            100.  # it percent of noise ceiling
        score_percentNC = ((out['score'])/nc_avg_R2) * \
            100.  # avg (score) percent of noise ceiling
        if self.config["task"] == "fmri":
            results = {
                "EVC %": evc_percentNC,
                "IT %": it_percentNC,
                "fMRI Avg %": score_percentNC
            }
        else:
            results = {
                "Early %": evc_percentNC,
                "Late %": it_percentNC,
                "MEG Avg %": score_percentNC
            }

        return results

    def run(self, rdms):
        results = {}
        for task in constants.EVALUATE_DICT.keys():
            self.config["task"] = task
            res = self.test_submission(rdms[task])
            for key, val in res.items():
                results[key] = val
        return results
