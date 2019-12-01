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
from utils import utils
import utils.constants as constants
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


class Evaluate():
    def __init__(self, config):
        self.config = config
        self.keys = {
            "fmri": ["EVC R2",
                     "EVC % Noise Ceiling",
                     "EVC significance",
                     "IT R2",
                     "IT % Noise Ceiling",
                     "IT significance",
                     "fMRI Avg R2",
                     "fMRI Avg % Noise Ceiling"],
            "meg": ["Early R2",
                    "Early % Noise Ceiling",
                    "Early significance",
                    "Late R2",
                    "Late % Noise Ceiling",
                    "Late significance",
                    "MEG Avg R2",
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
        if self.config.task == "fmri":
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
            self.config.exp_id, "results/all") if self.config.exp_id is not None else "results/all"
        best_path = os.path.join(
            self.config.exp_id, "results/best") if self.config.exp_id is not None else "results/best"
        utils.makedirs(all_path)
        df.to_excel(os.path.join(all_path, net_name+".xlsx"))
        best_df = pd.DataFrame()
        # print(self.config.task, df.columns, sep=" : ")
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
            self.config.exp_id, "results/final") if self.config.exp_id is not None else "results/final"
        utils.makedirs(path)
        writer = pd.ExcelWriter(os.path.join(path,
                                             "Main_Results_"+self.config.task+".xlsx"))
        for index_substr in self.keys[self.config.task]:
            best = df.filter(like=index_substr, axis=0)
            best.to_excel(writer, sheet_name=index_substr)
        writer.save()
        writer.close()

    def test_submission(self, submit_file_dir, results_file_name):
        image_set_details = constants.EVALUATE_DICT[self.config.task][self.config.image_set]
        target_file = image_set_details['target_file']
        target_names = constants.EVALUATE_DICT[self.config.task]["target_names"]
        nc_EVC_R2 = image_set_details['nc_EVC_R2'] if self.config.task == "fmri" else image_set_details['nc_early_R2']
        nc_IT_R2 = image_set_details['nc_IT_R2'] if self.config.task == "fmri" else image_set_details['nc_late_R2']
        nc_avg_R2 = image_set_details['nc_avg_R2']

        print("Target File", target_file)
        target = utils.load(target_file)
        submit_file_name = constants.SUBMIT_FMRI_FILE if self.config.task == "fmri" else constants.SUBMIT_MEG_FILE

        layer = submit_file_dir.split("/")[-1]
        df = pd.DataFrame()
        for subdir, dirs, files in os.walk(submit_file_dir):
            if len(dirs) == 0 and len(files) != 0:
                file = os.path.join(subdir, submit_file_name)
                submit = utils.load(file)
                out = self.evaluate(submit, target, target_names=target_names)
                evc_percentNC = ((out[target_names[0]][0])/nc_EVC_R2) * \
                    100.  # evc percent of noise ceiling
                it_percentNC = ((out[target_names[1]][0])/nc_IT_R2) * \
                    100.  # it percent of noise ceiling
                score_percentNC = ((out['score'])/nc_avg_R2) * \
                    100.  # avg (score) percent of noise ceiling
                if self.config.task == "fmri":
                    results = {
                        "EVC R2": [out[target_names[0]][0]],
                        "EVC % Noise Ceiling": [evc_percentNC],
                        "EVC significance": [out[target_names[0]][1]],
                        "IT R2": [out[target_names[1]][0]],
                        "IT % Noise Ceiling": [it_percentNC],
                        "IT significance": [out[target_names[1]][1]],
                        "fMRI Avg R2": [out['score']],
                        "fMRI Avg % Noise Ceiling": [score_percentNC]
                    }
                else:
                    results = {
                        "Early R2": [out[target_names[0]][0]],
                        "Early % Noise Ceiling": [evc_percentNC],
                        "Early significance": [out[target_names[0]][1]],
                        "Late R2": [out[target_names[1]][0]],
                        "Late % Noise Ceiling": [it_percentNC],
                        "Late significance": [out[target_names[1]][1]],
                        "MEG Avg R2": [out['score']],
                        "MEG Avg % Noise Ceiling": [score_percentNC]
                    }

                df = pd.DataFrame(results, index=[layer])
        return df

    def execute(self, rdms_path):
        df, best_df, temp = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        net = None
        for subdir, dirs, files in os.walk(rdms_path):
            if len(dirs) == 0 and len(files) != 0:
                if net != subdir.split('/')[-2] and not df.empty:
                    best_df = best_df.append(temp)
                    df, temp = pd.DataFrame(), pd.DataFrame()

                net = subdir.split('/')[-2]
                df = df.append(self.test_submission(subdir, net))
                temp = self.write_excel(df, net+self.config.task +
                                        self.config.image_set)
                print(constants.LINE_SEPERATOR)
        print(net, self.config.task, self.config.image_set, len(temp), sep=" : ")
        if net:
            best_df = best_df.append(temp)
        return best_df

    def _run(self):
        if self.config.image_set == "all":
            for image_set in self.config.image_sets:
                self.execute(os.path.join(self.config.rdms_dir,
                                          image_set+'images_rdms', self.config.distance))
        else:
            self.execute(os.path.join(self.config.rdms_dir,
                                      self.config.image_set+'images_rdms', self.config.distance))

    def run(self):
        if self.config.fullblown or self.config.evaluate_results:
            for task in constants.EVALUATE_DICT.keys():

                main_df = pd.DataFrame()
                self.config.task = task
                for image_set in self.config.image_sets:
                    self.config.image_set = image_set
                    path = os.path.join(self.config.rdms_dir, self.config.distance,
                                        image_set+'images_rdms')
                    main_df = main_df.append(self.execute(path))
                self.write_final_results(main_df)
            return
        self._run()
