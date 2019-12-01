LINE_SEPERATOR = '\n'+'=' * 20+'\n'
SUBMIT_FMRI_FILE = 'submit_fmri.mat'
SUBMIT_MEG_FILE = 'submit_meg.mat'

FORWARD_SLASH = '/'
UNDER_SCORE = "_"
# Text file constants
TXT_EXTENSION = '.txt'
APPEND_MODE = 'a+'


# defines the noise ceiling squared correlation values for EVC and IT, for the training (92, 118) and test (78) image sets

EVALUATE_DICT = {
    'fmri': {
        'target_names': ['EVC_RDMs', 'IT_RDMs'],
        '92': {
            'target_file': 'lib/target_files/target_fmri_92.mat',
            'nc_EVC_R2':  0.1589,
            'nc_IT_R2': 0.3075,
            'nc_avg_R2': (0.1589 + 0.3075)/2.
        },
        '118': {
            'target_file': 'lib/target_files/target_fmri_118.mat',
            'nc_EVC_R2':  0.1048,
            'nc_IT_R2': 0.0728,
            'nc_avg_R2': (0.1048 + 0.0728)/2.
        },
        '78': {
            'target_file': 'lib/target_files/target_fmri_challenge.mat',
            'nc_EVC_R2':  0.0640,
            'nc_IT_R2': 0.0647,
            'nc_avg_R2': (0.0640 + 0.0647)/2.
        }
    },
    'meg': {
        'target_names': ['MEG_RDMs_early', 'MEG_RDMs_late'],
        '92': {
            'target_file': 'lib/target_files/target_meg_92.mat',
            'nc_early_R2': 0.4634,
            'nc_late_R2': 0.2275,
            'nc_avg_R2': (0.4634+0.2275)/2.
        },
        '118': {
            'target_file': 'lib/target_files/target_meg_118.mat',
            'nc_early_R2': 0.3468,
            'nc_late_R2': 0.2265,
            'nc_avg_R2': (0.3468+0.2265)/2.
        },
        '78': {
            'target_file': 'lib/target_files/target_meg_challenge.mat',
            'nc_early_R2': 0.3562,
            'nc_late_R2': 0.4452,
            'nc_avg_R2': (0.3562+0.4452)/2.
        }
    }
}
