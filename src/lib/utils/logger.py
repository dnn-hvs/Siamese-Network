from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
USE_TENSORBOARD = True
try:
    import tensorboardX
    print('Using tensorboardX')
except:
    USE_TENSORBOARD = False


class Logger(object):
    def __init__(self, config):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        args = dict((name, getattr(config, name)) for name in dir(config)
                    if not name.startswith('_'))
        print(args)
        file_name = os.path.join(config.save_dir, 'config.txt')
        with open(file_name, 'wt') as config_file:
            config_file.write(
                '==> torch version: {}\n'.format(torch.__version__))
            config_file.write('==> cudnn version: {}\n'.format(
                torch.backends.cudnn.version()))
            config_file.write('==> Cmd:\n')
            config_file.write(str(sys.argv))
            config_file.write('\n==> config:\n')
            for k, v in sorted(args.items()):
                config_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = config.save_dir + '/logs_{}'.format(time_str)

        if not os.path.exists(os.path.dirname(log_dir)):
            os.mkdir(os.path.dirname(log_dir))
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.log = open(log_dir + '/log.txt', 'w')
        self.epoch_loss_file = open(log_dir+'/epoch_vs_loss.txt', 'w')
        self.log_dir = log_dir
        try:
            os.system('mv {}/config.txt {}/'.format(config.save_dir, log_dir))
        except:
            pass
        self.start_line = True

    def write(self, txt, log=True):
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        if not log:
            self.epoch_loss_file.write('{}: {}'.format(time_str, txt))
        else:
            if self.start_line:
                self.log.write('{}: {}'.format(time_str, txt))
            else:
                self.log.write(txt)

            self.start_line = False
            if '\n' in txt:
                self.start_line = True
                self.log.flush()

    def error(self, exception):
        error_file = open(self.log_dir + '/error.txt', 'w')
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        error_file.write(
            '{}: The Exception is :: {}'.format(time_str, exception))

    def close(self):
        self.log.close()
        self.epoch_loss_file.close()
