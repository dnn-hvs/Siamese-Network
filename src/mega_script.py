import yaml
import sys

args_galaxy = {}

with open('config.yaml') as ymfile:
    cfg = yaml.safe_load(ymfile)

    print(cfg)

    for arch in cfg:
        args_galaxy[arch] = []
        for task in ['fmri', 'meg']:
            for region in ['early', 'late']:
                for foveate in [True, False]:
                    args = {}
                    args[arch] = arch
                    args[task] = task
                    args[region] = region
                    args['train_dir'] = sys.argv[1]
                    args['test_dir'] = sys.argv[2]
                    args['foveate'] = foveate
                    args['load_model'] = ''
                    args['resume'] = False
                    args['lr'] = cfg[arch]['lr']
                    args['gpus'] = cfg[arch]['gpus']
                    args['optim'] = cfg[arch]['optim']
                    args['num_epochs'] = int(cfg[arch]['epochs'])
                    args['batch_size'] = int(cfg[arch]['batch_size'])
                    args['num_workers'] = int(
                        cfg[arch]['num_workers'])
                    args['num_freeze_layers'] = int(
                        cfg[arch]['num_freeze_layers'])
                    args_galaxy[arch].append(args)

print(args_galaxy.keys())
print(len(args_galaxy['alex']))
