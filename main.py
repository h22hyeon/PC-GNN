import argparse
import yaml
import torch
import time
import numpy as np
from collections import defaultdict, OrderedDict
from src.utils import log
from src.model_handler import ModelHandler

################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config):
    ckp = log()
    config_lines = print_config(config)
    ckp.write_train_log(config_lines, print_line=False)
    ckp.write_test_log(config_lines, print_line=False)

    set_random_seed(config['seed']) # 실험에 사용된 seed를 가중치 이름에 넣어 저장되도록 한다.
    model = ModelHandler(config, ckp)
    f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.train()
    
    ckp.write_test_log("F1-Macro: {}".format(f1_mac_test))
    ckp.write_test_log("AUC: {}".format(auc_test))
    ckp.write_test_log("G-Mean: {}".format(gmean_test))


def multi_run_main(config):
    config_lines = print_config(config)
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list): # Multiple run일 경우 해당 실험에 사용된 seed를 기록한다.
            hyperparams.append(k)

    f1_list, f1_1_list, f1_0_list, auc_list, gmean_list = [], [], [], [], []
    # configuration 오브젝트들을 튜플로 저장한다.
    configs = grid(config)
    for i, cnf in enumerate(configs):
        ckp = log()
        print('Running {}:\n'.format(i))
        for k in hyperparams:
            cnf['save_dir'] += '{}_{}_'.format(k, cnf[k]) # Seed에 따라 실험별 가중치를 저장할 수 있도록 한다.
            ckp.write_train_log(config_lines + f"\nCurrent Seed: {cnf[k]}", print_line=False)
            ckp.write_test_log(config_lines + f"\nCurrent Seed: {cnf[k]}", print_line=False)
        print(cnf['save_dir'])
        set_random_seed(cnf['seed'])
        st = time.time()
        model = ModelHandler(cnf, ckp)
        # AUC-ROC / Gmean을 기록한다.
        f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.train()
        ckp.write_test_log("F1-Macro: {}".format(f1_mac_test))
        ckp.write_test_log("AUC: {}".format(auc_test))
        ckp.write_test_log("G-Mean: {}".format(gmean_test))
        f1_list.append(f1_mac_test)
        f1_1_list.append(f1_1_test)
        f1_0_list.append(f1_0_test)
        auc_list.append(auc_test)
        gmean_list.append(gmean_test)
        print("Running {} done, elapsed time {}s".format(i, time.time()-st))
    
    ckp2 = log()
    ckp2.multi_run_log(config_lines)
    ckp2.multi_run_log("F1-Macro: {}".format(f1_list))
    ckp2.multi_run_log("AUC: {}".format(auc_list))
    ckp2.multi_run_log("G-Mean: {}".format(gmean_list))

    # 기록된 AUC-ROC / Gmean의 평균을 계산하도록 한다.
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list, ddof=1)
    f1_1_mean, f1_1_std = np.mean(f1_1_list), np.std(f1_1_list, ddof=1)
    f1_0_mean, f1_0_std = np.mean(f1_0_list), np.std(f1_0_list, ddof=1)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list, ddof=1)
    gmean_mean, gmean_std = np.mean(gmean_list), np.std(gmean_list, ddof=1)

    ckp2.multi_run_log("F1-Macro: {}+{}".format(f1_mean, f1_std))
    ckp2.multi_run_log("F1-binary-1: {}+{}".format(f1_1_mean, f1_1_std))
    ckp2.multi_run_log("F1-binary-0: {}+{}".format(f1_0_mean, f1_0_std))
    ckp2.multi_run_log("AUC: {}+{}".format(auc_mean, auc_std))
    ckp2.multi_run_log("G-Mean: {}+{}".format(gmean_mean, gmean_std))



################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    # Configuration 파일을 불러와 train setting을 출력한다.
    config_lines = ""
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        line = "{} -->   {}".format(keystr, val)
        config_lines += lines
        print(line)
    print("**************** MODEL CONFIGURATION ****************")
    
    return config_lines

def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


################################################################################
# Module Command-line Behavior #
################################################################################
if __name__ == '__main__':
    cfg = get_args()
    # config 파서에 해당하는 configuration 파일의 오브젝트들을 튜플로 반환한다.
    config = get_config(cfg['config'])
    if cfg['multi_run']:
        multi_run_main(config)
    else:
        main(config)
