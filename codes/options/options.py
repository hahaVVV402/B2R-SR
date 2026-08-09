import os
import os.path as osp
import logging
import yaml
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()
REPO_ROOT = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))


def load_dotenv(root=REPO_ROOT):
    """Load a repository .env once without overriding the process environment."""
    path = osp.join(root, '.env')
    if not osp.isfile(path):
        return
    with open(path, mode='r') as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].strip()
            if '=' not in line:
                raise ValueError('Invalid .env line {}: {}'.format(number, raw.rstrip()))
            key, value = line.split('=', 1)
            key, value = key.strip(), value.strip()
            if (not key or not key.replace('_', '').isalnum()
                    or key[0].isdigit()):
                raise ValueError('Invalid .env key on line {}: {}'.format(number, key))
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


def _expand_environment(value):
    if isinstance(value, dict):
        return type(value)((key, _expand_environment(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if '${' in expanded:
            raise ValueError('Unresolved environment variable in option: {}'.format(value))
        return expanded
    return value


def load(opt_path):
    load_dotenv()
    with open(opt_path, mode='r') as handle:
        opt = yaml.load(handle, Loader=Loader)
    if not isinstance(opt, dict):
        raise TypeError('Option file must contain a YAML mapping: {}'.format(opt_path))
    return _expand_environment(opt)


def dump(opt, path):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, mode='w') as handle:
        yaml.dump(opt, handle, Dumper=Dumper, default_flow_style=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse(opt_path, is_train=True):
    opt = load(opt_path)
    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    opt['is_train'] = is_train
    if opt['distortion'] == 'sr':
        scale = opt['scale']

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if opt['distortion'] == 'sr':
            dataset['scale'] = scale
        is_lmdb = False
        if dataset.get('dataroot_GT', None) is not None:
            dataset['dataroot_GT'] = osp.expanduser(dataset['dataroot_GT'])
            if dataset['dataroot_GT'].endswith('lmdb'):
                is_lmdb = True
        if dataset.get('dataroot_LQ', None) is not None:
            dataset['dataroot_LQ'] = osp.expanduser(dataset['dataroot_LQ'])
            if dataset['dataroot_LQ'].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        if dataset['mode'].endswith('mc'):  # for memcached
            dataset['data_type'] = 'mc'
            dataset['mode'] = dataset['mode'].replace('_mc', '')

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)
    opt['path']['root'] = REPO_ROOT
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_state'] = osp.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = osp.join(experiments_root, 'val_images')

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # network
    if opt['distortion'] == 'sr' and opt['model']!="calssify_sr":
        opt['network_G']['scale'] = scale

    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get(
                'pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
