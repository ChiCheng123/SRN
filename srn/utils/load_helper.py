import torch
import logging
logger = logging.getLogger('global')

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def restore_from(model, ckpt_path):
    logger.info('restore from {}'.format(ckpt_path))
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path, map_location = lambda storage, loc: storage.cuda(device))
    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    model.load_state_dict(ckpt_model_dict, strict=False)
    return model


