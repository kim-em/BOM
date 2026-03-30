from .unet import TinyUNet
from .simvp import SimVPLite
from .convgru import ConvGRUNet
from .ds_convlstm import DSConvLSTMNet
from .flownet import FlowNowcaster

MODEL_REGISTRY = {
    'unet': TinyUNet,
    'simvp': SimVPLite,
    'convgru': ConvGRUNet,
    'ds_convlstm': DSConvLSTMNet,
    'flow': FlowNowcaster,
}
