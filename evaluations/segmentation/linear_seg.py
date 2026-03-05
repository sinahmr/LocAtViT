from mmcv.cnn import ConvModule
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class LinearSegHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, inputs):
        x = inputs[0]
        x = self.conv(x)
        out = self.cls_seg(x)
        return out
