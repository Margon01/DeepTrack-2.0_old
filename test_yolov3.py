#%%

import deeptrack as dt
import skimage.measure as measure
import numpy as np

im_size = 256
scatterer_a = dt.Ellipse(
    radius=8 * dt.units.px, position=lambda: np.random.rand(2) * 256, class_index=1
)
scatterer_b = dt.Ellipse(
    radius=(12, 7) * dt.units.px,
    rotation=lambda: np.random.rand() * np.pi * 2,
    position=lambda: np.random.rand(2) * 256,
    class_index=2,
)

optics = dt.Fluorescence(output_region=(0, 0, im_size, im_size))

scatterer_a = scatterer_a ^ (lambda: np.random.randint(1, 4))
scatterer_b = scatterer_b ^ (lambda: np.random.randint(1, 4))
sample = scatterer_a & scatterer_b


image = optics(sample)
# image.plot()

#%%


class ForEach(dt.Feature):
    __distributed__ = False

    def __init__(self, feature, **kwargs):
        super().__init__(**kwargs)
        self.feature = self.add_feature(feature)

    def get(self, list_of_ims, **_):
        return [self.feature(im) for im in list_of_ims]


def get_projected_mask(image):
    return np.any(image > 0, axis=-1, keepdims=True) * image.get_property("class_index")


as_masks = dt.SampleToMasks(
    lambda: get_projected_mask,
    number_of_masks=1,
    output_region=optics.output_region,
)

masks = sample >> ForEach(as_masks)


def masks_to_bbox(layer):

    bw_layer = layer > 0
    label = measure.label(bw_layer)
    CC = measure.regionprops(label)
    left, top, _, right, bottom, _ = CC[0].bbox

    return [top, left, bottom - top, right - left, int(np.max(layer))]


bboxes = masks >> masks_to_bbox

#%%
import matplotlib.pyplot as plt

data = image & bboxes


res, *bboxes_res = data.update()()

# plt.imshow(res)
# plt.show()
# %%
from deeptrack.models.yolo.yolo import YOLOv3

model = YOLOv3(
    (256, 256, 1),
    3,
    [8, 16, 32],
    ANCHORS=[
        10,
        13,
        16,
        30,
        33,
        23,
        30,
        61,
        62,
        45,
        59,
        119,
        116,
        90,
        156,
        198,
        373,
        326,
    ],
    XYSCALE=[1.2, 1.1, 1.05],
    IOU_LOSS_THRESH=0.5,
)
#%%
model.compile(optimizer="adam")
#%%

from deeptrack.models.yolo.dataset import YoloDataGenerator

generator = YoloDataGenerator(
    feature=data,
    input_size=np.array([256]),
    num_class=3,
    label_function=lambda d: np.array(d[1:]),
    batch_size=16,
    min_data_size=100,
    max_data_size=200,
)

generator.anchors = model.anchors
#%%
with generator:
    model.fit(generator, epochs=10)
# %%

A, B = generator[0]

# %%
