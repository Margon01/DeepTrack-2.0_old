import numpy as np
import tensorflow as tf
from tensorflow import keras
from . import utils
from . import commons as common
from . import backbone


class YOLObase(keras.Model):
    def __init__(
        self, input_size, NUM_CLASS, STRIDES, ANCHORS, XYSCALE, IOU_LOSS_THRESH
    ):
        """Base abstract implementation of yolo.

        Parameters
        ----------
        input_size: int
           Size of the images
        NUM_CLASS: int
           Number of distinct classes in training set
        STRIDES: list of int
           List of network strides
        ANCHORS: list of int
           List of anchor sizes
        XYSCALE: list of int
           List of rescale factors
        """
        super().__init__()
        self.input_size = input_shape
        self.num_class = NUM_CLASS
        self.strides = STRIDES
        self.anchors = ANCHORS
        self.xyscale = XYSCALE
        self.iou_loss_thresh = IOU_LOSS_THRESH
        self.score_threshold = 0.4

    def compute_loss(self, pred, conv, label, bboxes, i=0):
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = self.strides[i] * output_size
        conv = tf.reshape(
            conv, (batch_size, output_size, output_size, 3, 5 + self.num_class)
        )

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[
            :, :, :, :, 3:4
        ] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = utils.bbox_iou(
            pred_xywh[:, :, :, :, np.newaxis, :],
            bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :],
        )
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(
            max_iou < self.iou_loss_thresh, tf.float32
        )

        conf_focal = tf.pow(respond_bbox - pred_conf, 2)

        conf_loss = conf_focal * (
            respond_bbox
            * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=respond_bbox, logits=conv_raw_conf
            )
            + respond_bgd
            * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=respond_bbox, logits=conv_raw_conf
            )
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_prob, logits=conv_raw_prob
        )

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def train_step(self, data):
        image_data, target = data
        with tf.GradientTape() as tape:
            pred_result = self(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):

                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = self.compute_loss(
                    pred,
                    conv,
                    target[i][0],
                    target[i][1],
                    i=i,
                )
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            return {
                "loss": total_loss,
                "giou_loss": giou_loss,
                "conf_loss": conf_loss,
                "prob_loss": prob_loss,
            }

    def feature_maps_to_bbox_tensors(self, feature_maps, training=False):

        decode = decode_train if training else decode_detect

        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode(
                    fm,
                    self.input_size // 8,
                    self.num_class,
                    self.strides,
                    self.anchors,
                    i,
                    self.xyscale,
                )
            elif i == 1:
                bbox_tensor = decode(
                    fm,
                    self.input_size // 16,
                    self.num_class,
                    self.strides,
                    self.anchors,
                    i,
                    self.xyscale,
                )
            else:
                bbox_tensor = decode(
                    fm,
                    self.input_size // 32,
                    self.num_class,
                    self.strides,
                    self.anchors,
                    i,
                    self.xyscale,
                )
            if training:
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
            else:
                bbox_tensors.append(bbox_tensor[0])
                bbox_tensors.append(bbox_tensor[1])
        return bbox_tensors

    def filter_boxes(self, box_xywh, scores, input_shape):
        score_threshold = self.score_threshold
        scores_max = tf.math.reduce_max(scores, axis=-1)

        mask = scores_max >= score_threshold
        class_boxes = tf.boolean_mask(box_xywh, mask)
        pred_conf = tf.boolean_mask(scores, mask)
        class_boxes = tf.reshape(
            class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]]
        )
        pred_conf = tf.reshape(
            pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]]
        )

        box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

        input_shape = tf.cast(input_shape, dtype=tf.float32)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = (box_yx - (box_hw / 2.0)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.0)) / input_shape
        boxes = tf.concat(
            [
                box_mins[..., 0:1],  # y_min
                box_mins[..., 1:2],  # x_min
                box_maxes[..., 0:1],  # y_max
                box_maxes[..., 1:2],  # x_max
            ],
            axis=-1,
        )
        # return tf.concat([boxes, pred_conf], axis=-1)
        return (boxes, pred_conf)

    def call(self, x, training=False):
        y = self.model(x, training=training)
        decoded_y = self.feature_maps_to_bbox_tensors(y, training=training)
        if training:
            return decoded_y

        if not training:
            pred_bbox = decoded_y[::2]
            pred_prob = decoded_y[1::2]

            pred_bbox = tf.concat(pred_bbox, axis=1)
            pred_prob = tf.concat(pred_prob, axis=1)

            boxes, pred_conf = self.filter_boxes(
                pred_bbox,
                pred_prob,
                input_shape=tf.shape(x)[1:3],
            )

            return tf.concat([boxes, pred_conf], axis=-1)
        return y


import tensorflow as tf


class YOLOv3(YOLObase):
    def __init__(
        self, input_shape, NUM_CLASS,
        STRIDES=[8, 16, 32],
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
        ):
        super().__init__(
            input_shape[0], NUM_CLASS, STRIDES, ANCHORS, XYSCALE, IOU_LOSS_THRESH
        )

        input_layer = keras.layers.Input(input_shape)

        route_1, route_2, conv = backbone.darknet53(input_layer)

        conv = common.convolutional(conv, (1, 1, 1024, 512))
        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024, 512))
        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024, 512))

        conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = common.convolutional(
            conv_lobj_branch,
            (1, 1, 1024, 3 * (self.num_class + 5)),
            activate=False,
            bn=False,
        )

        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)

        conv = common.convolutional(conv, (1, 1, 768, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))

        conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = common.convolutional(
            conv_mobj_branch,
            (1, 1, 512, 3 * (self.num_class + 5)),
            activate=False,
            bn=False,
        )

        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)

        conv = common.convolutional(conv, (1, 1, 384, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))

        conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = common.convolutional(
            conv_sobj_branch,
            (1, 1, 256, 3 * (self.num_class + 5)),
            activate=False,
            bn=False,
        )

        self.model = keras.Model(input_layer, [conv_sbbox, conv_mbbox, conv_lbbox])


class YOLOv4(YOLObase):
    def __init__(self, input_shape, NUM_CLASS,
        STRIDES=[8, 16, 32],
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
        ):
        super().__init__(
            input_shape[0], NUM_CLASS, STRIDES, ANCHORS, XYSCALE, IOU_LOSS_THRESH
        )

        input_layer = keras.layers.Input(input_shape)
        route_1, route_2, conv = backbone.cspdarknet53(input_layer)

        route = conv
        conv = common.convolutional(conv, (1, 1, 512, 256))

        conv = common.upsample(conv)
        route_2 = common.convolutional(route_2, (1, 1, 512, 256))
        conv = tf.concat([route_2, conv], axis=-1)

        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))

        route_2 = conv
        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.upsample(conv)
        route_1 = common.convolutional(route_1, (1, 1, 256, 128))
        conv = tf.concat([route_1, conv], axis=-1)

        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))

        route_1 = conv
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = common.convolutional(
            conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False
        )

        conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
        conv = tf.concat([conv, route_2], axis=-1)

        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))

        route_2 = conv
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = common.convolutional(
            conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False
        )

        conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
        conv = tf.concat([conv, route], axis=-1)

        conv = common.convolutional(conv, (1, 1, 1024, 512))
        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024, 512))
        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024, 512))

        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = common.convolutional(
            conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False
        )

        return keras.Model(input_layer, (conv_sbbox, conv_mbbox, conv_lbbox))


def decode_train(
    conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]
):

    conv_output = tf.reshape(
        conv_output,
        (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS),
        name=f"reshape{i}",
    )

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(
        conv_output, (2, 2, 1, NUM_CLASS), axis=-1
    )

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(
        tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1]
    )

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (
        (tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid
    ) * STRIDES[i]
    pred_wh = tf.exp(conv_raw_dwdh) * ANCHORS[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_xywh = pred_xywh

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def decode_detect(
    conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]
):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(
        conv_output,
        (batch_size, output_size, output_size, 3, 5 + NUM_CLASS),
        name=f"reshape{i}",
    )

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(
        conv_output, (2, 2, 1, NUM_CLASS), axis=-1
    )

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (
        (tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid
    ) * STRIDES[i]
    pred_wh = tf.exp(conv_raw_dwdh) * ANCHORS[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob
