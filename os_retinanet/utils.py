from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import cv2
import numpy as np
import os
import sys



# ===================================================
# ------------------visualize_image------------------
# ===================================================
"""
This script is adapted from
https://github.com/Project-MONAI/tutorials/blob/main/detection/visualize_image.py
"""
def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def visualize_one_xy_slice_in_3d_image(gt_boxes, image, pred_boxes, gt_box_index=0):
    """
    Prepare a 2D xy-plane image slice from a 3D image for visualization.
    It draws the (gt_box_index)-th GT box and predicted boxes on the same slice.
    The GT box will be green rect overlayed on the image.
    The predicted boxes will be red boxes overlayed on the image.

    Args:
        gt_boxes: numpy sized (M, 6)
        image: image numpy array, sized (H, W, D)
        pred_boxes: numpy array sized (N, 6)
    """
    draw_box = gt_boxes[gt_box_index, :]
    draw_box_center = [round((draw_box[axis] + draw_box[axis + 3] - 1) / 2.0) for axis in range(3)]
    draw_box = np.round(draw_box).astype(int).tolist()
    draw_box_z = draw_box_center[2]  # the z-slice we will visualize

    # draw image
    draw_img = normalize_image_to_uint8(image[:, :, draw_box_z])
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)

    # draw GT box, notice that cv2 uses Cartesian indexing instead of Matrix indexing.
    # so the xy position needs to be transposed.
    # cv2.rectangle(
    #     draw_img,
    #     pt1=(draw_box[1], draw_box[0]),
    #     pt2=(draw_box[4], draw_box[3]),
    #     color=(0, 255, 0),  # green for GT
    #     thickness=1,
    # )
    for bbox in gt_boxes:
        bbox = np.round(bbox).astype(int).tolist()
        if bbox[5] < draw_box[2] or bbox[2] > draw_box[5]:
            continue
        cv2.rectangle(
            draw_img,
            pt1=(bbox[1], bbox[0]),
            pt2=(bbox[4], bbox[3]),
            color=(0, 255, 0),  # red for predicted box
            thickness=1,
        )

    # draw predicted boxes
    for bbox in pred_boxes:
        bbox = np.round(bbox).astype(int).tolist()
        if bbox[5] < draw_box[2] or bbox[2] > draw_box[5]:
            continue
        cv2.rectangle(
            draw_img,
            pt1=(bbox[1], bbox[0]),
            pt2=(bbox[4], bbox[3]),
            color=(255, 0, 0),  # red for predicted box
            thickness=1,
        )
    return draw_img


# ===================================================
# ------------------warmup_scheduler-----------------
# ===================================================
"""
This script is adapted from
https://github.com/Project-MONAI/tutorials/blob/main/detection/warmup_scheduler.py
https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
"""
class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0.
                    If multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler (e.g., ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            multiplier (float): Target learning rate multiplier.
            total_epoch (int): Target learning rate is reached at total_epoch, gradually.
            after_scheduler: After target_epoch, use this scheduler (e.g., ReduceLROnPlateau).
        """
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater than or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        """
        Compute the learning rate at the current epoch.
        """
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        """
        Adjust learning rate when using ReduceLROnPlateau after warmup.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        """
        Perform a single step.
        """
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# ===================================================
# ------------------useful functions-----------------
# ===================================================
def extract_patches(img, box) -> list:
    patches = []
    for item in box:
        print("****"*15)
        print(item)
        print(f"x --> {int(item[3]) - int(item[0])}")
        print(f"y --> {int(item[4]) - int(item[1])}")
        print(f"z --> {int(item[5]) - int(item[2])}")
        patches.append(img[:, int(item[0]):int(item[3]),
                              int(item[1]):int(item[4]),
                              int(item[2]):int(item[5])])
    return patches


def make_dirs(output_path):
    try:
        os.mkdir(output_path)
        print("The parent path successfully created: ", end='')
        print(output_path)
    except FileExistsError:
        sys.exit(FileExistsError)
    try:
        os.mkdir(output_path+"/tfevents/")
        print("The Tensorboard path successfully created...")
    except FileExistsError:
        sys.exit(FileExistsError)
    try:
        os.mkdir(output_path+"/model/")
        print("The Model path successfully created...")
    except FileExistsError:
        sys.exit(FileExistsError)

    try:
        os.mkdir(output_path + "/smpl_outputs/")
        print("The samples path successfully created...")
    except FileExistsError:
        sys.exit(FileExistsError)


