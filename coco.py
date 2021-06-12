# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import tqdm

from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation

from config import config as cfg
from dataset import DatasetRegistry, DatasetSplit
from pycocotools.coco import COCO

__all__ = ['register_coco']


class COCODetection(DatasetSplit):

    def __init__(self, basedir, split, folder_name=None):
        """
        Args:
            basedir (str): root of the dataset which contains the subdirectories for each split and annotations
            split (str): the name of the split, e.g. "train2017".
                The split has to match an annotation file in "annotations/" and a directory of images.

        Examples:
            For a directory of this structure:

            DIR/
              annotations/
                instances_XX.json
                instances_YY.json
              XX/
              YY/

            use `COCODetection(DIR, 'XX')` and `COCODetection(DIR, 'YY')`
        """
        if folder_name is None:
            folder_name = 'coco_' + split

        basedir = os.path.expanduser(basedir)
        self._imgdir = os.path.realpath(
            os.path.join(
                basedir, folder_name
            )
        )
        assert os.path.isdir(self._imgdir), f"{self._imgdir} is not a directory!"
        #annotation_file = os.path.join(
        #    basedir, 'annotations/instances_{}.json'.format(split))
        annotation_file = _get_annotation_file(basedir, folder_name)
        assert os.path.isfile(annotation_file), annotation_file

        self.coco = COCO(annotation_file)
        self.annotation_file = annotation_file
        logger.info(
            "Instances loaded from {0}. folder_name: {1} / split: {2} / imgdir: {3}".format(
                annotation_file, folder_name, split, self._imgdir
            )
        )

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    def print_coco_metrics(self, results):
        """
        Args:
            results(list[dict]): results in coco format
        Returns:
            dict: the evaluation metrics
        """
        from pycocotools.cocoeval import COCOeval
        ret = {}
        has_mask = "segmentation" in results[0]  # results will be modified by loadRes

        cocoDt = self.coco.loadRes(results)
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
        for k in range(6):
            ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

        if len(results) > 0 and has_mask:
            cocoEval = COCOeval(self.coco, cocoDt, 'segm')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            for k in range(6):
                ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
        return ret

    def load(self, add_gt=True, add_mask=False):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'image_id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        with timed_operation('Load annotations for {}'.format(
                os.path.basename(self.annotation_file))):
            img_ids = self.coco.getImgIds()
            img_ids.sort()
            # list of dict, each has keys: height,width,id,file_name
            imgs = self.coco.loadImgs(img_ids)

            for idx, img in enumerate(tqdm.tqdm(imgs)):
                img['image_id'] = img.pop('id')
                img['file_name'] = os.path.join(self._imgdir, img['file_name'])
                if idx == 0:
                    # make sure the directories are correctly set
                    assert os.path.isfile(img["file_name"]), img["file_name"]
                if add_gt:
                    self._add_detection_gt(img, add_mask)
            return imgs

    def _add_detection_gt(self, img, add_mask):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        If add_mask is True, also add 'segmentation' in coco poly format.
        """
        # ann_ids = self.coco.getAnnIds(imgIds=img['image_id'])
        # objs = self.coco.loadAnns(ann_ids)
        objs = self.coco.imgToAnns[img['image_id']]  # equivalent but faster than the above two lines
        if 'minival' not in self.annotation_file:
            # TODO better to check across the entire json, rather than per-image
            ann_ids = [ann["id"] for ann in objs]
            assert len(set(ann_ids)) == len(ann_ids), \
                "Annotation ids in '{}' are not unique!".format(self.annotation_file)

        # clean-up boxes
        width = img.pop('width')
        height = img.pop('height')

        all_boxes = []
        all_segm = []
        all_cls = []
        all_iscrowd = []
        for objid, obj in enumerate(objs):
            if obj.get('ignore', 0) == 1:
                continue
            x1, y1, w, h = list(map(float, obj['bbox']))
            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do make an assumption here that (0.0, 0.0) is upper-left corner of the first pixel
            x2, y2 = x1 + w, y1 + h

            # np.clip would be quite slow here
            x1 = min(max(x1, 0), width)
            x2 = min(max(x2, 0), width)
            y1 = min(max(y1, 0), height)
            y2 = min(max(y2, 0), height)
            w, h = x2 - x1, y2 - y1
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 1 and w > 0 and h > 0:
                all_boxes.append([x1, y1, x2, y2])
                # all_cls.append(self.COCO_id_to_category_id.get(obj['category_id'], obj['category_id']))
                all_cls.append(obj['category_id'])
                iscrowd = obj.get("iscrowd", 0)
                all_iscrowd.append(iscrowd)

                if add_mask:
                    segs = obj['segmentation']
                    if not isinstance(segs, list):
                        assert iscrowd == 1
                        all_segm.append(None)
                    else:
                        valid_segs = [np.asarray(p).reshape(-1, 2).astype('float32') for p in segs if len(p) >= 6]
                        if len(valid_segs) == 0:
                            logger.error("Object {} in image {} has no valid polygons!".format(objid, img['file_name']))
                        elif len(valid_segs) < len(segs):
                            logger.warn("Object {} in image {} has invalid polygons!".format(objid, img['file_name']))
                        all_segm.append(valid_segs)

        # all geometrically-valid boxes are returned
        if len(all_boxes):
            img['boxes'] = np.asarray(all_boxes, dtype='float32')  # (n, 4)
        else:
            img['boxes'] = np.zeros((0, 4), dtype='float32')
        cls = np.asarray(all_cls, dtype='int32')  # (n,)
        if len(cls):
            assert cls.min() > 0, "Category id in COCO format must > 0!"
        img['class'] = cls  # n, always >0
        img['is_crowd'] = np.asarray(all_iscrowd, dtype='int8')  # n,
        if add_mask:
            # also required to be float32
            img['segmentation'] = all_segm

    def training_roidbs(self):
        return self.load(add_gt=True, add_mask=cfg.MODE_MASK)

    def inference_roidbs(self):
        return self.load(add_gt=False)

    def eval_inference_results(self, results, output=None):
        continuous_id_to_COCO_id = {v: k for k, v in self.COCO_id_to_category_id.items()}
        for res in results:
            # convert to COCO's incontinuous category id
            # if res['category_id'] in continuous_id_to_COCO_id:
            #    res['category_id'] = continuous_id_to_COCO_id[res['category_id']]
            # COCO expects results in xywh format
            box = res['bbox']
            box[2] -= box[0]
            box[3] -= box[1]
            res['bbox'] = [round(float(x), 3) for x in box]

        if output is not None:
            with open(output, 'w') as f:
                json.dump(results, f)
        if len(results):
            # sometimes may crash if the results are empty?
            return self.print_coco_metrics(results)
        else:
            return {}


def _get_annotation_file(basedir, folder_name):
    ann_folder = os.path.join(basedir, folder_name)
    files = os.listdir(ann_folder)
    json_files = list(filter(lambda x: x.endswith('.json'), files))
    assert len(json_files) == 1, "Must have only one annotation json file"
    full_name = f'{basedir}{folder_name}/{json_files[0]}'
    logger.info(f"Annotation file: {full_name}")
    return full_name


def _get_class_names(basedir, name):
    # annotation_file = '{0}/annotations/instances_{1}.json'.format(basedir, name)
    annotation_file = _get_annotation_file(basedir, name)
    coco = COCO(annotation_file)
    categories = [''] * (len(coco.cats) + 1)
    categories[0] = 'BG'
    for k, category in coco.cats.items():
        categories[category['id']] = category['name']
    return categories


def register_coco(basedir):
    basedir_expanded = os.path.expanduser(basedir)

    for folder_name in os.listdir(basedir_expanded):
        # Filter out files and folders which does not start coco_
        if not os.path.isdir(os.path.join(basedir_expanded, folder_name)):
            continue
        if not folder_name.startswith('coco_'):
            continue
        split = folder_name.replace('coco_', '')
        class_names = _get_class_names(basedir_expanded, folder_name)
        logger.info('registering dataset {0}'.format(folder_name))
        DatasetRegistry.register(folder_name, lambda x=split: COCODetection(basedir_expanded, x))
        DatasetRegistry.register_metadata(folder_name, 'class_names', class_names)
        logger.info('dataset {0} registered.'.format(folder_name))

