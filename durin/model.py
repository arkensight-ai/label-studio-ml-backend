import os
import pathlib
import logging
import torch
from collections import OrderedDict

from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase, ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from aule.groundingdino.models.GroundingDINO.groundingdino import build_predictor
from aule.data.transforms import make_transforms, collate_grounding
from PIL import Image
from label_studio_ml.utils import (
    get_image_size,
    get_single_tag_keys,
    DATA_UNDEFINED_NAME,
)
from aule.utils.detections import filter_detections
from lxml import etree

os.environ["LABEL_STUDIO_URL"] = "http://192.168.194.233:8080/"
os.environ["LABEL_STUDIO_ACCESS_TOKEN"] = "39a37704f39832438351055116c8eed4de80940e"

def clean_state_dict(state_dict, remove_query_generator_weight: bool = True):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'bert.embeddings.position_ids' in k:
            continue
        
        k = k.removeprefix("module.").removeprefix("model.")
        if remove_query_generator_weight and "dn_query_generator" in k:
            continue
        new_state_dict[k] = v
    return new_state_dict

from typing import List

logger = logging.getLogger(__name__)

BOX_THRESHOLD = os.environ.get("BOX_THRESHOLD", 0.2)
TEXT_THRESHOLD = os.environ.get("TEXT_THRESHOLD", 0.2)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device {device}")

groundingdino_model = build_predictor()
groundingdino_model.to(device).eval()

checkpoint = torch.load(
    "durin/checkpoints/rotterdam-youtube.pth",
    weights_only=True,
)
groundingdino_model.load_state_dict(clean_state_dict(checkpoint), strict=True)

transforms = make_transforms("predict")


class GroundingDINO(LabelStudioMLBase):
    
    def setup(self):
        label_config = self.label_config
        tree = etree.fromstring(label_config)
        self.labels = [node.attrib["value"] for node in tree.iter("Label")]

    
    def _get_thresholds(self, annotation: Optional[Dict] = None) -> Dict:
        out = {}
        try:
            from_name_box, _, _ = self.get_first_tag_occurence(
                'Number', 'Image', name_filter=lambda n: n.startswith('box_threshold'))
        except Exception as e:
            logger.warning(f"Error getting box_threshold: {e}. Use default values: {BOX_THRESHOLD}")
            out['box_threshold'] = BOX_THRESHOLD
            out['from_name_box'] = None
        else:
            if annotation and 'result' in annotation:
                out['box_threshold'] = next((r['value']['number'] for r in annotation['result'] if r['from_name'] == from_name_box), None)
            else:
                out['box_threshold'] = self.get(from_name_box)

            if not out['box_threshold']:
                out['box_threshold'] = BOX_THRESHOLD
            out['from_name_box'] = from_name_box

        try:
            from_name_text, _, _ = self.get_first_tag_occurence(
                'Number', 'Image', name_filter=lambda n: n.startswith('text_threshold'))
        except Exception as e:
            logger.warning(f"Error getting text_threshold: {e}. Use default values: {TEXT_THRESHOLD}")
            out['text_threshold'] = TEXT_THRESHOLD
            out['from_name_text'] = None
        else:
            if annotation and 'result' in annotation:
                out['text_threshold'] = next((r['value']['number'] for r in annotation['result'] if r['from_name'] == from_name_text), None)
            else:
                out['text_threshold'] = self.get(from_name_text)

            if not out['text_threshold']:
                out['text_threshold'] = TEXT_THRESHOLD
            out['from_name_text'] = from_name_text

        logger.info(f"Thresholds: {out}")
        return out

    def get_results(self, all_points, all_scores, all_labels, all_lengths, from_name_r, to_name_r):

        results = []
        total_score = 0
        for points, scores, lengths, labels in zip(all_points, all_scores, all_lengths, all_labels):
            # random ID
            label_id = str(uuid4())[:9]

            height, width = lengths
            score = scores.item()
            total_score += score

            results.append({
                'id': label_id,
                'from_name': from_name_r,
                'to_name': to_name_r,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'rotation': 0,
                    'width': float((points[2] - points[0]) * 100),
                    'height': float((points[3] - points[1]) * 100),
                    'x': float(points[0] * 100),
                    'y': float(points[1] * 100),
                    'rectanglelabels': [labels],
                },
                'score': score,
                'type': 'rectanglelabels',
            })

        total_score /= max(len(results), 1)

        return {
            'result': results,
            'score': total_score,
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:

        print("TASKS", tasks, context)
        assert len(tasks) == 1, "Only one task is supported for now"
        task = tasks[0]

        cap_list = self.labels
        prompt = ". ".join(map(str.lower, cap_list)) + "."
        if not prompt:
            logger.warning("Prompt not found")
            ModelResponse(predictions=[])

        from_name_r, to_name_r, value = self.get_first_tag_occurence('RectangleLabels', 'Image')

        thresh_controls = self._get_thresholds(context)
        BOX_THRESHOLD = float(thresh_controls['box_threshold'])
        TEXT_THRESHOLD = float(thresh_controls['text_threshold'])

        all_points = []
        all_scores = []
        all_lengths = []
        raw_img_path = task['data'][value]

        try:
            img_path = get_local_path(
                raw_img_path,
                task_id=task.get('id')
            )
        except Exception as e:
            logger.error(f"Error getting image path: {e}")
            return ModelResponse(predictions=[])

        image = Image.open(img_path).convert('RGB')
        transformerd_image, _ = transforms(image, {})
        inputs = collate_grounding([{
            'image': transformerd_image.to(device),
            'caption': prompt,
            'cap_list': cap_list,
        }])

        
        with torch.inference_mode() and torch.autocast("cuda", dtype=torch.float16):
            results = groundingdino_model.predict(
                **inputs,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )[0]

        H = image.height
        W = image.width
        
        boxes_xyxy = results['boxes']
        scores = results['scores']
        labels = results['labels']
        label_names = results['label_names']
        
        indices_kept = filter_detections(
            boxes_xyxy,
            scores,
            labels,
            score_threshold=TEXT_THRESHOLD,
            iou_threshold=0.5,
            overlap_threshold=0.75,
        )
        boxes_xyxy = boxes_xyxy[indices_kept]
        scores = scores[indices_kept]
        labels = labels[indices_kept]
        label_names = [label_names[idx] for idx in indices_kept]
        
        points = boxes_xyxy.cpu().numpy()

        for point, logit in zip(points, scores):
            all_points.append(point)
            all_scores.append(logit)
            all_lengths.append((H, W))

        predictions = self.get_results(all_points, all_scores, label_names, all_lengths, from_name_r, to_name_r)

        return ModelResponse(predictions=[predictions])

    def fit(self, event, data, **additional_params):
        logger.debug(f'Data received: {data}')
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            return

        prompt = self._get_prompt(data['annotation'])
        if prompt:
            logger.info(f'Storing prompt: {prompt}')
            self.set('prompt', prompt['prompt'])
        else:
            logger.warning("Prompt not found")

        th = self._get_thresholds(data['annotation'])
        self.set('BOX_THRESHOLD', str(th['box_threshold']))
        self.set('TEXT_THRESHOLD', str(th['text_threshold']))
