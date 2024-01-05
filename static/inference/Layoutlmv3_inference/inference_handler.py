from .utils import load_model,load_processor,normalize_box,compare_boxes,adjacent
from .annotate_image import get_flattened_output,annotate_image
from PIL import Image,ImageDraw, ImageFont
import logging
import torch
import json
import os


logger = logging.getLogger(__name__)

class ModelHandler(object):
    def __init__(self):
        self.model = None
        self.model_dir = None
        self.device = 'cpu'
        self.error = None
        self.initialized = False
        self._raw_input_data = None
        self._processed_data = None
        self._images_size = None

    def initialize(self, context):
        try:
            logger.info("Loading transformer model")
            self._context = context
            properties = self._context
            self.model_dir = properties.get("model_dir")
            self.model = self.load(self.model_dir)
            self.initialized = True
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self.error = str(e)

    def preprocess(self, batch):
        try:
            inference_dict = batch
            self._raw_input_data = inference_dict
            processor = load_processor()
            images = [Image.open(path).convert("RGB")
                      for path in inference_dict['image_path']]
            self._images_size = [img.size for img in images]
            words = inference_dict['words']
            boxes = [[normalize_box(box, images[i].size[0], images[i].size[1])
                      for box in doc] for i, doc in enumerate(inference_dict['bboxes'])]
            encoded_inputs = processor(
                images, words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
            self._processed_data = encoded_inputs
            return encoded_inputs
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            self.error = str(e)
            return None

    def load(self, model_dir):
        try:
            model = load_model(model_dir)
            return model
        except Exception as e:
            logger.error(f"Error loading LayoutLMv3 model: {str(e)}")
            self.error = str(e)
            return None

    def inference(self, model_input):
        try:
            with torch.no_grad():
                inference_outputs = self.model(**model_input)
                predictions = inference_outputs.logits.argmax(-1).tolist()
            results = []
            for i in range(len(predictions)):
                tmp = dict()
                tmp[f'output_{i}'] = predictions[i]
                results.append(tmp)
            return [results]
        except Exception as e:
            logger.error(f"Error in inference: {str(e)}")
            self.error = str(e)
            return None

    def postprocess(self, inference_output):
        try:
            docs = []
            k = 0
            for page, doc_words in enumerate(self._raw_input_data['words']):
                doc_list = []
                width, height = self._images_size[page]
                for i, doc_word in enumerate(doc_words, start=0):
                    word_tagging = None
                    word_labels = []
                    word = dict()
                    word['id'] = k
                    k += 1
                    word['text'] = doc_word
                    word['pageNum'] = page + 1
                    word['box'] = self._raw_input_data['bboxes'][page][i]
                    _normalized_box = normalize_box(
                        self._raw_input_data['bboxes'][page][i], width, height)
                    for j, box in enumerate(self._processed_data['bbox'].tolist()[page]):
                        if compare_boxes(box, _normalized_box):
                            if self.model.config.id2label[inference_output[0][page][f'output_{page}'][j]] != 'O':
                                word_labels.append(
                                    self.model.config.id2label[inference_output[0][page][f'output_{page}'][j]][2:])
                            else:
                                word_labels.append('other')
                    if word_labels != []:
                        word_tagging = word_labels[0] if word_labels[0] != 'other' else word_labels[-1]
                    else:
                        word_tagging = 'other'
                    word['label'] = word_tagging
                    word['pageSize'] = {'width': width, 'height': height}
                    if word['label'] != 'other':
                        doc_list.append(word)
                spans = []
                def adjacents(entity): return [
                    adj for adj in doc_list if adjacent(entity, adj)]
                output_test_tmp = doc_list[:]
                for entity in doc_list:
                    if adjacents(entity) == []:
                        spans.append([entity])
                        output_test_tmp.remove(entity)

                while output_test_tmp != []:
                    span = [output_test_tmp[0]]
                    output_test_tmp = output_test_tmp[1:]
                    while output_test_tmp != [] and adjacent(span[-1], output_test_tmp[0]):
                        span.append(output_test_tmp[0])
                        output_test_tmp.remove(output_test_tmp[0])
                    spans.append(span)

                output_spans = []
                for span in spans:
                    if len(span) == 1:
                        output_span = {"text": span[0]['text'],
                                    "label": span[0]['label'],
                                    "words": [{
                                        'id': span[0]['id'],
                                        'box': span[0]['box'],
                                        'text': span[0]['text']
                                    }],
                                    }
                    else:
                        output_span = {"text": ' '.join([entity['text'] for entity in span]),
                                    "label": span[0]['label'],
                                    "words": [{
                                        'id': entity['id'],
                                        'box': entity['box'],
                                        'text': entity['text']
                                    } for entity in span]

                                    }
                    output_spans.append(output_span)
                docs.append({f'output': output_spans})
            return [json.dumps(docs, ensure_ascii=False)]

        except Exception as e:
            logger.error(f"Error in postprocessing: {str(e)}")
            self.error = str(e)
            return None


    def handle(self, data, context):
        try:
            if not self.initialized:
                self.initialize(context)

            if data is None:
                return None

            model_input = self.preprocess(data)
            if model_input is None:
                return None

            model_out = self.inference(model_input)
            if model_out is None:
                return None

            inference_out = self.postprocess(model_out)[0]
            with open('temp/LayoutlMV3InferenceOutput.json', 'w') as inf_out:
                inf_out.write(inference_out)
            inference_out_list = json.loads(inference_out)
            flattened_output_list = get_flattened_output(inference_out_list)
            print('Ready for Annotation')
            for i, flattened_output in enumerate(flattened_output_list):
                annotate_image(data['image_path'][i], flattened_output)
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            self.error = str(e)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
