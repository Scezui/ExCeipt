class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.model = None
        self.model_dir = None
        self.device = 'cpu'
        self.error = None
        # self._context = None
        # self._batch_size = 0
        self.initialized = False
        self._raw_input_data = None
        self._processed_data = None
        self._images_size = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        logger.info("Loading transformer model")

        self._context = context
        properties = self._context
        # self._batch_size = properties["batch_size"] or 1
        self.model_dir = properties.get("model_dir")
        self.model = self.load(self.model_dir)
        self.initialized = True

    def preprocess(self, batch):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        # assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
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

    def load(self, model_dir):
        """The load handler is responsible for loading the hunggingface transformer model.
        Returns:
            hf_pipeline (Pipeline): A Hugging Face Transformer pipeline.
        """
        # TODO model dir should be microsoft/layoutlmv2-base-uncased
        model = load_model(model_dir)
        return model

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # TODO load the model state_dict before running the inference
        # Do some inference call to engine here and return output
        with torch.no_grad():
            inference_outputs = self.model(**model_input)
            predictions = inference_outputs.logits.argmax(-1).tolist()
        results = []
        for i in range(len(predictions)):
            tmp = dict()
            tmp[f'output_{i}'] = predictions[i]
            results.append(tmp)

        return [results]