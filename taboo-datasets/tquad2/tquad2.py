import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive

_CITATION = ""

_DESCRIPTION = ""

_URL = "https://huggingface.co/datasets/husnu/tquad2/raw/main/"
_URLS = {
    "train": _URL + "tquad_train_data_v2.json",
    "dev": _URL + "tquad_dev_data_v2.json",
}


class TQuAD2Config(datasets.BuilderConfig):
    """BuilderConfig for TQuAD2."""

    def __init__(self, **kwargs):
        """BuilderConfig for TQuAD2.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TQuAD2Config, self).__init__(**kwargs)


class TQuAD2(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        TQuAD2Config(name="tquad2", version=datasets.Version("2.0.0"), description="TQuAD2 dataset"),
    ]

    IDS_ = []

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://huggingface.co/datasets/husnu/tquad2",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for example in squad["data"]:
                title = example.get("title", "")
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        question = qa["question"]
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"] for answer in qa["answers"]]

                        # if id_ is already in the dataset, we skip it
                        while id_ in self.IDS_:
                            if isinstance(id_, int):
                                id_ = id_ + 1
                            else:
                                id_ = id_ + "_duplicate"

                        self.IDS_.append(id_)

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
