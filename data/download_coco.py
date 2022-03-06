import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("coco-2017", split="validation", dataset_dir='./')
session = fo.launch_app(dataset)