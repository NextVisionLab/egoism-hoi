# Exploiting Multimodal Synthetic Data for Egocentric Human-Object Interaction Detection in an Industrial Scenario
## Prerequisites
* Python==3.9
* Pytorch>=1.9.0

## Installation
Create a new conda env
```
conda create --name ego_hoi python=3.9
conda activate ego_hoi
```

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

## Inference
Run the command below for an example of inference. A new folder **output_detection** will be created with the visualization. Check more about argparse parameters in inference.py.
```
python inference.py --weights_path <weights_path> --images_path <images_path> --video_path <video_path>
```