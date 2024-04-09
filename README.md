# mlx-clay
A basic script for running an ensemble of TinyLlamas on MLX to annotate an Argilla Dataset

### Getting started with mlx-clay

You can use [https://github.com/stevengans/ollama-clay](https://github.com/stevengans/ollama-clay) as a starting point and fill-in and customize the sections required. You will need to create an Argilla dataset and also download TinyLlama weights, see the dependencies for more details:

##### Dependencies:
- TinyLlama weights: [link to weights](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Argilla dataset: [link to repo](https://github.com/argilla-io/argilla)
- `pip install requirements.txt`

Once the above steps are complete you can run ```python tinyllamascanjudge.py```. (You can also adjust the number of TinyLlamas to run)
