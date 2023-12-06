# Claude2Alpaca7B-Playgorund
Repo of the code for Gradio Playground with Ctransformers of 7B parameters q4 GGUF model

## Instructions
- create a new directory
- Create a virtualEnvironment and activate it
- Install the dependencies
- Download the python file and the png file
- download from [Hugging Face Hub](https://huggingface.co/TheBloke/claude2-alpaca-7B-GGUF) the GGUF file claude2-alpaca-7b.Q4_0.gguf
- put it into the subfolder `model`

### Dependencies
```
pip install gradio
pip install ctransformers
```
### Run the GUI
from the terminal, with the venv activated, run
```
python claude2AlpacaPG_full.py
```
