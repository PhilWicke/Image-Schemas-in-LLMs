# Image Schemas in LLMs (Working Title)

This repository holds code for the experiments of ongoing work with LLMs and Image Schemas. Feel free to use the conda environment through:

`conda env create -f schemas.yml`

You need to put your huggingface authetication token in the file:

`hf.key`

### Experiment 01: Richardson et. al (2001)

Paper: ["Language is Spatial": Experimental Evidence for Image Schemas of Concrete and Abstract
Verbs](https://escholarship.org/content/qt9vs820bx/qt9vs820bx.pdf)

Two different versions of the first experiment by Richardson et. al (2001) have been implemented. In the first version (labelled "01"), different prompts have been tested to probe the LM with version of Unicode options with the final prompt being:

"Select the image that best represents the event described by the sentence: "+action_word+"\n[◯→▢]\n\n[◯←▢]\n\n[◯\n↑\n▢]\n\n[◯\n↓\n▢]\n\nThe best representation is [◯"
