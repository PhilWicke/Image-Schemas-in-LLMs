# Exploring Spatial Schema Intuitions in Large Language and Vision Models

This repository holds code for the experiments of ongoing work with LLMs and Image Schemas. Feel free to use the conda environment through:

`conda env create -f schemas.yml`

You need to put your huggingface authetication token in the file:

`hf.key`

### Experiment 1 & Experiment 2: Gibbs  et al. (1994) & Beitel et al. (2001)

Original Papers:

[Taking a Stand on the Meanings of Stand: Bodily Experience as Motivation for Polysemy](https://doi.org/10.1093/jos/11.4.231)

[The Embodied Approach to the Polysemy of the Spatial Preposition On](https://doi.org/10.1075/cilt.177.11bei)



### Experiment 3: Richardson et al. (2001)

Original Paper: ["Language is Spatial": Experimental Evidence for Image Schemas of Concrete and Abstract
Verbs](https://escholarship.org/content/qt9vs820bx/qt9vs820bx.pdf)

Three versions of the first experiment by Richardson et. al (2001) have been implemented. 


**Version 01** (labelled "01") represents a **text-image version**. Here, different prompts have been tested to probe the LM with version of Unicode options with the final prompt being:

`"Select the image that best represents the event described by the sentence: "+action_word+"\n[◯→▢]\n\n[◯←▢]\n\n[◯\n↑\n▢]\n\n[◯\n↓\n▢]\n\nThe best representation is [◯"`

**Version 02** (labelled "02") represents a **text version**. It uses the same prompt, but the unicode has been replaced with words describing the cardinal direction of the image:

`"Select the CONCEPT that best represents the event described by the sentence: "+action_word+". CONCEPTS: UP, DOWN, LEFT, RIGHT.\nThe best representation is CONCEPT:"`

**Version 03** (labelled "03") represents an **image version**. It uses a similar prompt, but in this version a Vision-Language-Model provides an answer based in image information.
