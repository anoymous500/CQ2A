# CQ2A
# Environment
`pip install -r requirements.txt`
# Dataset
The folder `Dataset` contains the context derived from the test set of Boolq\NarrativeQA and Squad2. In our method, we will use these contexts to generate questions.
# Run
For entity question generation
`python entity_generation.py`
For relation question generation
`python relation_generation.py`
# manual_check
The results of the manual annotation in the experiment are in the `manual_check.zip`
# prompt
The prompts we use in the experiment are included in `prompt_list`
