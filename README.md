# Character-level-RNN

This is an exercise in the Udacity course, Intro to Deep Learning with PyTorch. This repo is a character level generator/predector implementation in PyTorch. In this implementation, the model takes the pre-processed text data in one-hot encoding and tokenization as inputs and performs model training in character levels. After the training phase, we leverage the well-trained model to predict what the next character should be for a given input character.

# Usage
This script has been tested on anaconda3 and PyTorch 0.4.0.

Train a stacked LSTM network:
```
python character_gen.py --text_file_path=<path to a text file>
                        --saved_model_path=<path to a saved model>
```

Generate new text:
```
python character_gen.py --train=0 --test=1
```

# Generated Text
```
"Yes; I won't care."

"Which have not seemed to speak of the position is that you say that you will get up to the marning," he said at a strong woold. "And I have seemed, I can't stand to her, what do you know, that if it were this short
to sat," said Vronsky, who had
not become any many of his wife's significance
to her, he was
taking her head into the subject
and this solution had stopped a can is showing through their face and he could, and would
say in sick and
```
