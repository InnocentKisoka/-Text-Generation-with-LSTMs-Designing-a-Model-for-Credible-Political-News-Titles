This repository contains a project focused on generating news titles using Long Short-Term Memory (LSTM) networks. The objective is to train a model capable of generating credible news titles starting from a prompt based on real-world data. The dataset used is a collection of news articles with multiple categories, but this implementation focuses on the "Politics" category, containing 35,602 entries.
Key Features:
 Text Preprocessing: Includes tokenization, word-to-integer mappings, padding, and batching, ensuring that the data is ready for training.
LSTM Model: The core of the model is an LSTM network with embeddings and an optional stacked LSTM architecture. The model aims to minimize perplexity while generating believable news titles.
Evaluation: Implements two sampling strategies for sentence generation: random sampling and greedy sampling. Perplexity and loss metrics are used to evaluate the model's performance.
Training: The model is trained using standard backpropagation, with optional truncated backpropagation through time (TBBTT) for handling long sequences. The goal is to achieve a loss value below 1.5.
Visualization: Loss and perplexity plots are generated during training to visualize the model's improvement.
Generated Titles: At various training checkpoints, sample titles are generated to observe model progression and sentence quality.
Structure:
Data Preprocessing:
Access and filter the dataset to focus on the "Politics" category.
Tokenize titles and map words to integers.
Implement padding and batching strategies to handle variable-length sequences.
Model Architecture:
Use an LSTM network with embedding layers and possible stacked LSTM layers.
Implement dropout, fully connected layers, and a custom initial state method.
Sentence Generation:
Implement functions for random sampling and greedy sampling from the model's predicted distributions.
Generate news titles by prompting the model with a seed sentence and completing it using one of the sampling strategies.
Training:
Train the model using a standard loop, monitoring perplexity and loss at regular intervals.
Experiment with truncated backpropagation to improve efficiency and handle long sequences.
Evaluation:
Generate and evaluate news titles both during and after training using different sampling strategies.
Report on the credibility of the generated titles and the effectiveness of the trained model.
Bonus:
A bonus section explores the relationship between word embeddings, specifically whether similar operations in the model’s embedding space (like "King" - "Man" + "Woman" = "Queen") produce meaningful results.
