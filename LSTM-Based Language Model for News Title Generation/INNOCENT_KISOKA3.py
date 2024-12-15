'''
Assignment 3
INNOCENT KISOKA
'''
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt

# Set the seed
seed = 42
torch.manual_seed(seed)
# Probably, this below must be changed if you work with a M1/M2/M3 Mac
torch.cuda.manual_seed(seed) # for CUDA
torch.backends.cudnn.deterministic = True # for CUDNN
torch.backends.benchmark = False # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

# Auxiliary function
def keys_to_values(keys, map, default_if_missing=None):
  return [map.get(key, default_if_missing) for key in keys]

if __name__ == "__main__":
    '''
    Data
    '''

    # Question 1
    ds = load_dataset("heegyu/news-category-dataset")
    print("Initial dataset")
    print(ds['train'])
    print()

    # TODO: what's next?
    # Question 2
    # Filter politics news
    politics_news = ds['train'].filter(lambda x: x['category'] == 'POLITICS')
    print(f"Number of politics news: {len(politics_news)}")


    print("Filtered dataset")
    print(politics_news)
    print()


    # Question 3
    # Tokenize titles
    tokenized_politics_news = politics_news.map(
        lambda x: {
            'tokenized_headline': [word.lower() for word in x['headline'].split(" ")] + ["<EOS>"]
        }
    )

    # Print the first 3 tokenized headlines
    print("First three tokenized headlines:")
    for i in range(3):
        print(tokenized_politics_news[i]['tokenized_headline'])
    print()

    # Question 4
    word_vocab = ["<EOS>"] + sorted(set(word.lower() for item in tokenized_politics_news['tokenized_headline'] for word in item)) + ["PAD"]
    word_to_int = {word: idx for idx, word in enumerate(word_vocab)}
    int_to_word = {idx: word for word, idx in word_to_int.items()}
    print("Vocabulary:")
    # print(word_vocab)
    print()
    print("Word to integer mapping:")
    # print(word_to_int)
    print()
    print("Integer to word mapping:")
    # print(int_to_word)
    print()

    # Report 5 most common words
    word_counts = Counter(word for item in tokenized_politics_news['tokenized_headline'] for word in item)
    most_common_words = word_counts.most_common(5)
    print("5 most common words:", most_common_words)

    # Total number of unique words
    vocab_size = len(word_vocab)
    print("Number of unique words I ended up with:", vocab_size)
    print()

    # Question 5

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data_as_str, word_to_int):
            """
            Initialize the dataset with tokenized word sequences and word-to-integer mapping.

           
            """
            self.data_as_int = []

            
            for seq_as_str in data_as_str:
                seq_as_int = keys_to_values(seq_as_str, word_to_int, word_to_int.get("<PAD>", 0))
                self.data_as_int.append(seq_as_int)

        def __len__(self):
            """
            Return the number of sequences in the dataset.
            """
            return len(self.data_as_int)

        def __getitem__(self, ix):
            """
            Retrieve the input-target pair for a certain index.

            
            """
            item = self.data_as_int[ix]

            # Slice x and y from sample
            x = item[:-1]  # All except the last word
            y = item[1:]   # All except the first word
            return torch.tensor(x), torch.tensor(y)

    dataset = Dataset(tokenized_politics_news['tokenized_headline'], word_to_int)
    # Access dataset items
    print("First Item (Input, Target):", dataset[0])
    # Dataset length
    print("Dataset size:", len(dataset))
    print()

    # Question 6
    def collate_fn(batch, pad_value):
      
      data, targets = zip(*batch)

      # Pad sequences using the pad_sequence function
      padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=pad_value)
      padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_value)

      return padded_data, padded_targets

    batch_size = 32
    if batch_size == 1:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                collate_fn=lambda b: collate_fn(b, word_to_int["PAD"]),
                                shuffle=True)

    # Iterate through the DataLoader
    for padded_inputs, padded_targets in dataloader:
        print(padded_inputs.shape)  # Shape: (batch_size, max_seq_len)
        print(padded_targets.shape)  # Shape: (batch_size, max_seq_len)
        break  # Stop after the first batch
    print()

    '''
    Model definition 
    '''
    class Model(nn.Module):
      def __init__(self, map, hidden_size, emb_dim, n_layers=2, dropout_p=0.2):
          super(Model, self).__init__()

          self.vocab_size  = len(map)
          self.hidden_size = hidden_size
          self.emb_dim     = emb_dim
          self.n_layers    = n_layers
          self.dropout_p   = dropout_p

          # Embedding layer
          self.embedding = nn.Embedding(
              num_embeddings=self.vocab_size,
              embedding_dim=self.emb_dim,
              padding_idx=map["PAD"]
          )

          # LSTM layer (stacked)
          self.lstm = nn.LSTM(
              input_size=self.emb_dim,
              hidden_size=self.hidden_size,
              num_layers=self.n_layers,
              batch_first=True,
              dropout=self.dropout_p if self.n_layers > 1 else 0  # Apply dropout only if n_layers > 1
          )

          # Dropout layer for regularization
          self.dropout = nn.Dropout(self.dropout_p)

          # Fully connected layer to predict next word in sequence
          self.fc = nn.Linear(self.hidden_size, self.vocab_size)

      def forward(self, x, prev_state):
          

          # Embedding lookup
          embed = self.embedding(x)  # Shape: (batch_size, seq_length, emb_dim)

          # Forward pass through LSTM
          output, (h_n, c_n) = self.lstm(embed, prev_state)  # output: (batch_size, seq_length, hidden_size)

          # Apply dropout to the LSTM output
          output = self.dropout(output)

          # Output projection to vocabulary size (next word prediction)
          out = self.fc(output)  # Shape: (batch_size, seq_length, vocab_size)

          return out, (h_n, c_n)

      def init_state(self, b_size=1):
          """
          Initialize the hidden state (h_0) and cell state (c_0) for the LSTM.
          Shape: (num_layers, batch_size, hidden_size)
          """
          device = next(self.parameters()).device  # Get the device the model is on
          h_0 = torch.zeros(self.n_layers, b_size, self.hidden_size).to(device)
          c_0 = torch.zeros(self.n_layers, b_size, self.hidden_size).to(device)
          return h_0, c_0
    '''
    Evaluation - part 1
    '''
    # TODO
    def random_sample_next(probs):
        """
        Randomly sample the next word based on the probability distribution
        Args:
       
        """
        # Convert probabilities to a distribution
        probs = torch.softmax(probs, dim=-1)
        # Sample from the distribution
        return torch.multinomial(probs, 1).item()

    def sample_argmax(probs):
        """
        Pick the word with highest probability
        Args:
            probs: probability distribution over vocabulary
       
        """
        return torch.argmax(probs).item()

    def sample(prompt, model, sampling_strategy, max_len=20):
        """
        Generate a complete sentence starting from a prompt
        """
        model.eval()
        device = next(model.parameters()).device  
        
        # Convert prompt words to indices
        input_indices = [word_to_int.get(word.lower(), word_to_int["PAD"]) for word in prompt]
        # Create tensor with shape (1, sequence_length) and move to correct device
        input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)
        
        # Initialize model state
        state = model.init_state()
        
        # Generate words until EOS token or max length reached
        generated = list(prompt)
        
        for _ in range(max_len):
            # Get model predictions
            with torch.no_grad():
                output, state = model(input_tensor, state)
            
            # Get probabilities for next word
            next_word_probs = output[0, -1]  # Take last timestep
            
            # Sample next word
            next_word_idx = sampling_strategy(next_word_probs)
            
            # Convert to word and append to generated sequence
            next_word = int_to_word[next_word_idx]
            generated.append(next_word)
            
            # Stop if EOS token generated
            if next_word == "<EOS>":
                break
                
            # Update input for next iteration - shape should be (1, 1)
            input_tensor = torch.tensor([[next_word_idx]]).to(device)
        
        model.train()
        
        return generated

    # Evaluate model before training

    """
    Perform pre-training evaluation by generating text using both sampling strategies.
    """
    print("Pre-training evaluation:")
    model = Model(word_to_int, hidden_size=128, emb_dim=8, n_layers=2, dropout_p=0.2)
    prompt = ["the", "president", "wants"]
    
    print("\nSampling strategy:")
    for i in range(3):
        generated = sample(prompt, model, random_sample_next)
        print(f"Generation {i+1}: {' '.join(generated)}")
        
    print("\nGreedy strategy:")
    for i in range(3):
        generated = sample(prompt, model, sample_argmax)
        print(f"Generation {i+1}: {' '.join(generated)}")
    print()
    '''
    Training
    '''
    # Add device configuration at the start
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Training parameters
    num_epochs = 12
    learning_rate = 0.001
    batch_size = 32
    clip_value = 1.0

    # Initialize model with given parameters and move to device
    model = Model(
        word_to_int, 
        hidden_size=1024,
        emb_dim=150,
        n_layers=1,
        dropout_p=0.2
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Lists to store metrics
    train_losses = []
    perplexities = []
    
    # Standard training loop
   
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_num, (inputs, targets) in enumerate(dataloader):
            # Move batch to device
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Initialize hidden state with the current batch size (on device)
            current_batch_size = inputs.size(0)
            state = tuple(s.to(DEVICE) for s in model.init_state(current_batch_size))
            
            optimizer.zero_grad()
            output, state = model(inputs, state)
            
            # Rest of the training loop remains the same
            output = output.view(-1, len(word_to_int))
            targets = targets.view(-1)
            
            loss = criterion(output, targets)
            epoch_loss += loss.item()
            num_batches += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            
            state = tuple(s.detach() for s in state)
        
        # Calculate average loss and perplexity
        avg_loss = epoch_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        train_losses.append(avg_loss)
        perplexities.append(perplexity.item())
        
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Perplexity: {perplexity:.4f}")
        
        # Generate sample text at different training stages
        if epoch == 0 or epoch == num_epochs//2 or epoch == num_epochs-1:
            print(f"\n--- Generated Text at Epoch {epoch+1} ---")
            prompt = ["the", "president", "wants"]
            generated = sample(prompt, model, random_sample_next)
            print(f"Prompt used : '{' '.join(prompt)}'")
            print(f"Sentence generated: '{' '.join(generated)}'\n")
            print()
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.axhline(y=1.5, color='r', linestyle='--', label='Target Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(perplexities)
    plt.title('Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    print("\nStarting TBPTT training...")
    
    # Reset model  for TBPTT
    model = Model(
        word_to_int, 
        hidden_size=2048, 
        emb_dim=150,
        n_layers=1,
        dropout_p=0.2
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 5
    tbptt_losses = []
    tbptt_perplexities = []
    tbptt_chunk = 20  

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_chunks = 0
        
        for batch_num, (inputs, targets) in enumerate(dataloader):
            # Move data to device
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Initialize state with the current batch size
            current_batch_size = inputs.size(0)
            state = tuple(s.to(DEVICE) for s in model.init_state(current_batch_size))
            
            optimizer.zero_grad()
            
            # Split sequence into chunks for TBPTT
            chunk_inputs = torch.split(inputs, tbptt_chunk, dim=1)
            chunk_targets = torch.split(targets, tbptt_chunk, dim=1)
            
            chunk_loss = 0
            for t in range(len(chunk_inputs)):
                # Forward pass on chunk
                output, state = model(chunk_inputs[t], state)
                
                # Calculate loss
                output = output.reshape(-1, len(word_to_int))
                chunk_targets_flat = chunk_targets[t].reshape(-1)
                loss = criterion(output, chunk_targets_flat)
                chunk_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Detach state for next chunk
                state = tuple(s.detach() for s in state)
            
            epoch_loss += chunk_loss
            num_chunks += len(chunk_inputs)
        
        # Calculate metrics
        avg_loss = epoch_loss / num_chunks
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        tbptt_losses.append(avg_loss)
        tbptt_perplexities.append(perplexity.item())
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Perplexity: {perplexity:.4f}")
        
        # Generate sample text at different training stages
        if epoch == 0 or epoch == num_epochs//2 or epoch == num_epochs-1:
            print(f"\n--- Generated Text at Epoch {epoch+1} ---")
            prompt = ["the", "president", "wants"]
            generated = sample(prompt, model, random_sample_next)
            print(f"Generated: {' '.join(generated)}")
        
    # Plot TBPTT training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(tbptt_losses)
    plt.title('TBPTT Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.axhline(y=1.5, color='r', linestyle='--', label='Target Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(tbptt_perplexities)
    plt.title('TBPTT Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.tight_layout()
    plt.savefig('tbptt_metrics.png')
    plt.show()
    
    '''
    Evaluation, part 2
    '''
    print("\nEvaluation Part 2:")
    print("Generating sentences with different strategies...")
    
    # Set prompt
    prompt = ["the", "president", "wants"]
    print(f"\nStarting prompt: '{' '.join(prompt)}'")
    
    print("\nSampling strategy generations:")
    for i in range(3):
        generated = sample(prompt, model, random_sample_next)
        print(f"Generation {i+1}: {' '.join(generated)}")
        
    print("\nGreedy strategy generations:")
    for i in range(3):
        generated = sample(prompt, model, sample_argmax)
        print(f"Generation {i+1}: {' '.join(generated)}")
    
    '''
    Bonus question
    '''
    # Perform the vector arithmetic: king - man + woman
    result_vector = model.embedding.weight[word_to_int['king']].detach() - model.embedding.weight[word_to_int['man']].detach() + model.embedding.weight[word_to_int['woman']].detach()
    
    # Calculate cosine similarity with all word vectors
    similarities = torch.nn.functional.cosine_similarity(result_vector.unsqueeze(0), model.embedding.weight)
    
    # Find the word with the highest similarity
    most_similar_word_idx = torch.argmax(similarities).item()
    most_similar_word = int_to_word[most_similar_word_idx]
    
    print("Most similar word to 'king - man + woman':", most_similar_word)