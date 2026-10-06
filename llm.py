import torch, torch.nn as nn, torch.nn.functional as F

# use of GPU if available, otherwise CPU
dev = "cuda" if torch.cuda.is_available() else "cpu"

# read input text and create character-level vocabulary
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Build a vocabulary of unique characters in the text
chars = sorted(set(text))

# stoi: Create a mapping from characters to indices
# itos: Create a mapping from indices to characters
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

# Convert the entire text into a tensor of indices using the stoi mapping
data = torch.tensor([stoi[c] for c in text])

# Define the vocabulary size and model hyperparameters
V = len(chars)

# block: The length of the input sequence for the model
# batch: The number of sequences processed in parallel
# d: The dimensionality of the model's hidden representations
# heads: The number of attention heads in the multi-head attention mechanism
# layers: The number of transformer blocks in the model
block, batch, d, heads, layers = 128, 64, 256, 4, 4

def get_batch():

    # Randomly select starting indices for the sequences in the batch
    ix = torch.randint(len(data) - block - 1, (batch,))

    # Create input (x) and target (y) tensors by slicing the data tensor
    x = torch.stack([data[i:i + block] for i in ix])

    # Create target (y) tensor by slicing the data tensor, shifted by one position
    y = torch.stack([data[i + 1:i + block + 1] for i in ix])

    # Move the input and target tensors to the appropriate device (GPU or CPU)
    return x.to(dev), y.to(dev)

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer normalization stabilizes the values passed through the network.
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        # Multi-head self-attention lets characters use earlier characters
        # in the sequence to help predict the next character.
        self.attn = nn.MultiheadAttention(
            d, heads, batch_first=True
        )

        # Feed-forward network that processes each position's representation.
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),  # Expand the hidden dimension.
            nn.GELU(),            # Apply a nonlinear activation function.
            nn.Linear(4 * d, d)   # Return to the original dimension.
        )
        
    def forward(self, x):
        """Pass a batch of character representations through the block."""

        T = x.size(1)  # Sequence length.

        # Create a causal mask. True entries block attention to future tokens.
        # This prevents the model from seeing the answer while predicting it.
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1
        )

        # Normalize the input before attention.
        h = self.ln1(x)

        # Attention output is added to the original input (residual connection).
        x = x + self.attn(
            h, h, h, attn_mask=mask, need_weights=False
        )[0]

        # Apply the feed-forward network and another residual connection.
        return x + self.mlp(self.ln2(x))                                      # residual

class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        # Convert character IDs into learned vectors.
        self.tok = nn.Embedding(V, d)

        # Give each position in the sequence its own learned vector.
        self.pos = nn.Embedding(block, d)

        # Stack multiple Transformer blocks.
        self.blocks = nn.Sequential(
            *[Block() for _ in range(layers)]
        )

        # Normalize the final representations and predict vocabulary scores.
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)


    def forward(self, idx, targets=None):
        """Predict the next character at every position."""

        # idx has shape (batch size, sequence length).
        # Add character embeddings and positional embeddings together.
        positions = torch.arange(idx.size(1), device=idx.device)
        x = self.tok(idx) + self.pos(positions)

        # Process the sequence through the Transformer blocks.
        x = self.blocks(x)

        # Produce a score (logit) for every possible character.
        logits = self.head(self.ln(x))

        loss = None

        # During training, compare predictions with the correct next characters.
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                targets.reshape(-1)
            )

        # Return predictions and, when training, the loss.
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, n):
        """Generate n new characters, one character at a time."""

        for _ in range(n):
            # Keep only the most recent block characters as context.
            logits, _ = self(idx[:, -block:])

            # Convert the last position's scores into probabilities.
            probs = F.softmax(logits[:, -1], dim=-1)

            # Randomly sample the next character according to its probabilities.
            next_char = torch.multinomial(probs, num_samples=1)

            # Append the new character to the sequence.
            idx = torch.cat([idx, next_char], dim=1)

        return idx

# Create the model and move its parameters to the CPU or GPU.
model = GPT().to(dev)

# AdamW adjusts model parameters to reduce prediction error.
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Train the model for 3,000 steps.
for step in range(3000):
    x, y = get_batch()

    # Calculate predictions and the training loss.
    _, loss = model(x, y)

    # Clear old gradients, calculate new gradients, and update parameters.
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Print the loss every 200 steps to monitor training.
    if step % 200 == 0:
        print(step, loss.item())

# Start generation with character ID 0 and generate 500 new characters.
# Note: ID 0 is the first character in the sorted vocabulary, not necessarily
# a special start-of-text token.
start = torch.zeros((1, 1), dtype=torch.long, device=dev)
out = model.generate(start, 500)

# Convert generated character IDs back into readable text.
print("".join(itos[i] for i in out[0].tolist()))