# 🎭 Shakespeare Next Word Predictor

Ever wondered what word Shakespeare might use next? This project teaches a computer to read Shakespeare's *Hamlet* and then **predict the next word** in any sentence — just like how your phone keyboard suggests the next word when you type!

---

## 🤔 What Does This Project Do?

You give the model a sentence like:

> *"who's there and for it"*

And the model predicts what word comes next — for example: **"that's"**

The model has read all of *Hamlet* and learned the patterns of how words follow each other. So it can make smart guesses about what word should come next.

---

## 🧠 The Big Picture — How It Works (Step by Step)

### Step 1 — Get the Text Data
We use Shakespeare's *Hamlet* as our training data. It's freely available through a Python library called **NLTK** (Natural Language Toolkit), which has a collection of classic books called the Gutenberg corpus.

```
Hamlet text → "To be or not to be, that is the question..."
```

### Step 2 — Clean the Text
We convert all the text to **lowercase** so the model doesn't treat "The" and "the" as two different words.

```
"To Be Or Not" → "to be or not"
```

### Step 3 — Tokenization (Convert Words to Numbers)
Computers don't understand words — they only understand numbers. So we use a **Tokenizer** to give every unique word a number.

```
"hamlet" → 5
"king"   → 12
"love"   → 34
```

The most common words get small numbers, and rare words get large numbers.

### Step 4 — Create Training Sequences (N-grams)
We slide through the text and create lots of mini-sequences. Each sequence teaches the model: *"given these words, what comes next?"*

For example, from the sentence `"to be or not to be"`:
```
[to, be]              → or
[to, be, or]          → not
[to, be, or, not]     → to
[to, be, or, not, to] → be
```

This gives us thousands of examples for the model to learn from.

### Step 5 — Padding (Make All Sequences the Same Length)
All our sequences are different lengths, but the model needs them to be the same size. So we add zeros at the beginning of shorter sequences to make them equal length. This is called **padding**.

```
[0, 0, 0, 5, 12] ← short sequence padded with zeros
[3, 7, 2, 5, 12] ← already full length
```

### Step 6 — Split into Training and Testing Sets
We split the data:
- **80% for training** — the model learns from this
- **20% for testing** — we check how well the model learned on unseen data

### Step 7 — Build and Train the LSTM Model
We build a neural network using **LSTM** (Long Short-Term Memory) layers. LSTM is specially designed to understand sequences — it can remember what came earlier in a sentence to make better predictions.

Our model has these layers in order:

| Layer | What It Does |
|-------|-------------|
| **Embedding** | Converts word numbers into rich 100-number vectors that capture meaning |
| **LSTM (150 units)** | Reads the sequence and remembers important patterns |
| **Dropout (20%)** | Randomly ignores some neurons during training to prevent over-memorizing |
| **LSTM (100 units)** | Does another round of pattern learning |
| **Dense + Softmax** | Outputs a probability for every word in the vocabulary |

Think of the output as: *"There's a 40% chance the next word is 'king', 30% chance it's 'love', 10% chance it's 'death'..."* — and we pick based on those chances.

### Step 8 — Predict the Next Word
You give the model a sentence, it tokenizes and pads it, runs it through the trained network, and picks the next word based on the probability output.

---

## 🗂️ Project Structure

```
shakespeare-next-word-predictor/
│
├── notebook.ipynb    ← All the code is here (Jupyter Notebook)
├── data.txt          ← Hamlet text saved as a file (auto-created when you run)
└── README.md         ← This file!
```

---

## 🛠️ What You Need to Install

Make sure you have Python installed, then install these libraries:

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn nltk
```

Then, inside Python (or in the notebook), download the Hamlet text:

```python
import nltk
nltk.download('gutenberg')
```

---

## ▶️ How to Run the Project

1. **Clone or download this repository:**
   ```bash
   git clone https://github.com/your-username/shakespeare-next-word-predictor.git
   cd shakespeare-next-word-predictor
   ```

2. **Open the notebook:**
   ```bash
   jupyter notebook notebook.ipynb
   ```
   Or simply open it in **Google Colab** (recommended — it's free and has GPU support).

3. **Run all the cells from top to bottom.** The model will train automatically. Training might take a few minutes.

4. **Try your own sentence!** Find this cell near the bottom:
   ```python
   text = "who's there and for it"
   ```
   Change it to any phrase from Hamlet and run the cell to see the prediction.

---

## ⚙️ Training Settings

| Setting | Value | What It Means |
|---------|-------|---------------|
| Epochs | Up to 100 | Max number of times the model reads all the data |
| Early Stopping | Patience = 5 | Training stops automatically if the model stops improving for 5 rounds |
| Validation Split | 20% | 20% of data is kept aside to test the model |
| Embedding Size | 100 | Each word is represented as a list of 100 numbers |
| LSTM Units | 150 → 100 | Size of the memory in each LSTM layer |
| Dropout | 20% | Randomly disables 20% of neurons to reduce overfitting |

---

## 📊 Libraries Used

| Library | Purpose |
|---------|---------|
| **NLTK** | Download and access Shakespeare's Hamlet text |
| **TensorFlow / Keras** | Build and train the LSTM neural network |
| **NumPy** | Handle arrays and numerical operations |
| **Pandas** | Data handling |
| **Matplotlib / Seaborn** | Visualizations (loss/accuracy plots) |
| **Scikit-learn** | Split data into train and test sets |

---

## 💡 Key Concepts Explained Simply

**What is LSTM?**
LSTM stands for *Long Short-Term Memory*. It's a type of neural network that's great at understanding sequences — like sentences — because it can remember earlier words while reading later ones. Normal networks forget quickly; LSTM has a built-in "memory".

**What is an Embedding?**
Instead of using a plain number like `5` to represent a word, an embedding converts it into a list of 100 numbers like `[0.3, -0.1, 0.8, ...]`. These numbers capture *meaning* — similar words end up with similar lists of numbers.

**What is Dropout?**
During training, dropout randomly "switches off" 20% of the neurons. This forces the model to not rely too heavily on any one path, making it generalize better to new sentences.

**What is Softmax?**
Softmax is the final step. It takes the model's raw outputs and converts them into probabilities that all add up to 100%. So the model says: *"I'm 40% sure the next word is X, 30% sure it's Y..."*

---

## 🔮 Example Output

```python
# Input
text = "who's there and for it"

# Output
"that's"
```

The output changes slightly each run because we **sample** from the probability distribution instead of always picking the single highest probability word. This makes the predictions feel more natural and less repetitive.

---

## 🚧 Limitations

- The model only predicts **one word at a time** (not full sentences).
- It's trained only on *Hamlet*, so it works best with Shakespearean-style phrases.
- The predictions can sometimes seem random — that's normal for small language models.

---

## 🚀 Ideas to Improve This Project

- **Generate full sentences** by feeding the predicted word back in as input, repeatedly.
- **Train on more Shakespeare plays** for a richer vocabulary and better predictions.
- **Try a Transformer model** (like a mini GPT) instead of LSTM for better results.
- **Build a simple web app** where users can type a phrase and see predictions live.

---

## 📜 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Shakespeare's *Hamlet* text from the [NLTK Gutenberg Corpus](https://www.nltk.org/book/ch02.html)
- Model built using [TensorFlow / Keras](https://www.tensorflow.org/)
- Trained on Google Colab with a T4 GPU
