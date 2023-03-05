import numpy as np
import re
import math
import random
from collections import defaultdict

# Read corpus
def read_corpus(fname):
    print("Reading corpus {}...".format(fname))
    texts = []
    with open(fname, 'r') as corpus:
        for line in corpus:
            # Clean text
            line = line.lower().strip()
            line = re.sub(r'[^\w\s,.?!"]+', '', line)
            texts.append(line.split())
            
    return texts

# Get ngrams of sequence (list of words)
def get_ngrams(sequence, n):
    # Modify sequence to include START/STOP
    if n <= 2:
        sequence = ['START'] + sequence
    else:
        sequence = ['START'] * (n-1) + sequence

    sequence += ['STOP']

    n_grams = []

    for i in range(0, len(sequence) - n + 1):
        n_gram_tup = tuple(sequence[i:i+n])
        n_grams.append(n_gram_tup)

    return n_grams

# Get counts for each n-gram in corpus
def get_ngram_counts(corpus, n):
    ngram_counts = defaultdict(int)
    for seq in corpus:
        n_grams = get_ngrams(seq, n)
        for ng in n_grams:
            ngram_counts[ng] += 1
    
    return ngram_counts

# Get vocabulary of corpus
def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  

# Partition training and test data
def train_test_split(X, y, train_size=.9):
    num_train_instances = int(len(X) * train_size)
    random.shuffle(X)

    X_train = X[:num_train_instances]
    y_train = y[:num_train_instances]

    X_test = X[num_train_instances:]
    y_test = y[num_train_instances:]

    return X_train, X_test, y_train, y_test

# ngram classifier class
class NGramClassificationModel():
    # Pass filenames for human and gpt corpora
    def __init__(self, hum_corpus_file, gpt_corpus_file):
        self.hum_corpus = read_corpus(hum_corpus_file)
        self.gpt_corpus = read_corpus(gpt_corpus_file)

        self.split_data()

    # Create train and test sets
    def split_data(self, p=.9):
        self.x_train_hum, self.x_test_hum, self.y_train_hum, self.y_test_hum = train_test_split(self.hum_corpus, [0] * len(self.hum_corpus), train_size=p)
        self.x_train_gpt, self.x_test_gpt, self.y_train_gpt, self.y_test_gpt = train_test_split(self.gpt_corpus, [1] * len(self.gpt_corpus), train_size=p)

        print("Computing ngrams...")
        self.x_train_trigrams, self.x_test_trigrams = get_ngram_counts(self.x_train_gpt + self.x_train_hum, 3), get_ngram_counts(self.x_test_gpt + self.x_test_hum, 3)
        self.x_train_bigrams, self.x_test_bigrams = get_ngram_counts(self.x_train_gpt + self.x_train_hum, 2), get_ngram_counts(self.x_test_gpt + self.x_test_hum, 2)

    # Train ngram models by calculating unigrams, bigrams, and trigrams in training data
    # Set lexicon size and class priors
    def train_ngrams(self):
        print("Training ngrams...")
        # Unigram counts for train 
        self.x_train_hum_unigrams = get_ngram_counts(self.x_train_hum, 1)
        self.x_train_gpt_unigrams = get_ngram_counts(self.x_train_gpt, 1)

        # Bigram counts for train
        self.x_train_hum_bigrams = get_ngram_counts(self.x_train_hum, 2)
        self.x_train_gpt_bigrams = get_ngram_counts(self.x_train_gpt, 2)

        # Trigram counts for train
        self.x_train_hum_trigrams = get_ngram_counts(self.x_train_hum, 3)
        self.x_train_gpt_trigrams = get_ngram_counts(self.x_train_gpt, 3)

        # Get lexicon size |V|
        self.full_lexicon_size = len(get_lexicon(self.hum_corpus).union(get_lexicon(self.gpt_corpus)))

        # Set class priors
        self.class_prior_hum = len(self.y_train_hum) / (len(self.y_train_hum) + len(self.y_train_gpt))
        self.class_prior_gpt = len(self.y_train_gpt) / (len(self.y_train_hum) + len(self.y_train_gpt))
        

    # Input -> seq: list of words, n: 2 or 3 for trigram or bigram model
    # Return log probabilities that sequence belongs to human or gpt corpora, respectively
    def class_sequence_prob(self, seq, n):
        conditional_prior_hum = 0
        conditional_prior_gpt = 0

        ngrams = get_ngrams(seq, n)
        
        for ng in ngrams:
            # Compute numerators and denominators for conditional priors
            # Use Laplacian smoothing
            if n == 2:
                num_hum = self.x_train_hum_bigrams[ng] + 1
                num_gpt = self.x_train_gpt_bigrams[ng] + 1

                den_hum = self.x_train_hum_unigrams[ng[:1]] + self.full_lexicon_size
                den_gpt = self.x_train_gpt_unigrams[ng[:1]] + self.full_lexicon_size
            elif n == 3:
                num_hum = self.x_train_hum_trigrams[ng] + 1
                num_gpt = self.x_train_gpt_trigrams[ng] + 1

                den_hum = self.x_train_hum_bigrams[ng[:2]] + self.full_lexicon_size
                den_gpt = self.x_train_gpt_bigrams[ng[:2]] + self.full_lexicon_size

            # Use summation of log probs for product of probabilities
            conditional_prior_hum += math.log2(num_hum / den_hum)
            conditional_prior_gpt += math.log2(num_gpt / den_gpt)

        # Log probabilities of class prior * conditional prior (arg max of P(y) * P(w_{1:n} | y))
        # Denominator is the same, so can disregard
        hum_prob = math.log2(self.class_prior_hum) + conditional_prior_hum
        gpt_prob = math.log2(self.class_prior_gpt) + conditional_prior_gpt
        
        return hum_prob, gpt_prob
    
    # Evaluate model on test set using n-grams
    # n: 2 for bigrams, 3 for trigrams
    def evaluate_model(self, n):
        correct = 0
        total = 0

        # Evaluate each test set
        for x in self.x_test_hum:
            hum_prob, gpt_prob = self.class_sequence_prob(x, n)

            if hum_prob > gpt_prob:
                correct += 1
            
            total += 1

        for x in self.x_test_gpt:
            hum_prob, gpt_prob = self.class_sequence_prob(x, n)
            
            if gpt_prob > hum_prob:
                correct += 1
            
            total += 1

        return correct / total
    
    # Calculate OOV Rate for n-grams using train and test sets
    def oov_rate(self, n):
        not_in_train = 0

        train_ngrams = self.x_train_trigrams if n == 3 else self.x_train_bigrams
        test_ngrams = self.x_test_trigrams if n == 3 else self.x_test_bigrams

        total_ngrams = 0

        for n in test_ngrams.keys():
            if n not in train_ngrams.keys():
                # Add number of times bigram in test set unseen in train set appears
                not_in_train += test_ngrams[n]

            total_ngrams += test_ngrams[n]

        oov_rate = not_in_train / total_ngrams

        #print(not_in_train, total_ngrams)
        #print("OOV Rate: {}".format(oov_rate))
        return oov_rate

# Text generation class
class TextGenerator():
    def __init__(self, corpus_file):
        self.corpus = read_corpus(corpus_file)
        self.corpus_bigrams = get_ngram_counts(self.corpus, 2)
        self.corpus_trigrams = get_ngram_counts(self.corpus, 3)
    
    # Generate sentences using random probability weighted sampling of candidate words
    def generate_sentence(self, n, sentence_len=20, T=50):
        print("Generating sentence...")

        # Define ngrams
        ngrams = self.corpus_trigrams if n == 3 else self.corpus_bigrams

        # Add START tokens
        result = ["START", "START"] if n == 3 else ["START"]

        # Sentence generating loop
        counter = 0
        upper_bound = sentence_len + n
        while counter < upper_bound:
            # Possible token candidates
            next_token_candidates = [c for c in ngrams.keys() if ngrams[c] > 0 and c[:n-1] == tuple(result[counter:counter+n-1])]

            # Reset
            if len(next_token_candidates) == 0:
                print("No token candidates: regenerating...")

                counter = 0
                result = ["START"] if n == 2 else ["START", "START"]
                continue

            next_token = ''

            # Randomly choose first token
            if counter == 0:
                next_token = random.choice(next_token_candidates)[-1]
            else:
                # Choose the next tokens with highest probability using weighted random sampling
                try:
                    prob_sum = sum([math.exp(ngrams[tuple(result[counter:counter+n-1] + [next_token[-1]])] / T) for next_token in next_token_candidates])
                    
                    # Calculate probabilities for tokens using exp and T
                    # Normalize to [0, 1] if possible (min-max scaling)
                    next_token_candidates_prob = np.array([math.exp(ngrams[tuple(result[counter:counter+n-1] + [x[-1]])] / T) / prob_sum for x in next_token_candidates])
                    next_token_candidates_prob = (next_token_candidates_prob - next_token_candidates_prob.min()) / (next_token_candidates_prob.max() - next_token_candidates_prob.min())

                    next_token = random.choices(next_token_candidates, next_token_candidates_prob, k=1)[0][-1]
                except Exception:
                    # Reset
                    print("Invalid probabilities: regenerating...")

                    counter = 0
                    result = ["START"] if n == 2 else ["START", "START"]
                    continue
            
            # Add START token if STOP token appears to generate new phrase
            if next_token == "STOP":
                result.append("START")
                counter += 1
                upper_bound += 1

                continue
                
            result.append(next_token)
            counter += 1

        #print(' '.join(result[n-1:]))
        #print()
        return ' '.join([t for t in result[n-1:] if t != "START"])

def run_classifier():
    ngram_classifier = NGramClassificationModel('humvgpt/hum.txt', 'humvgpt/gpt.txt')
    ngram_classifier.train_ngrams()
    
    print("Bigram accuracy: {}".format(ngram_classifier.evaluate_model(2)))
    print("Trigram accuracy: {}".format(ngram_classifier.evaluate_model(3)))
    
    print("OOV Rate (Bigram): {}".format(ngram_classifier.oov_rate(2)))
    print("OOV Rate (Trigram): {}".format(ngram_classifier.oov_rate(3)))

    print()

def run_sentence_generator():
    gpt_sentence_generator = TextGenerator('humvgpt/gpt.txt')
    hum_sentence_generator = TextGenerator('humvgpt/hum.txt')

    print("Sentence Generation (GPT, Bigrams, T=50):\n {}".format(gpt_sentence_generator.generate_sentence(2, T=50)))
    print("Sentence Generation (Hum, Bigrams, T=50):\n {}".format(hum_sentence_generator.generate_sentence(2, T=50)))

    print("Sentence Generation (GPT, Trigrams, T=50):\n {}".format(gpt_sentence_generator.generate_sentence(3, T=50)))
    print("Sentence Generation (Hum, Trigrams, T=50):\n {}".format(hum_sentence_generator.generate_sentence(3, T=50)))

    print("Sentence Generation (GPT, Bigrams, T=50):\n {}".format(gpt_sentence_generator.generate_sentence(2, T=50)))
    
    print("Sentence Generation (GPT, Bigrams, T=500):\n {}".format(gpt_sentence_generator.generate_sentence(2, T=500)))
    print("Sentence Generation (Hum, Bigrams, T=500):\n {}".format(hum_sentence_generator.generate_sentence(2, T=500)))

    print("Sentence Generation (GPT, Trigrams, T=500):\n {}".format(gpt_sentence_generator.generate_sentence(3, T=500)))
    print("Sentence Generation (Hum, Trigrams, T=500):\n {}".format(hum_sentence_generator.generate_sentence(3, T=500)))

    print("Sentence Generation (GPT, Bigrams, T=1000):\n {}".format(gpt_sentence_generator.generate_sentence(2, T=1000)))
    print("Sentence Generation (Hum, Bigrams, T=1000):\n {}".format(hum_sentence_generator.generate_sentence(2, T=1000)))

    print("Sentence Generation (GPT, Trigrams, T=1000):\n {}".format(gpt_sentence_generator.generate_sentence(3, T=1000)))
    print("Sentence Generation (Hum, Trigrams, T=1000):\n {}".format(hum_sentence_generator.generate_sentence(3, T=1000)))

    #print("Sentence Generation (GPT, Bigrams, T=20):\n {}".format(gpt_sentence_generator.generate_sentence(2, T=20)))
    #print("Sentence Generation (Hum, Bigrams, T=20):\n {}".format(hum_sentence_generator.generate_sentence(2, T=20)))

    #print("Sentence Generation (GPT, Trigrams, T=20):\n {}".format(gpt_sentence_generator.generate_sentence(3, T=20)))
    #print("Sentence Generation (Hum, Trigrams, T=20):\n {}".format(hum_sentence_generator.generate_sentence(3, T=20)))

    print()
    

if __name__ == "__main__":
    run_classifier()
    run_sentence_generator()

    