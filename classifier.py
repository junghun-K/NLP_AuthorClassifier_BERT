import numpy as np
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE
# Preprocessing text
from nltk.tokenize import RegexpTokenizer
# Tested with different smoothing
from nltk.lm import KneserNeyInterpolated, Laplace, StupidBackoff
from sklearn.model_selection import train_test_split
import sys

# Bonus Point
def ngrams(sentence, n):
    ngrams = []
    for i in range(len(sentence) - n + 1):
        ngrams.append(np.array([sentence[i+j] for j in range(n)]))
    return ngrams

### Preprocessing data using RegexpTokenizer: picks out sequences of alphanumeric characters as tokens and drops everything else
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = list(map(lambda s: tokenizer.tokenize(s), text))
    text = [ele for ele in text if ele != []]   # remove empty list

    return text

def test_mapping(element):
    test = list(pad_both_ends(element, n=2))
    test = ngrams(test, n=2)
    return test

# print(sys.argv)

# Get input from terminal
terminal_input = sys.argv
n = len(terminal_input)
authors_file = terminal_input[1]

if ('.txt' in authors_file):
    pass
else:
    authors_file = authors_file + '.txt'


testmode = False

# testfile = 
if n == 4:
    testfile = terminal_input[3]
    testmode = True

    if ('.txt' in testfile):
        pass
    else:
        testfile = testfile + '.txt'



with open(authors_file) as f:
    authorlist = f.read().replace('.txt', '').splitlines()

# Open and read files
texts = {}
for author in authorlist:
    with open(f'./ngram_authorship_train/{author}.txt') as f:
        texts[author] = preprocess_text(f.readlines())


# Normal mode
n = 2  # ngram order
dev_sets = {}
train_sets = {}

# test mode
if (testmode):
    # Get testfile
    for author in authorlist:
        train_sets[author] = texts[author]
    try:
        with open(f'./ngram_authorship_train/{testfile}') as f:
            dev_set = preprocess_text(f.readlines())
    except:
        with open(f'{testfile}') as f:
            dev_set = preprocess_text(f.readlines())
        
    # Train
    models = {}
    print('training LMs... (this may take a while)')
    for author in authorlist:
        train, vocab = padded_everygram_pipeline(n, train_sets[author])
        models[author] = StupidBackoff(order=n)
        models[author].fit(train, vocab)

    # Compute perplexity
    test_set = list(map(test_mapping, dev_set))
    inf_counts = {author: 0 for author in authorlist}
    INF = float('inf')
    for item in test_set:
        min_perplexity = models[authorlist[0]].perplexity(item)
        choice = authorlist[0]
        for model in authorlist[1:]:
            perplexity = models[model].perplexity(item)
            if perplexity < min_perplexity:
                min_perplexity = perplexity
                choice = model

        if perplexity == INF:
            inf_counts[author] += 1
        print(choice)
# Testmode false
else:
    print('splitting into training and development...')
    for author in authorlist:
        train_sets[author], dev_sets[author] = train_test_split(texts[author], test_size=0.1, random_state=50)

    # Train
    models = {}
    print('training LMs... (this may take a while)')
    for author in authorlist:
        train, vocab = padded_everygram_pipeline(n, train_sets[author])
        models[author] = StupidBackoff(order=n)
        models[author].fit(train, vocab)

    correct_counts = {author: 0 for author in authorlist}
    inf_counts = {author: 0 for author in authorlist}
    INF = float('inf')
    perplexity_min = {author: INF for author in authorlist}
    min_sentences = {author: [] for author in authorlist}
    for author in authorlist:
        test_set = list(map(test_mapping, dev_sets[author]))

        for item in test_set:
           
            min_perplexity = models[authorlist[0]].perplexity(item)
            choice = authorlist[0]
            for model in authorlist[1:]:
                perplexity = models[model].perplexity(item)
                if perplexity < min_perplexity:
                    min_perplexity = perplexity
                    choice = model
            if perplexity == float('inf'):
                inf_counts[author] += 1
            elif choice == author:
                correct_counts[author] += 1
                if (min_perplexity < perplexity_min[author]):
                    perplexity_min[author] = min_perplexity
                    min_sentences[author] = item
           


    percentages = {}
    for author in authorlist:
        percentages[author] = round(correct_counts[author] / (len(dev_sets[author])-inf_counts[author]) * 100, 1)

    print('Results on dev set:')
    for author in authorlist:
        print(f'{author : <10} {percentages[author]}%')

    print('\nMin perplexity of dev set of each author (bonus point):')
    for author in authorlist:
        words = [b[1] for b in min_sentences[author]][:-1]
        print(f'{author : <10} {round(perplexity_min[author], 2)} with {words}')