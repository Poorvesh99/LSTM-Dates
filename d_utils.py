
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from torch.nn.functional import one_hot
import torch
import numpy as np
import datetime
import re

fake = Faker()
Faker.seed(12345)
random.seed(12345)

# Define format of the data we would like to generate
BASE_FORMATS = [
    # Common digit-based formats
    "dd/MM/yyyy", "MM/dd/yyyy", "yyyy-MM-dd", "dd-MM-yyyy", "yyyy/MM/dd",
    "dd.MM.yyyy", "MM.dd.yyyy", "dd MM yyyy",

    # With 2-digit years
    "dd/MM/yy", "MM-dd-yy", "yy-MM-dd", "dd.MM.yy", "yy/MM/dd",

    # Long and medium textual formats
    "d MMM yyyy", "d MMMM yyyy", "dd MMM yyyy",
    "d MMM, yyyy", "d MMMM, yyyy", "MMMM d, yyyy",
    "d-MMMM-yyyy", "d/MMM/yyyy", "dd/MMM/yyyy",

    # Abbreviated and short forms
    "MMM d, yyyy", "yyyy MMM d", "MMM dd yy",
    "EEE, d MMM yyyy", "EEEE, d MMMM yyyy",  # day names
    "d MMM yy", "dd MMM yy", "yy MMM d",

    # ISO variations and dots/slashes
    "yyyy.MM.dd", "yyyy/MM/dd", "yyyy MM dd",

    # Minimalist or compact styles
    "d M yyyy", "yy M d", "yy/MM/dd", "M-d-yy", "yyMMdd",

    # Reversed and regional variants
    "yyyy-dd-MM", "MMM d yyyy", "d-MM-yy", "MMMM d yyyy",

    # Include Faker's built-in
    "short", "medium", "long", "full"
]

EXTRA_FORMATS = [
    # Month–Day–Year (numeric)
    "MM/dd/yyyy", "MM-dd-yyyy", "MM.dd.yyyy", "MM dd yyyy",
    "MMM d, yyyy", "MMMM d, yyyy",              # e.g. "Apr 5, 2009", "April 5, 2009"
    "MM/dd/yy", "MM-dd-yy",                     # 2-digit year

    # Day–Year–Month (month last)
    "d yyyy MMM",   "dd yyyy MMMM",             # e.g. "5 2009 Apr", "05 2009 April"
    "d yy MMM",     "dd yy MMMM",               # 2-digit year variants
    "d, yyyy MMMM", "dd, yyyy MMM",             # with comma

    # Year–Month–Day (ISO & variations)
    "yyyy-MM-dd", "yyyy/MM/dd", "yyyy.MM.dd",
    "yyyy MMM d",  "yyyy MMMM dd",              # e.g. "2009 Apr 05", "2009 April 05"

    # Middle-month formats (for completeness)
    "d MMM yyyy", "dd MMMM, yyyy", "d MMM, yyyy",
    "dd.MM.yyyy", "d-MMMM-yyyy",

    # Day-name variants
    "EEE, MMM d, yyyy", "EEEE, MMMM d yyyy",

    # Built-in locale levels
    "short", "medium", "long", "full"
]

ALL_FORMATS = list(dict.fromkeys(BASE_FORMATS + EXTRA_FORMATS))

FORMATS = ALL_FORMATS

# change this if you want it to work with another language
LOCALES = ['en_US']


NOISE_PROB = 0.2   # 20% chance to apply each noise type

def add_noise(s: str) -> str:
    # 1) Randomly flip case on some letters
    def flip_case(c):
        if random.random() < NOISE_PROB:
            return c.upper() if c.islower() else c.lower()
        return c
    s = ''.join(flip_case(c) for c in s)

    # 2) Randomly duplicate or drop spaces
    if random.random() < NOISE_PROB:
        s = re.sub(r' ', lambda m: ' ' * random.choice([1,2,3]) if random.random()<0.5 else '', s)

    # 3) Swap separators (/, -, .) at random
    if random.random() < NOISE_PROB:
        sep_choices = ['/', '-', '.', ' ']
        s = re.sub(r'[/\-.]', lambda m: random.choice(sep_choices), s)

    # 4) Inject occasional character typo (swap adjacent chars)
    if random.random() < NOISE_PROB:
        idx = random.randrange(len(s)-1)
        s = s[:idx] + s[idx+1] + s[idx] + s[idx+2:]

    return s

def load_date():
    """
        Loads some fake dates 
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_between_dates(date_start=datetime.date(1900, 1, 1),
                               date_end=datetime.date(2050, 12, 31))

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US') # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
        human_readable = add_noise(human_readable)
        machine_readable = dt.isoformat()
        
    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt

def load_dataset(m):
    """
        Loads a dataset with m examples and vocabularies
        :m: the number of examples to generate
    """
    
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    Tx = 30
    

    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))
    
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], 
                     list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v:k for k,v in inv_machine.items()}
 
    return dataset, human, machine, inv_machine





def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    X, Y = zip(*dataset)

    X = torch.tensor([string_to_int(i, Tx, human_vocab) for i in X])
    Y = torch.tensor([string_to_int(t, Ty, machine_vocab) for t in Y])

    Xoh = np.array(list(map(lambda x: one_hot(x, num_classes=len(human_vocab)), X))).astype('float64')
    Yoh = np.array(list(map(lambda x: one_hot(x, num_classes=len(machine_vocab)), Y))).astype('float64')

    return X, Y, Xoh, Yoh


def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"

    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"

    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """

    # make lower to standardize
    string = string.lower()
    string = string.replace(',', '')

    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))

    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))

    # print (rep)
    return rep


def int_to_string(ints, inv_vocab):
    
    l = [inv_vocab[i] for i in ints]
    return l

