
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from torch.nn.functional import one_hot
import torch
import numpy as np
import datetime

fake = Faker()
Faker.seed(12345)
random.seed(12345)

# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY','MMMM-d-YYY',
           'MMMM d, YYY',
           'dd.MM.YY'
           'YYYY.MM.DD'
           'YY MM DD'
           'YY DD MM'
           'YY M DD'
           'YYY D M'
           'd-MMMM-YYY', 'YY-M-DD',
           "%d/%m/%Y",
           "%Y-%m-%d",
           "%B %d, %Y",
           "%d %B %Y",
           "%b %d %Y",
           "%A, %d %B %Y",
           "%d-%m-%y",
           "%m/%d/%y",
           "%d.%m.%Y",
           "%Y.%m.%d",
           "%Y/%m/%d",
           "%d %b, %Y",
           "%a %b %d %Y",
           "%d %b %y",
           "%m-%d-%Y",
           "%d/%b/%Y"
           ]

# change this if you want it to work with another language
LOCALES = ['en_US']

def load_date():
    """
        Loads some fake dates 
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_between_dates(date_start=datetime.date(1800, 1, 1),
                               date_end=datetime.date(2099, 12, 31))

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US') # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
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

