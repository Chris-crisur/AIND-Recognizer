import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # for each test word's X and lengths, see how this scores on the training set's model of the word.
    # higher values for the score (Log Likelihood) indicate a closer match to a word
    for testing_word, (X, lengths) in test_set.get_all_Xlengths().items():
      probs = {}
      best_score = except_score = float("-Inf")
      best_word = None
      for trained_word, model in models.items():
        try:
          score = model.score(X,lengths)
        except:
          score = except_score
        probs[trained_word] = score
        if score>best_score:
          best_score = score
          best_word = trained_word

      # append (copied) dict to probabilities list.
      probabilities.append(probs.copy())
      # append best guess
      guesses.append(best_word)
    return probabilities, guesses
