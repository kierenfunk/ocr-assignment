"""Classification system.

Implementation of a nearest neighbour classifier with PCA dimension reduction.
Error correction using word lookup is also used. 

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg
import pickle
import string

def reduce_dimensions(feature_vectors_full, model):
    """Implementation of PCA dimension reduction. Uses 10 best axes with the largest range.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    pca_data = range(10)
    covx = np.cov(feature_vectors_full, rowvar=0)
    if 'fvectors_train' not in model:
        # if in training stage
        N = covx.shape[0]
        w, v = scipy.linalg.eigh(covx, eigvals=(N - 10, N - 1))
        v = np.fliplr(v)
        pca_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), v)
        # save results for testing stage
        model['v'] = v.tolist()
    else:
        # if in testing stage
        v = np.array(model['v'])
        pca_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), v)
    
    return pca_data[:,:10]


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)
    
    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size
    model_data['unique_ratio'] = []

    infile = open('../data/Extra/markov_pmatrix.pickle','rb')
    model_data['markov_states'] = pickle.load(infile)
    infile.close()

    wordFile = open('../data/Extra/wordlist.txt','r')
    model_data['words'] = [i.strip() for i in wordFile.readlines()]
    wordFile.close()

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


def classify_page(page, model):
    """Nearest neighbour classifier, returns the label of the closest neighbour to the character
    however, 25 nearest neighbours are stored in model for use in the error correction stage

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """

    # a counter varaible for storing k nearest neighbour output from the classify stage
    if 'classify_counter' not in model:
        model['classify_counter'] = 0
    model['classify_counter'] += 1
    
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    
    # implementation of nearest neighbour
    x = np.dot(page, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(page * page, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance
    # closest neighbour
    nearest = np.argmax(dist, axis=1)

    # 25 nearest neighbours
    nearestk = np.fliplr(np.argsort(dist, axis=1))[:,:25]
    # store in model dictionary for error correction stage
    model['nearest-page{}'.format(model['classify_counter'])] = nearestk.tolist()

    # get labels of the closest neighbours
    label = labels_train[nearest]

    # calculate a noise level score for the page
    numUnique = np.sum(np.array([len(np.unique(labels_train[nearestk[i]])) for i in range(len(label))]))
    # store in model for error correction stage
    model['unique_ratio'].append(numUnique/len(label))
    
    return label

def letterToIndex(letter):
    """Converts a character to it's corresponding index for use with the markov state transition matrix
    
    parameters:

    letter - a character
    """
    if ord(letter) > 64 and ord(letter) < 91:
        # if capital letter
        return ord(letter)-65
    if ord(letter) > 96 and ord(letter) < 123:
        # if lower case letter
        return ord(letter)-71
    return -1

def harmonicSeriesNN(nearestK):
    """Implementation of k nearest neighbours using harmonic series weightings.
    (not used in current implementation)
    
    parameters:

    nearestK - the k closest neighbours to the character in question from the classify stage
    """
    # an array with 1, 1/2, 1/3, ..., 1/k
    harmonicSeries = np.ones(len(nearestK))/np.arange(1, len(nearestK)+1, 1)
    # unique letters and counts for each nearestk
    unique, counts = np.unique(nearestK, return_counts=True)
    for i in range(len(unique)):
        # apply harmonic weightings to counts
        counts[i] = counts[i]*np.sum(harmonicSeries[np.where(nearestK==unique[i])[0]])
    return unique[np.argmax(counts)]

def harmonicSeriesWeights(nearestK):
    """Given a set of characters that are closest to a character that needs to be classified, a 
    harmonic weighting scheme is applied based on the proximity of each character. Characters that
    are closer are given a higher weighting. The set of characters and the counts after weighting
    has been applied is returned.
    
    parameters:

    nearestK - the k closest neighbours to the character in question from the classify stage
    """
    # an array with 1, 1/2, 1/3, ..., 1/k
    harmonicSeries = np.ones(len(nearestK))/np.arange(1, len(nearestK)+1, 1)
    # unique letters and counts for each nearestk
    unique, counts = np.unique(nearestK, return_counts=True)
    for i in range(len(unique)):
        # apply harmonic weightings to counts
        counts[i] = counts[i]*np.sum(harmonicSeries[np.where(nearestK==unique[i])[0]])
    return unique,counts

def markovProbs(chars,first_letter,markov_prob,elements,counts):
    """Calculates the probability of a character given the character preceding it using markov chains.
    Combines the probability of character given the character before it, and the characters predicted using k
    nearest neighbours to give a more accurate prediction.
    
    parameters:

    chars - the character to be predicted and the character preceding it
    first_letter - boolean to indicate if the character to be predicted succeeds a space
    markov_prob - a 53x53 matrix storing probabilities of moving from one character to the next
    elements - the characters that have been predicted using k nearest neighbours from the classify stage
    counts - the frequency of each character that has been predicted with k nearest neighbours from the classify stage
    """

    # convert characters to index values for the markov states matrix
    charInds = [letterToIndex(char) for char in chars]
    elementsInds = [letterToIndex(char) for char in elements]

    # find punctuation
    nonLetters = np.array(elementsInds+charInds)
    nonLetters = np.any(nonLetters < 0) or np.any(nonLetters > 51)
    if nonLetters:
        # if predicted to be punctuation, or character preceding is punctuation, return
        return chars[1]
    
    if first_letter:
        # if the first letter in the word, calculate probability given a space comes before it
        charInds[0] = 52

    # get markov 'state change' probabilities
    charProbs = [markov_prob[charInds[0]][i] for i in elementsInds]
    # get k-NN probabilities
    kNN_Probs = np.array(counts)/np.sum(np.array(counts))
    # multiply probabilities
    combined = np.multiply(np.array(charProbs),kNN_Probs)
    # return most probable character
    return elements[np.argmax(combined)]
    

def nextBestWord(word,wordList,levshtein_max=5):
    """Finds the closest word by levenshtein distance and word ranking from a list of english words.
    returns the next best word in a list of characters
    
    parameters:

    word - the word that may need correction
    wordList - a list of dictionary words of various lengths, words with upper and lower case only, no punctuation
    levshtein_max - any word with a levenshtein distance greater than this parameter will not be considered
    """

    # adjust for words with the same length as the query word
    wordListAdj = np.array([list(i) for i in wordList if len(i)==len(word)])
    # make array of the query word
    wordArray = np.array([list(word) for i in range(wordListAdj.shape[0])])
    # calculate levenshtein scores
    scores = wordListAdj!=wordArray

    if scores.ndim < 2:
        # if no more than one word match, return original word query
        return list(word)

    levshtein_current = 2
    numViable = 0
    while numViable < 1 and levshtein_current <= levshtein_max:
        # find viable options in the wordList
        numViable = np.sum(np.sum(scores,axis=1) < levshtein_current)
        levshtein_current+=1
    if numViable < 1:
        # if no options found, return original word query
        return list(word)

    # viable options found, sorted by levenshtein scores
    viable = wordListAdj[np.argsort(np.sum(scores,axis=1))][:numViable]
    # convert from character arrays to strings
    viable = [''.join(i) for i in list(viable)]
    # calculate rankings based on frequency that each word appears in the language
    rankings = np.array([wordList.index(i) for i in viable])
    # return highest ranking word
    return list(viable[np.argsort(rankings)[0]])

def markov_estimation(labels,spaces,nearestk,model):
    """estimate characters based on a markov transition probability matrix  
    
    parameters:

    labels - the output classification label for each feature vector
    spaces - an array indicating where the last letter for each word is
    nearestK - the k closest neighbours from the classify stage
    model - dictionary, stores the output of the training stage    
    """

    markov_prob = np.array(model['markov_states'])

    for i in range(1,len(labels)):
        if model['unique_ratio'][model['correct_counter']-1] < 1.5:
            # if low noise page, use harmonic weightings, and 5 nearest neighbours
            unique, counts = harmonicSeriesWeights(np.array(model['labels_train'])[nearestk[i,:5]])
        else:
            # if high noise page, get unique letters and counts in 25 nearest neighbours
            unique, counts = np.unique(np.array(model['labels_train'])[nearestk[i]], return_counts=True)
        if len(unique) > 1:
            # if more than two characters in nearest neighbours, apply markov probability estimation
            labels[i] = markovProbs(labels[i-1:i+1],spaces[i-1]==1,markov_prob,unique,counts)

    return labels

def word_esimtation(wordLengths,labels,model,levenshteinDist):
    """An implementation of a word lookup algorithm for error correction
    
    parameters:

    wordLengths - an array storing the lengths of each word
    labels - the output classification label for each feature vector
    model - dictionary, stores the output of the training stage
    levenshteinDist - the maximum allowable levenshtein distance between two words to be considered feasible
    """
    labelsIndex = 0
    for i in wordLengths:
        # get the next word
        startInd = labelsIndex
        endInd = labelsIndex+i
        theWord = labels[startInd:endInd]
        if theWord[-1] in string.punctuation:
            # if last character is punctuation
            endInd -= 1
        if theWord[0] in string.punctuation:
            # if first character is punctuation
            startInd += 1
        # convert word to string
        word = ''.join(labels[startInd:endInd])
        # return empty string if word contains punctuation punctuation
        word = ''.join(c for c in labels[startInd:endInd] if c.lower() in string.ascii_lowercase)

        if word not in model['words'] and len(word) > 0 and len(word) == len(labels[startInd:endInd]):
            # estimate new characters
            labels[startInd:endInd]=nextBestWord(word,model['words'],levenshteinDist)
        labelsIndex += i
    return labels


def correct_errors(page, labels, bboxes, model):
    """Implementation of error correction using a combination of markov transition probabilities and a
    word lookup.
    
    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """

    # a counter for finding output from the classify stage in 'model'
    if 'correct_counter' not in model:
        model['correct_counter'] = 0
    model['correct_counter'] += 1

    # k-nearest neighbours from classification stage
    nearestk = np.array(model['nearest-page{}'.format(model['correct_counter'])])

    # identify spaces between characters on the page
    topLeftX = bboxes[:,0]
    topLeftX = np.delete(topLeftX,0) # shift values to the left
    topLeftX = np.append(topLeftX,0) # shift values to the left
    spaces = topLeftX-bboxes[:,2] # get distance of spaces between the characters
    # truth array, true if character is last character in the word
    spaces = np.add(spaces < -20,spaces > 6)

    # use markov transition matrix estimation
    labels = markov_estimation(labels,spaces,nearestk,model)
    
    # estimate characters based on word lookup
    levenshteinDist = 3
    if model['unique_ratio'][model['correct_counter']-1] > 1.5:
        # if a high noise page, increase the allowable levenshtein distance
        levenshteinDist = 5

    # get word lengths
    spaceInds = np.where(spaces==True) # indexes of spaces
    wordLengths = np.diff(spaceInds)[0] # distances between indexes
    wordLengths = np.insert(wordLengths,0,spaceInds[0][0]+1) # insert first word length

    # use word lookup, to correct characters
    labels = word_esimtation(wordLengths,labels,model,levenshteinDist)

    return labels
