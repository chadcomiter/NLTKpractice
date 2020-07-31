##Define function Tokenizer that has one parameter list. 
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer

##Defines a function SentTokenizer that takes a list of lines and returns a list of tokenized sentences
def SentTokenizer(lines):
    output = []
    for line in lines:
        output.append(sent_tokenize(line))
    return output
    
##Defines a function WordTokenizer that take a list of sentences and returns tokenized words.
def WordTokenizer(listOfSents):
    output = []
    for sentence in listOfSents:
        words = word_tokenize(sentence)
        output.append(words)
    return output
        
##Define a function, TxtFileLiner, with one parameter filepath. 
## TxtFileLiner reads through the .txt file and will return one giant string of all of the text in the file.
## Example "some_very_long_string = TxtFileLiner('/Users/path/')"
## This is helpful for cleaning an external txt file for tokenizing with NLTK.

def TxtFileLiner(filepath):
    output = []
    txtfile = open(filepath)
    lined = txtfile.readlines()
    for l in lined:
        output.append(l)
    txtfile.close()
    output = ''.join(output)
    return output

##Defines a function - Tagger, that takes one parameter, text.
##Tagger iterates through tokenized sentences and tags words with parts of speech.
##Output should be a list of tuples ie [('Congress', 'NNP), ('President', 'NNP')]
def Tagger(text):
    output = []
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            output.append(tagged)

    except Exception as e:
        print(str(e))
    
    return output

##Defines a function - NounChucker, that has one parameter, a list of tokenized sentences.
##Chunker groups tagged words into meaningful "noun phrases", and then returns the chunks.
##This chinks as well, and should be edited as such if you would like to change your chiking.
def NounChunker(tokenized):
    output = []
    try: 
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                output.append(subtree)
        return output
            
    
    except Exception as e:
        print(str(e))

##Defines a function feature finder that finds the top words in each list, marking their prescence as either positive, negative, or none.
def FeatureFinder(list):
    words = set(list)
    features = {}
    for w in list:
        features[w] = w in words
    return features