# Topic identification methods for big data applications
Author: Silvio Gerli, Roberto Ascari, Sonia Migliorati, Teresa Cigna and Matteo Borrotti

# Abstract
Nowadays, the global amount of written texts grows with higher speed every day. For instance since 2011 the number of posts per minute on Facebook increased from 650K to 3M. In this context, these unstructured data represent the source of an enormous amount of information that should be extracted by using automatic engines. Natural Language Processing (NLP) is a field of Artificial Intelligence devoted to analyzing and understanding human language as it is spoken and written. One common task of NLP is topic identification, related to the recognition of a text's topic(s). 
 
Two popular methods for modeling latent topics are latent Dirichlet allocation (LDA) and correlated topic model (CTM). Both of them assume that each word composing a document is associated with a latent topic so that documents are represented as probability distributions over the topics, whereas topics are represented as probability distributions over the set of words composing the collection of documents. 
LDA and CTM differ in the prior distribution assigned to topics, thus showing different pros and cons.
In this work, LDA and CTM are tested and compared in a big data context by analyzing a  large set of documents (with certain topics) automatically downloaded from the web by means of a modern crawler.

In particular,  both the performance of LDA and CTM as classification tools, and their ability to adapt to modern large corpora of short documents are tested. Moreover, the issue of automatic interpretation of the identified topics is tackled.

# Files
websiteslist.txt : list of websites used for downloading the texts 

scrapedTexts_undersampling.csv: corpus example 

The corpus used in the paper can be downloaded at the following link: https://files.sinte.net/topicidentification/scrapedTexts.csv.zip
