# Sentiment Polarity Detection using Word Sense Disambiguation

The text classification task Sentiment Analysis (SA) is concerned with automatically identifying emotions and opinions in texts [[1]](#1). The goal of one of its tasks, Sentiment Polarity Detection (SPD) is to classify texts into those that carry positive and those that carry negative emotions. SPD relies on Machine Learning (ML) methods, both supervised and unsupervised, which are trained on corpora of texts that are categorized as either positive or negative. Also, Natural Language Processing (NLP) techniques, such as Bag-of-Words, TF-IDF, Part-of-Speech and n-grams can be used as features for an ML model to predict the polarity. One of those techniques is word representations such as word embeddings and those more advanced such as contextualized and sense embeddings.

In this notebook, we will explore (deep) neural network models that have recently gained immense popularity as a method to obtain word and text representations. These models employ low dimensional vectors (embeddings), trained on large corpora. Word embeddings have proven to preserve semantic information of words, the most famous models being Word2Vec [[2]](#2), GloVe [[3]](#3) and FastText [[4]](#4). Usually, sentences and short texts are represented as the average of their corresponding word representations. 

Even though word embeddings gained quite popularity in practice, there still exists the inherited problem of polysemy since word embeddings do not completely represent the word meaning. Word embeddings capture one meaning per word, i.e. a word gets assigned a single embedding, even if that word is polysemous (ex. mouse the animal, computer mouse). Consequently, using word embeddings to represent sentences could also suffer from the same drawback. Due to the complexity of the problem, our main focus, for now, will be on the average of representations of sentence words and their senses as a feature to be fed to the ML model.

One of the solutions to represent word senses and tackle the problem of polysemy in short texts is to apply results from the domains of Word Sense Disambiguation (WSD) and sense representation learning. The WSD is a still open problem and an AI-complete task in NLP that involves mapping words to their right senses, given the context [[5]](#5). The WSD and sense representation learning can benefit from each other (see [[6]](#6)). Sense representations can be learned in both supervised and unsupervised fashion (see [[7]](#7)). 


## Methodology

We have relied on the following WSD approaches to disambiguate word senses and assign corresponding sense embeddings:
* Lesk algorithm, proposed by [[8]](#8), is used to determine the sense of the word by using dictionary definitions, i.e. glosses adopted from [WordNet](https://wordnet.princeton.edu/)
* Ewiser [[9]](#9) is a WSD system that has recently achieved state-of-the-art performance on the WSD task. It is a neural architecture that integrates knowledge bases (WordNet) and supervised WSD. 
* DeConf [[10]](#10), a model that contains word embeddings de-conflated by exploiting deep knowledge from WordNet
* [Retrofit](https://github.com/mfaruqui/retrofitting) [[11]](#11), a model that deals with retrofitting, i.e. improving the quality of word embeddings by leveraging the relations such as the synonymy, hypernymy, hyponymy and paraphrase from lexical resources like WordNet. Retrofit "moves" embeddings in the vector space so that they mirror the WordNet structure. 

In short, the following methods were employed:
1. Pre-trained [Word2Vec](https://code.google.com/archive/p/word2vec/) (W2V)
2. Retrofitted Word2Vec (RW2V)
3. Pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) (GV)
4. Retrofitted GloVe (RGV)
5. Pre-trained [FastText](https://fasttext.cc/docs/en/english-vectors.html) (FT)
8. [Lesk disambiguator](http://www.nltk.org/howto/wsd.html#:~:text=Lesk%20Algorithm&text=Given%20an%20ambiguous%20word%20and,from%20nltk) with [DeConf sense embeddings](https://pilehvar.github.io/deconf/)
9. [Ewiser disambiguator](https://github.com/SapienzaNLP/ewiser) with DeConf sense embeddings

We used an ensemble of four ML models to do the classification:
1. [Multilayer Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
2. [Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
3. [K-Nearest Neighbours Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
4. [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

The input to the classifier was the average of word or sense embeddings, while the output classes were -1 for negative and 1 for a positive review. 

## Dataset

The [polarity dataset v2.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/) includes 1000 positive and 1000 negative review texts. The reviews were processed (down-cased) and the ratings were removed. Positive reviews are stored in the pos folder, while negative reviews were stored in the neg folder. 

We took additional text pre-processing steps using [nltk](https://www.nltk.org/) - tokenization, stopwords removal and lemmatization.

## Results and Future Outlook

We report the following results for the ensemble classifier:

|Method|TP|FP|TN|FN|F1 score|
|-|-|-|-|-|-|
|W2V|163|82|180|75|0.69|
|RW2V|101|31|217|151|0.61|
|GV|136|53|195|116|0.66|
|RGV|136|46|179|139|0.62|
|FT|156|63|187|94|0.68|
|Lesk|169|34|213|84|0.76|
|Ewiser|190|33|211|66|0.8|

The best performing model was the Ewiser with DeConf sense embeddings. The second best was the approach using Lesk disambiguation with DeConf. These results show that the best performance was achieved by using WSD and sense embeddings, compared to approaches that use word embeddings only. However, the difference is rather small compared to the resources that systems such as Ewiser demand. In the future, to improve the classification results, we will employ classification models such as deep neural networks and more recent word representation models such as ELMo [[12]](#12) and BERT [[13]](#13).

## Bibliography

<a id="1">[1]</a> Pang, B., & Lee, L. (2007). Opinion Mining and Sentiment Analysis. Found. Trends Inf. Retr., 2, 1-135.<br>
<a id="2">[2]</a> Mikolov T, Chen K, Corrado G, Dean J. (2013). Efficient Estimation of Word Representations in Vector Space. CoRR abs/1301.3781<br>
<a id="3">[3]</a> Pennington, Jeffrey & Socher, Richard & Manning, Christopher. (2014). Glove: Global Vectors for Word Representation. EMNLP. 14. 1532-1543. 10.3115/v1/D14-1162.<br>
<a id="4">[4]</a> Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, 5, 135-146.<br>
<a id="5">[5]</a> Navigli, R. (2009). Word Sense Disambiguation: A survey. ACM Computing Surveys, 41(2), 1-69.<br>
<a id="6">[6]</a> Chen, X., Liu, Z., & Sun, M. (2014). A unified model for word sense representation and disambiguation. In Proceedings of EMNLP, pp. 1025-1035, Doha, Qatar.<br>
<a id="7">[7]</a> Camacho-Collados, J., & Pilehvar, M.T. (2018). From Word to Sense Embeddings: A Survey on Vector Representations of Meaning. J. Artif. Intell. Res., 63, 743-788.<br>
<a id="8">[8]</a> Banerjee, S., & Pedersen, T. (2002). An adapted Lesk algorithm for Word Sense Disambiguation using WordNet. In Proceedings of the Third International Conference on Computational Linguistics and Intelligent Text Processing, CICLingâ€™02, pp. 136-145, Mexico City, Mexico.<br>
<a id="9">[9]</a> Bevilacqua, M., & Navigli, R. (2020, July). Breaking through the 80% glass ceiling: Raising the state of the art in Word Sense Disambiguation by incorporating knowledge graph information. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 2854-2864).<br>
<a id="10">[10]</a> Pilehvar, M. T., & Collier, N. (2016). De-conflated semantic representations. In Proceedings of EMNLP, pp. 1680-1690, Austin, TX.<br>
<a id="11">[11]</a> Faruqui, M., Dodge, J., Jauhar, S. K., Dyer, C., Hovy, E., & Smith, N. A. (2015). Retrofitting word vectors to semantic lexicons. In Proceedings of NAACL, pp. 1606-1615.<br>
<a id="12">[12]</a> Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. In Proceedings of NAACL, New Orleans, LA, USA.<br>
<a id="13">[13]</a> Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805<br>