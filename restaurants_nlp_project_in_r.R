#For this project, we will be working with Natural Language Processing
#We are analyzing TEXT, not a simple dataset 
#NLP is a branch of machine learning because we are doing predictive analysis on text
#We will eventually create a Bag of Words model


#Here we are importing the dataset
#The file is in .tsv format, which is TAB SEPARATED VALUES
#In this special instance, tab delimmited values might be better. This is because the reviews have punctuation.
#If you separated values by comma, you would be breaking up individual reviews 
#This dataset is filled with restaurant reviews. We have 1000 reviews. One column for the review, one column with 1-0 outcome of review
#We will build some machine learning models to predict whether the reviews are negative or positive
#1 indicates a positive review and 0 indicates a negative review
#Having quote = '' means we are ignoring quotes in the text file. We don't want a misinterpretation 
#In NLP, we must not identify the reviews as "factors" because we are analyzing the inside of the reviews. So stringsAsFactors = FALSE
restaurants = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)


#We need to clean the reviews. The whole review is in one column. We need to break apart the reviews to find common words
#But obviously we don't want to end up with too many columns
#The first step for text cleaning is to create a corpus. We will not clean the reviews directly in the dataset.

#Let's first install the tm package
#install.packages('tm')
#Then import the tm library
library(tm)

#Now let's start building the corpus
#Enter the column that is actually being cleaned. This is of course the Review column
corpus = VCorpus(VectorSource(restaurants$Review))

#Let's look at the first review in the corpus
#It's "Wow...Loved this place."
as.character(corpus[[1]])


#For our first cleaning step, we will make every word in the reviews to be lower case
#The function content_transformer will allow us to convert to lower case using the tolower function inside of it
corpus = tm_map(corpus, content_transformer(tolower))


#Let's look at the first review in the corpus after the lower case transformation
#It's "wow... loved this place." So the transformation worked. 
as.character(corpus[[1]])



#Let's look at the 841st review in the corpus
#It's "for 40 bucks a head, i really expect better food."
as.character(corpus[[841]])

#The next step in our cleaning process will be to remove numbers from our reviews
corpus = tm_map(corpus, removeNumbers)

#Let's look at the 841st review in the corpus
#It's "for  bucks a head, i really expect better food." So 40 was removed and replaced by a space. 
as.character(corpus[[841]])


#The next step in our cleaning process will be to remove punctuation from our reviews
corpus = tm_map(corpus, removePunctuation)

#Let's look at the first review in the corpus after the punctuation transformation
#It's "wow loved this place." So the transformation worked. The ellipsis is gone. 
as.character(corpus[[1]])


#The next step in our cleaning process will be to remove irrelvant words from our reviews
#First let's install the SnowballC package
#install.packages("SnowballC")
#Then we will import the SnowballC library
library(SnowballC)
#The stopwords function has default irrelevant words to remove
corpus = tm_map(corpus, removeWords, stopwords())


#Let's look at the first review in the corpus after the remove words transformation
#It's "wow loved  place" So the word "this" was removed 
as.character(corpus[[1]])


#The next step in our cleaning process will be stemming
#Stemming involves finding the root of the word
corpus = tm_map(corpus, stemDocument)

#Let's look at the first review in the corpus after the stemming transformation
#It's "wow love place" So the word "loved" was replaced simply by "love"
as.character(corpus[[1]])



#The next step in our cleaning process will be removing white spaces from our reviews
corpus = tm_map(corpus, stripWhitespace)

#Let's look at the 841st review in the corpus
#It's "buck head realli expect better food" So the spaces were definitely removed. 
as.character(corpus[[841]])



#We have finished our cleaning, at least preliminary
#Now it is time to create our sparse matrix. The reviews in rows, words in columns
#Essentially we will be making our Bag of Words model
#This first line is creating the sparse matrix
#We end up getting 1000 rows (expected) and 1577 columns (quite a lot)
#One good way to reduce the amount of columns is to FILTER the words you want (or don't want)
dtm = DocumentTermMatrix(corpus)


#Here we will update our sparse matrix
#We are keeping 99.9% of the most frequent words
#More technically, it will keep the columns that have the most "1s" since sparse = 0
#Since each column is a word, if that column is mostly 0s, then it clearly doesn't appear in very many reviews
#The amount of columns was reduced to 691 by doing this
dtm = removeSparseTerms(dtm, 0.999)



#So we created our Bag of Words model above. Now we need to create a classification model
#We want to know which words lead to a 0 (negative review) or 1 (positive review)
#Our independent variables are in the sparse matrix. We must combine them with the "Liked" column from our original dataset
#So first we converted our sparse matrix to a dataframe. We called it "reviews"
reviews = as.data.frame(as.matrix(dtm))

#Here we are adding the Liked column from the original dataset to our new dataset
#We officially have our dependent variable.
reviews$Liked = restaurants$Liked


#Here we are encoding the target feature as factor
#We are doing this because of classification. Target variable must be 1-0
reviews$Liked = factor(reviews$Liked, levels = c(0, 1))

#Here we are splitting the dataset into the Training set and Test set
#First we import the caTools library that allows us to split the data
library(caTools)

#We are setting the seed so we get consistent results
set.seed(123)

#Officially splitting the data. 80% split
split = sample.split(reviews$Liked, SplitRatio = 0.80)
training_set = subset(reviews, split == TRUE)
test_set = subset(reviews, split == FALSE)


#So, the classification model we will use is Random Forest Classification
#Here we are fitting RFC to the Training set
#First we import the randomForest library
library(randomForest)

#Now we set up our RFC classifier
#No need for formula = y~x
#We will start with 10 trees in our forest
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)


#Here we are predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])


#Here we are making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
#So we have 82 + 77 correct predictions and 23 + 18 incorrect predictions
cm


#Some good classification models for NLP include:
#CART, C5.0, Maximum Entropy
