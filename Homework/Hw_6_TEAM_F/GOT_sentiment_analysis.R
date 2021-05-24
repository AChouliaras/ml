if (!require("pacman")) install.packages("pacman")
pacman::p_load("ROAuth","NLP", "twitteR", "syuzhet","tm","SnowballC","topicmodels")
library("NLP", lib.loc="C:/Users/Admin/.conda/envs/ml/Lib/R/library")
library("twitteR", lib.loc="C:/Users/Admin/.conda/envs/ml/Lib/R/library")
library("syuzhet", lib.loc="C:/Users/Admin/.conda/envs/ml/Lib/R/library")
library("tm", lib.loc="C:/Users/Admin/.conda/envs/ml/Lib/R/library")
library("SnowballC", lib.loc="C:/Users/Admin/.conda/envs/ml/Lib/R/library")
library("stringi", lib.loc="C:/Users/Admin/.conda/envs/ml/Lib/R/library")
library("topicmodels", lib.loc="C:/Users/Admin/.conda/envs/ml/Lib/R/library")
library("ROAuth", lib.loc="C:/Users/Admin/.conda/envs/ml/Lib/R/library")

setup_twitter_oauth("RVeTmWH2A2ukkjZDwBe01zsFf", "glXL94GfruZnFFFS7hIOZbXB9RsIWVjS1FZhYYKdSmiejy64uG",
                    "16065195-HAL7NqTRtvE3XOqRoXZeUAsjmZmABJNZWdA9imYR1",
                    "6Z5KjZyDwAdSLQuzGvQ9ChmYj8dX8qRUkvPk8Nnukax2y")

tweets_g <- searchTwitter("#trump", n=1000,lang = "en")

got_tweets <- twListToDF(tweets_g)
got_text<- got_tweets$text
#convert all text to lower case
got_text<- tolower(got_text)
# Replace blank space (“rt”)
got_text <- gsub("rt", "", got_text)
# Replace @UserName
got_text <- gsub("@\\w+", "", got_text)
# Remove punctuation
got_text <- gsub("[[:punct:]]", "", got_text)
# Remove links
got_text <- gsub("http\\w+", "", got_text)
# Remove tabs
got_text <- gsub("[ |\t]{2,}", "", got_text)
# Remove blank spaces at the beginning
got_text <- gsub("^ ", "", got_text)
# Remove blank spaces at the end
got_text <- gsub(" $", "", got_text)
#create corpus
got_tweets.text.corpus <- Corpus(VectorSource(got_text))

#clean up by removing stop words
got_tweets.text.corpus <- tm_map(got_tweets.text.corpus, function(x)removeWords(x,stopwords()))

library("wordcloud", lib.loc="C:/Users/Admin/.conda/envs/ml/Lib/R/library")
#generate wordcloud
wordcloud(got_tweets.text.corpus,min.freq = 10,colors=brewer.pal(8, "Dark2"),random.color = TRUE,max.words = 500)

#getting emotions using in-built function
mysentiment_got<-get_nrc_sentiment((got_text))

#calculationg total score for each sentiment
Sentimentscores_got<-data.frame(colSums(mysentiment_got[,]))
names(Sentimentscores_got)<-"Score"
Sentimentscores_got<-cbind("sentiment"=rownames(Sentimentscores_got),Sentimentscores_got)
rownames(Sentimentscores_got)<-NULL

library(ggplot2)
#plotting the sentiments with scores
ggplot(data=Sentimentscores_got,aes(x=sentiment,y=Score))+geom_bar(aes(fill=sentiment),stat = "identity")+
  theme(legend.position="none")+
  xlab("Sentiments")+ylab("scores")+ggtitle("Sentiments of people behind the tweets on Donald Trump")
