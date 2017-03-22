train <- read.csv('train.csv')
atts <- read.csv('attributes.csv')
prods_desc <- read.csv('product_descriptions.csv')
atts <- atts[-c(2)]
atts <- aggregate(value ~ product_uid, data = atts, paste, collapse = ",")
train_att <- merge(train,atts,by='product_uid',all.x = TRUE)
train_descr <- merge(train,prods_desc,by='product_uid',all.x = TRUE)
train_descr_att <- merge(train_att,prods_desc,by='product_uid',all.x = TRUE)
# Install
install.packages("tm")  # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes
# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")

text <- train_descr_att['search_term']
docs <- VCorpus(VectorSource(text))
#docs <- tm_map(docs, removeWords, stopwords("english"))
#docs <- tm_map(docs, removeWords, c('ft.'))

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)

v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)

set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

library(cluster)   
d <- dist(t(dtmss), method="euclidian")   
fit <- hclust(d=d, method="ward")   
fit   
plot(fit, hang=-1)  