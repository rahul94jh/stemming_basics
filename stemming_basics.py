# import libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

text = "Very orderly and methodical he looked, with a hand on each knee, and a loud watch ticking \
a sonorous sermon under his flapped newly bought waist-coat,\
as though it pitted its gravity and longevity against the levity and evanescence of the brisk fire.\
Spirituality is a broad concept with room for many perspectives. In general, \
it includes a sense of connection to something bigger than ourselves, and it typically \
involves a search for meaning in life. As such, it is a universal human experience—something \
that touches us all. People may describe a spiritual experience as sacred or transcendent \
or simply a deep sense of aliveness and interconnectedness.Some may find that their \
spiritual life is intricately linked to their association with a church, temple,\
mosque, or synagogue. Others may pray or find comfort in a personal relationship \
with God or a higher power. Still others seek meaning through their connections \
to nature or art. Like your sense of purpose, your personal definition of \
spirituality may change throughout your life, adapting to your own experiences \
and relationships."

print(text)

# tokenize the text into words
tokens = word_tokenize(text.lower())
# remove stopwords
tokens = [token for token in tokens if token not in stopwords.words("english")]
print(tokens)

""" Just removing the stop words doesn't solve all the problems, the problem of having similar meaning words like win & winner, 
for handling these we use stemming which essetially chop or replace the words to its stem or root, porter and snowball are two
most common stemming technique with slight difference which we will see in the examples. """

""" Porter stemming :- http://snowball.tartarus.org/algorithms/porter/stemmer.html  """
# porter stemmer
stemmer = PorterStemmer()
porter_stemmed = [stemmer.stem(token) for token in tokens]
print(porter_stemmed)

""" Snowball stemming :- https://snowballstem.org/ """
""" SnowballStemmer have support for many languages as mentioned below :
('arabic', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 
'italian', 'norwegian', 'porter', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish') """
print(SnowballStemmer.languages)

# snowball stemmer
stemmer = SnowballStemmer("english")
snowball_stemmed = [stemmer.stem(token) for token in tokens]
print(snowball_stemmed)


""" Let's see the difference of stemming between porter and snowball stemmer """
# Create a dataframe containing tokens, porter stemmed tokens & snowball stemmed tokens in 3 columns
df = pd.DataFrame(
    {
        "token": tokens,
        "porter_stemmed": porter_stemmed,
        "snowball_stemmed": snowball_stemmed,
    }
)
df = df[["token", "porter_stemmed", "snowball_stemmed"]]
# Most of the stemming done are similar for both type of stemmer
print(df[(df.token != df.porter_stemmed) | (df.token != df.snowball_stemmed)])

# Orderly & General have difference in stemming performed by porter and snowball stemmer
print(df[(df.snowball_stemmed != df.porter_stemmed)])
