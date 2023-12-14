import spacy
import pandas as pd
from spacy.matcher import Matcher
from functools import reduce

nlp = spacy.load("en_core_web_sm")

def get_entities(sent,nlp):
  import spacy
  import pandas as pd
  from spacy.matcher import Matcher
## chunk 1
  #nlp = spacy.load("en_core_web_sm")  
  Ent1 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""

  #############################################################
  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
      
      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""      

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        
      ## chunk 5  
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text  
  try:
      return [ent1.strip(), ent2.strip()]
  except:
      return
  
def get_relation(sent,nlp):
  import spacy
  import pandas as pd
  from spacy.matcher import Matcher
  #nlp = spacy.load("en_core_web_sm")
  doc = nlp(sent)

  # Matcher class object 
  matcher = Matcher(nlp.vocab)

  #define the pattern 
  pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

  matcher.add("matching_1", [pattern]) 

  matches = matcher(doc)
  k = len(matches) - 1

  span = doc[matches[k][1]:matches[k][2]] 

  return(span.text)


def isin_row(a, b):
    from functools import reduce
    cols = cols or a.columns
    return reduce(lambda x, y:x&y, [a[f].isin(b[f]) for f in cols])

def get_kg(sent,nlp):
    entity_pairs = []
    relations = []

    entity_pairs.append(get_entities(sent,nlp))
    relations.append(get_relation(sent,nlp))
  
    indices = [i for i, x in enumerate(entity_pairs) if x != None]
    entity_pairs = [entity_pairs[i] for i in indices]
    relations = [relations[i] for i in indices]
    subject = [i[0] for i in entity_pairs]

    # extract object
    object = [i[1] for i in entity_pairs]

    decl = pd.DataFrame({'subject':subject, 'object':object, 'predicate':relations})
    return decl



