
test = 'Shohei Ohtani, nicknamed "Shotime", is a Japanese professional baseball pitcher'
import nltk

def blue_score(hypothesis_text,reference_text):
  hypothesis = hypothesis_text.split()
  reference = reference_text.split()
  references = [reference] 
  list_of_references = [references]
  list_of_hypotheses = [hypothesis]
  return nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses)

print(blue_score(test,test))

