import re, math, glob, random
from typing import NamedTuple
from collections import defaultdict, Counter
from machine_learning import split_data, train_test_split

def tokenize(text):
    text = text.lower()
    all_words = re.findall("[a-z0-9']+", text)
    return set(all_words)

assert tokenize("Data Science is science") == {"data", "science", "is"}

class Message(NamedTuple):
    text: str
    is_spam: bool
    
class NaiveBayesClassifier:
    def __init__(self, k = 0.5) -> None:
        self.k = k
        
        self.tokens = set()
        self.token_spam_counts = defaultdict(int)
        self.token_ham_counts = defaultdict(int)
        self.spam_messages = self.ham_messages = 0
    
    def train(self, messages: Message):
        for msg in messages:
            if msg.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
        
            # increment word counts
            for token in tokenize(msg.text):
                self.tokens.add(token)
                if msg.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1
    
    def _probabilities(self, token):
        '''returns P(token | spam) and P(token | not spam)'''
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]
        
        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)
        
        return p_token_spam, p_token_ham
    
    def predict(self, text):
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0
        
        # iterate over each word in our vocabulary
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            
            # if token appears in the message, add the log prob of seeing it
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
            
            # otherwise add the log probabilittty of _not_ seeing it
            # which is log(1-prob of seeing it)
            else:
                log_prob_if_spam = math.log(1.0 - prob_if_spam)
                log_prob_if_ham = math.log(1.0 - prob_if_ham)
            
            prob_if_spam = math.exp(log_prob_if_spam)
            prob_if_ham = math.exp(log_prob_if_ham)
            
            return prob_if_spam / (prob_if_ham + prob_if_spam)
    
messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

text = "hello spam"

probs_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),      # "spam"  (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    (0 + 0.5) / (1 + 2 * 0.5)       # "hello" (present)
]

probs_if_ham = [
    (0 + 0.5) / (2 + 2 * 0.5),      # "spam"  (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    (1 + 0.5) / (2 + 2 * 0.5),      # "hello" (present)
]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

def drop_final_s(word):
    return re.sub("s$", "", word)

def main():
    path = 'spam_data/*/*'
    
    data = []
    
    for filename in glob.glob(path):
        is_spam = "ham" not in filename
        
        with open(filename, erore='ignore') as email_file:
            for line in email_file:
                if line.startswith("Subject:"):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(subject, is_spam))
                    break
        
    random.seed(0)
    train_messages, test_messages = split_data(data, 0.75)
    
    model = NaiveBayesClassifier()
    model.train(train_messages)
    
    predicitons = [(message, model.predict(message.text)) for message in test_messages]
    
    confusion_matrix = Counter((message.is_spam, spam_probability > 0.5) 
                               for message, spam_probability in predicitons)
    
    print(confusion_matrix)
    
    def p_spam_given_token(token , model: NaiveBayesClassifier):
        probs_if_spam, probs_if_ham = model._probabilities(token)
        
        return probs_if_spam /  (probs_if_spam + probs_if_ham)
    
    words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))
    
    print("Spammiest words:", words[-10:])
    print("Hammiest words:", words[:10])
    
if __name__ == "__main__": main()