import nltk
# nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model_exp.h5')
import json
import random
intents = json.loads(open('F:\\farmer_bot_final-20230625T120404Z-001\\farmer_bot_final\\farrmer_intents.json', encoding='utf-8').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
new_queries=[]

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# def handle_unknown_intent():
#     # Add the user query to the new queries list
#     # new_queries.append(user_query)
    
#     # Craft a response message
#     response = "I'm sorry, but I couldn't find a suitable answer for your query. " 
#             #    "I have forwarded it to our team at the Knowledge Curation Center (KCC) for further assistance. " \
#             #    "They will get back to you s
#     return response

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    print(ints)
    list_of_intents = intents_json['intents']
    # print(list_of_intents)
    tag_match=False
    for i in list_of_intents:
        # print(tag)
        # print(i["tag"])
        if(i['tag']== tag and float(ints[0]['probability'])>.8):
            print(tag)
            result = random.choice(i['responses'])
            tag_match=True
            break

    if tag_match==True:
        return result
            
    else:
        result="I'm sorry, but I couldn't find a suitable answer for your query.I have forwarded it to our team at the Knowledge Curation Center (KCC) for further assistance. "
        # print(result)
        return result 
    
    # print(result)
    # return result

def chatbot_response(msg):
    f=open("queries.txt","a")
    f.write("\n")
    f.write(msg)
    f.close()
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# if __name__=='__main__':
#     chatbot_response("what is the best season to grow soyabean crop?")

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()
   