from flask import Flask, render_template, request
import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering

def bot(userText):
    qna_data = {"data":
    [
        {"title": "System Assistant",
         "paragraphs": [
             {
                 "context": "System Assistant is a good way of coordinating host access to the shared IP address, "
                            "however, it has its limitations. Good practices are required to avoid problems. For "
                            "example, an operator should avoid manual NW EC and NW CC commands for the "
                            "shared connections and always use the System Assistant script. To avoid mistakes, "
                            "scripts should be written to EC and CC all connections so the operator always "
                            "performs the same process for network connections. "
                            "Takeover scripts can be further improved by checking the resources of the critical "
                            "application (A). A script can be more certain of its actions if it checks for the presence "
                            "of something that can only exist when the critical application is running on this host "
                            "such as a disk, active entry, file or file attribute (for instance, IN USE).",
                 "qas": [
                     {"id": "Q1"
                      },
                 ]}]},]}

    text = qna_data["data"][0]["paragraphs"][0]["context"]
    input_dict = tokenizer(userText, text, return_tensors='tf')
    outputs = model(input_dict)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
    answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0]+1])
    
    return answer

app = Flask(__name__)

@app.route("/")
def home():    
    return render_template("home.html") 
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')
    answer = bot(userText)    
    return str(answer) 
if __name__ == "__main__": 
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')   
    app.run()

