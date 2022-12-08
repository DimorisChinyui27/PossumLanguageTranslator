#Requirements
# !pip install transformers
# pip install pytorch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)


model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

@app.route("/translatetext", methods=['POST'])
def translator():
    request_data = request.get_json()
    src_language = str(request_data["source_language"])
    tgt_language=str(request_data["targt_language"]) 
    text=str(request_data["text"])
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=src_language, tgt_lang=tgt_language, max_length = 3000)
    translated_text=translator(text)
    return jsonify(translated_text[0])


if __name__== "__main__":
    app.run(debug=True)
