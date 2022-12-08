# !pip install transformers

#! pip install https://github.com/huggingface/transformers/archive/nllb.zip -q

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="tam_Taml", tgt_lang='eng_Latn', max_length = 400)

translator("திஸ் ஐஸ் எ வெரி குட் மாடல் ")
