from django.shortcuts import render
from newspaper import Article
from summa import summarizer
from summa import keywords
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
# from openai import OpenAI
import google.generativeai as genai
import os
import numpy as np
import re
import pdfplumber

from rouge_score import rouge_scorer

from .models import users

# Create your views here.

def home(request):
    return render(request, "home.html")

def article(request):
    return render(request, "article.html", {})

def transcript(request):
    return render(request, "transcript.html", {})

def document(request):
    return render(request, "document.html", {})

def contactus(request):
    return render(request, "contactus.html", {})

def accuracy(request):

    article = Article("https://en.wikipedia.org/wiki/Suffering")
    article.download()
    article.parse()
    doc = article.text

    ext_summary = summarizer.summarize(doc, 0.2)

    with open('/home/nishant/Desktop/editor-master/article/ref_summ.txt', 'r') as file:
        ref_summ = file.read()

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score( ext_summary, ref_summ)

    wordLen1 = (len(re.findall(r'\w+', doc)))
    wordLen2 = (len(re.findall(r'\w+', ext_summary)))
    wordLen3 = (len(re.findall(r'\w+', ref_summ)))

    dictList = {"score": scores,
    "text": doc,
    "ext_summary": ext_summary,
    "ref_summ": ref_summ,
    "count1": wordLen1,
    "count2": wordLen2,
    "count3": wordLen3,
    }

    return render(request, "accuracy.html", dictList)


def split_text(text):
    max_chunk_size = 2048
    chunks = []
    current_chunk = ""
    for sentence in text.split("."):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_summary(text):
    input_chunks = split_text(text)
    output_chunks = []

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

    safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    ]
    model = genai.GenerativeModel(model_name='gemini-pro', safety_settings=safety_settings)

    for chunk in input_chunks:
        response = model.generate_content(f"Please summarize the following text:\n{chunk}\n\nSummary:")
        # response = client.chat.completions.create(
        #     messages=[
        #     {
        #     "role": "user",
        #     "content": (f"Please summarize the following text:\n{chunk}\n\nSummary:"),
        #     }
        #     ],
        #     model="davinci-002",
        #     )
        # summary = response.choices[0].text.strip()
        summary = response.text
        output_chunks.append(summary)

    return " ".join(output_chunks)

def transcriptdownload(request):

    url = request.POST.get('gettranscript')
    video_id = url.split('=')[1]

    json = YouTubeTranscriptApi.get_transcript(video_id)

    pars = ""
    for i in json:
        pars += i["text"]
        pars += " "

    # model_name = "facebook/bart-base"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    summ = generate_summary(pars)

    wordLen1 = (len(re.findall(r'\w+', pars)))
    wordLen2 = (len(re.findall(r'\w+', summ)))

    percent = "{:.2f}".format(((wordLen2/wordLen1) * 100))

    # text = summarizer(pars, max_length=300, min_length=130, do_sample=False)

    # print(text)

    # text = huggingface.get("summarization", model="t5-small")(para)

    # summ = text[0]['generated_text']

    dictList = {'text': pars,
    'summary': summ,
    'wordcount1': wordLen1,
    'wordcount2': wordLen2,
    'percent': percent,
    }

    return render(request, 'transcript.html', dictList)


def articledownload(request):

    url = request.POST.get('getarticle')
    sum_ratio = float(request.POST.get('ratio'))
    
    article = Article(url)
    article.download()
    article.parse()
    doc = article.text

    doctitle = article.title

    print(doctitle)

    ext_summary = summarizer.summarize(doc, sum_ratio)
    
    txt_keyword = keywords.keywords(doc)
    sum_keyword = keywords.keywords(ext_summary)

    list1 = (txt_keyword.split())
    list2 = (sum_keyword.split())

    #parameter = "{:.2f}".format(percentage2)

    wordLen1 = (len(re.findall(r'\w+', doc)))
    wordLen2 = (len(re.findall(r'\w+', ext_summary)))

    # print(keyword)

    # model_name = "facebook/bart-large-cnn"
    # tokenizer = BartTokenizer.from_pretrained(model_name)
    # model = BartForConditionalGeneration.from_pretrained(model_name)

    # input_text = ext_summary

    # inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
    # summary_ids = model.generate(inputs, max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    
    # abs_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # # T5 Abstractive Summarization

    # Load pre-trained T5 model and tokenizer
    # model_name = "t5-small"
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name)

    # # Input text to be summarized
    # input_text = ext_summary

    # # Tokenize and summarize the input text using T5
    # inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
    # summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    # abs_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


    percent = "{:.2f}".format(((wordLen2/wordLen1) * 100))

    dictList ={'text':doc,
    'title':doctitle,
    'summary': ext_summary,
    'wordcount1': wordLen1,
    'wordcount2': wordLen2,
    'percent': percent,
    }

    return render(request, 'article.html', dictList)

def thankyou(request):

    getname = request.POST.get('getname')
    getemail = request.POST.get('getemail')
    getmessage = request.POST.get('getmessage')

    users.objects.create(name=getname, email=getemail, message=getmessage)

    return render(request, 'thankyou.html', {'ack': 1})

def documentdownload(request):
    
    pdfFile = request.FILES['getpdf']
    stype = request.POST.get('summtype')

    paperContent = pdfplumber.open(pdfFile).pages 

    text = ""
    tldr_tag = "\n tl;dr:"
    for page in paperContent:
        text += page.extract_text() + tldr_tag
        if len(text) > 50000:
            text = text[0:50000]
            print("exceed\n")
            break
        print("iteration\n")

    print(text)
    print()


    if stype=="abstractive":
        summ = generate_summary(text)

    elif stype=="extractive":
        summ = summarizer.summarize(text, 0.2)

    wordLen1 = (len(re.findall(r'\w+', text)))
    wordLen2 = (len(re.findall(r'\w+', summ)))

    percent = "{:.2f}".format(((wordLen2/wordLen1) * 100))

    # text = summarizer(pars, max_length=300, min_length=130, do_sample=False)

    # print(text)

    # text = huggingface.get("summarization", model="t5-small")(para)

    # summ = text[0]['generated_text']

    dictList = {'text': text,
    'summary': summ,
    'wordcount1': wordLen1,
    'wordcount2': wordLen2,
    'percent': percent,
    }

    return render(request, 'document.html', dictList)