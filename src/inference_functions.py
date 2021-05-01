from transformers import (
    TFAutoModelWithLMHead,
    AutoTokenizer,
    pipeline,
    BlenderbotTokenizer,
    BlenderbotSmallTokenizer,
    BlenderbotForConditionalGeneration,
    Conversation,
)
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline, Conversation
import logging
from typing import List, Any, Tuple

conversational_pipeline = pipeline("conversational", device=0)

def blenderbot400M(utterance: str) -> List[str]:

    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
    inputs = tokenizer([utterance], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    responses = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in reply_ids
    ]
    return responses

def questions(question: str, text: str) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained('ibert-roberta-base')
    model = AutoTokenizer.from_pretrained('ibert-roberta-base')

    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    inputs = tokenizer(question, text, return_tensors='pt')
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])

    outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    loss = outputs.loss
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    return outputs

def general_questions(question: str, text: str) -> List[str]:
    nlp = pipeline("question-answering")
    return nlp
    #tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

    #model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

def checkpoint_optimization(utterance: str, checkpoint: Any) -> Tuple[str]:
    # instantiate sentence fusion model
    sentence_fuser = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse")
    tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

    input_ids = tokenizer('This is the first sentence. This is the second sentence.', add_special_tokens=False, return_tensors="pt").input_ids

    outputs = sentence_fuser.generate(input_ids)

    print(tokenizer.decode(outputs[0]))

def conversation(utterance: str, continuing_conversation=False):
    element_to_access = random.randint(0,len(letters)-1)
    conv1_start = blenderbot400M(utterance)[element_to_access]
    #conv1_start = "Let's watch a movie tonight - any recommendations?"
    conv2_start = utterance
    
    return conversational_pipeline([conv1_start, conv2_start])
    #conv1 = Conversation(conv1_start)
    #conv2 = Conversation(conv2_start)
    #conv1.add_user_input(conv1_next)
    #conv2.add_user_input(conv2_next)

    #conversational_pipeline([conv1, conv2])

def blenderbot3B(utterance: str):
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-3B")

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-3B")
    inputs = tokenizer([utterance], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    responses = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in reply_ids
    ]
    return responses

def blenderbot1B(utterance: str):
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-1B-distill")

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    inputs = tokenizer([utterance], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    responses = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in reply_ids
        ]
    return responses

        
def smalltalk(utterance: str) -> List[str]:

    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info("starting smalltalk")
    mname = "facebook/blenderbot-3B"
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    #model.to(device)
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    inputs = tokenizer([utterance], return_tensors="pt")#.to(device)
    reply_ids = model.generate(**inputs)
    responses = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in reply_ids
    ]
    return responses
