import os
import sys
import datetime
#import pyttsx3
import speech_recognition as sr

# import wikipedia
# import wolframalpha
import webbrowser
import smtplib
import random

# import gpt_2_simple as gpt2
import csv
from transformers import pipeline
from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
from transformers import AutoModelForSequenceClassification
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForSequenceClassification,
)

from transformers import (TFAutoModelWithLMHead, 
                         AutoTokenizer, 
                         pipeline, 
                         BlenderbotSmallTokenizer, 
                         BlenderbotForConditionalGeneration, 
                         Conversation)
from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
from transformers import pipeline
import random


from transformers import LongformerModel, LongformerTokenizer
from transformers import ReformerModelWithLMHead
from transformers import LongformerTokenizer, LongformerForQuestionAnswering

from transitions.extensions import HierarchicalGraphMachine as Machine
import torch
import pandas
from dataclasses import dataclass
import rpa as r

from typing import List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime

local_microphone = True
local_speaker = True


def can_you_type_that_out(query: str):
    r.init(visual_automation=True, chrome_browser=False)
    r.keyboard("[cmd][space]")
    r.keyboard("safari[enter]")
    r.keyboard("[cmd]t")
    r.keyboard("joker[enter]")
    r.wait(2.5)
    r.snap("page.png", "results.png")
    r.close()


class Query(BaseModel):
    uuid: str = uuid.uuid4()
    utterance_ts: datetime
    input: str
    output: str
    sentiment: str
    score: float


engine = pyttsx3.init("nsss")

# client = wolframalpha.Client('Get your own key')

# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[len(voices) - 1].id)


def talk(audio):
    print("Saati: " + audio)
    engine.say(audio)
    engine.runAndWait()


def greetMe():
    CurrentHour = int(datetime.now().hour)
    if CurrentHour >= 0 and CurrentHour < 12:
        talk("Good Morning!")

    elif CurrentHour >= 12 and CurrentHour < 18:
        talk("Good Afternoon!")

    elif CurrentHour >= 18 and CurrentHour != 0:
        talk("Good Evening!")


greetMe()


# model_name = "774M"
# sess = gpt2.start_tf_sess()

talk("Wait a sec got to get ready!")

# gpt2.load_gpt2(sess, model_name=model_name)

talk("Hey, it is Saati!")


def journal_sleep(response: str):
    CurrentHour = int(datetime.now().hour)
    if CurrentHour >= 0 and CurrentHour < 9:
        talk(" How well did you sleep ? ")
    elif CurrentHour >= 10 and CurrentHour <= 12:
        talk(" Did you sleep in? ")
    return response


# Scoring input as negative / positive


def film_scripts():

    """
    tokenizer = AutoTokenizer.from_pretrained("cpierse/gpt2_film_scripts")
    model = AutoModelWithLMHead.from_pretrained("cpierse/gpt2_film_scripts")
    # making sure dropout is turned off
    model.eval()
    max_length = 1000
    num_samples = 3

    output = model.generate(
            bos_token_id=random.randint(1,50000),
            do_sample=True,
            top_k=50,
            max_length = max_length,
            top_p=0.95,
            num_return_sequences=num_samples)

    decoded_output = []
    for sample in output:
            decoded_output.append(tokenizer.decode(
                    sample, skip_special_tokens=True))
            print(decoded_output[0])
    """
    pass


def compute_sentiment(utterance: str) -> dict:

    nlp = pipeline("sentiment-analysis")
    score = nlp(utterance)[0]
    # talk("The score was {}".format(score))
    return score


def guess_upvote_score(ctx: str):
    """
    Add a reddit / tweet composer and it will guess upvote score?
    """
    model_card = "microsoft/DialogRPT-updown"  # you can try other model_card listed in the table above
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    model = AutoModelForSequenceClassification.from_pretrained(model_card)

    def __score(cxt, hyp):
        model_input = tokenizer.encode(cxt + "<|endoftext|>" + hyp, return_tensors="pt")
        result = model(model_input, return_dict=True)
        return torch.sigmoid(result.logits)

    return __score(ctx, response)


# def get_session_context(query: Optional) -> str:
context = """ My name is satti. I have no idea of my purpose. I am programmed in python. I am release under the apache software licence.
	      """  # TODO make this a redis query.


def reinforcer(utterance: str, reward: float) -> any:
    # initialize trainer
    ppo_config = {"batch_size": 1, "forward_batch_size": 1}
    ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)

    # encode a query
    query_txt = utterance
    query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")

    # get model response
    response_tensor = respond_to_batch(gpt2_model, query_tensor)
    response_txt = gpt2_tokenizer.decode(response_tensor[0, :])

    # define a reward for response
    # (this could be any reward such as human feedback or output from another model)
    reward = torch.tensor([reward])

    # train model with ppo
    train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)

    return train_stats


def answer_question(query: str, context: str):
    """
    {'score': 0.5135612454720828, 'start': 35, 'end': 59, 'answer': 'huggingface/transformers'}
    """

    question_answerer = pipeline("question-answering")
    answer_to_question = question_answerer({"question": query, "context": context})
    talk(answer_to_question.get("answer", "I dont know"))
    return (query, answer_to_question["answer"])


## Start reformer
# Use for language generation from large amounts of information like books.
# Encoding
def encode(list_of_strings, pad_token_id=0):
    max_length = max([len(string) for string in list_of_strings])

    # create emtpy tensors
    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full(
        (len(list_of_strings), max_length), pad_token_id, dtype=torch.long
    )

    for idx, string in enumerate(list_of_strings):
        # make sure string is in byte format
        if not isinstance(string, bytes):
            string = str.encode(string)

        input_ids[idx, : len(string)] = torch.tensor([x + 2 for x in string])
        attention_masks[idx, : len(string)] = 1

    return input_ids, attention_masks


# Decoding
def decode(outputs_ids):
    decoded_outputs = []
    for output_ids in outputs_ids.tolist():
        # transform id back to char IDs < 2 are simply transformed to ""
        decoded_outputs.append(
            "".join([chr(x - 2) if x > 1 else "" for x in output_ids])
        )
    return decoded_outputs


def reformer(question: str):
    model = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
    encoded, attention_masks = encode(
        ["In 1965, Brooks left IBM to found the Department of"]
    )
    decoded = decode(model.generate(encoded, do_sample=True, max_length=300))
    talk(decoded)
    return decoded


##End Reformer


# summarizer = pipeline('summarization')
# summary = summarizer(JMC_TEXT_TO_SUMMARIZE)

##########################################################################
# def questions(input_text: str):                                        #
#     model_name = "uncased_L-12_H-768_A-12"                             #
#     model_dir = bert.fetch_google_bert_model(model_name, ".models")    #
#     model_ckpt = os.path.join(model_dir, "bert_model.ckpt")            #
#                                                                        #
#     bert_params = bert.params_from_pretrained_ckpt(model_dir)          #
#     l_bert = bert.BertModelLayer.from_params(bert_params, name="bert") #
##########################################################################

# use in Keras Model here, and call model.build()

# bert.load_bert_weights(l_bert, model_ckpt)      # should be called after model.build()


def smalltalk(UTTERANCE: str):
    mname = "facebook/blenderbot-90M"
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
    # UTTERANCE = "My friends are cool but they eat too many carbs."
    inputs = tokenizer([UTTERANCE], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    responses = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in reply_ids
    ]
    # logger.debug(responses)
    talk(responses[0])
    return responses

def smalltalk_memory(UTTERANCE: str):
	from transformers import AutoModelForCausalLM, AutoTokenizer
	import torch

	tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
	model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

	# Let's chat for 5 lines
	for step in range(5):
		# encode the new user input, add the eos_token and return a tensor in Pytorch
		new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

		# append the new user input tokens to the chat history
		bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

		# generated a response while limiting the total chat history to 1000 tokens, 
		chat_history_ids = model.generate(bot_input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id) #Changing to 100 for tweets.

		# pretty print last ouput tokens from bot
		print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

##Longformer for large texts


def longformer(TO_SUMMARIZE: str):
    tokenizer = LongformerTokenizer.from_pretrained(
        "allenai/longformer-large-4096-finetuned-triviaqa"
    )
    model = LongformerForQuestionAnswering.from_pretrained(
        "allenai/longformer-large-4096-finetuned-triviaqa", return_dict=True
    )
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    input_dict = tokenizer(question, text, return_tensors="tf")
    outputs = model(input_dict)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
    answer = " ".join(
        all_tokens[
            tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0] + 1
        ]
    )
    sequence_output = outputs.last_hidden_state
    pooled_output = outputs.pooler_output
    talk(pooled_output)
    return outputs


##############################################################################
# def gpt2_reinforcment(UTTERANCE: str):									 #
# 	tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb-ctrl")		 #
# 	model = AutoModel.from_pretrained("lvwerra/gpt2-imdb-ctrl")				 #
##############################################################################


def poems(input_text: str):  # run_name='/Users/r2q2/Projects/waifu2020/src/models'):
    talk("hey let me think about that")
    ####################################################################################
    ### sess = gpt2.start_tf_sess()                                                  ###
    ### gpt2.load_gpt2(sess,run_name="run1")                                         ###
    ### talk(gpt2.generate(sess,                                                     ###
    ###           #checkpoint_dir='/Users/r2q2/Projects/waifu2020/src/models/775M/', ###
    ###           length=250,                                                        ###
    ###           temperature=0.7,                                                   ###
    ###           prefix=input_text,                                                 ###
    ###                                                                              ###
    ###           return_as_list=True)[0])                                           ###
    ####################################################################################
    # from transformers import AutoTokenizer,
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # model = AutoModelWithLMHead.from_pretrained("gpt2")
    # generator = pipeline('text-generation', model='gpt2')
    # set_seed(42)
    # stuff_to_say = generator(input_text, max_length=30, num_return_sequences=1)
    # talk(stuff_to_say)
    # gpt2.download_gpt2(model_name=model_name)
    # stuff_to_say = gpt2.generate(sess,
    ##########################
    # model_name=model_name, #
    # prefix=input_text,     #
    # length=40,             #
    # temperature=0.7,       #
    # top_p=0.9,             #
    # nsamples=1,            #
    # batch_size=1,          #
    # return_as_list=True,   #
    # #reuse=True            #
    # )[0]                   #
    ##########################
    # import pdb; pdb.set_trace()
    # talk(stuff_to_say)
    # talk(what_to_say)
    # return stuff_to_say
    # TODO  add a feedback question here


########################################################################
# from mic_vad_streaming import ingest								   #
# def GetInput():													   #
# 	parameters = {'model' : '',										   #
# 				  'scorer' : '../deepspeech-0.9.1-models.scorer',	   #
# 				  }													   #
# 	voice_ingest(../)												   #
########################################################################


def voice_ingest(model, scorer, sample_rate=16000, vad_aggressiveness=3):
    # Load DeepSpeech model
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, "output_graph.pb")
        ARGS.scorer = os.path.join(model_dir, ARGS.scorer)

    talk("Initializing model...")
    logging.info("ARGS.model: %s", ARGS.model)
    model = deepspeech.Model(ARGS.model)
    if ARGS.scorer:
        logging.info("ARGS.scorer: %s", ARGS.scorer)
        model.enableExternalScorer(ARGS.scorer)

    # Start audio with VAD
    vad_audio = VADAudio(
        aggressiveness=ARGS.vad_aggressiveness,
        device=ARGS.device,
        input_rate=ARGS.rate,
        file=ARGS.file,
    )
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()

    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner="line")
    stream_context = model.createStream()
    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            if spinner:
                spinner.start()
            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            if ARGS.savewav:
                wav_data.extend(frame)
        else:
            if spinner:
                spinner.stop()
            logging.debug("end utterence")
            if ARGS.savewav:
                vad_audio.write_wav(
                    os.path.join(
                        ARGS.savewav,
                        datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav"),
                    ),
                    wav_data,
                )
                wav_data = bytearray()
            text = stream_context.finishStream()
            print("Recognized: %s" % text)
            if ARGS.keyboard:
                from pyautogui import typewrite

                typewrite(text)
            stream_context = model.createStream()


def GivenCommand():
<<<<<<< HEAD
	
	k = sr.Recognizer()
	with sr.Microphone() as source:
		print("Listening...")
		k.pause_threshold = 1
		audio = k.listen(source)
	try:
		Input = k.recognize_google(audio, language='en-us')
		
		print('You: ' + Input + '\n')

	except sr.UnknownValueError:
		talk('Gomen! I didn\'t get that! Try typing it here!')
		Input = str(input('Command: '))

	
	return Input


if __name__ == '__main__':

	while True:

		#Configuration
		recordSleep = True
		
		Input = GivenCommand()
		
		#print("Upvote score is %d".format( guess_upvote_score(Input)))

		Input = Input.lower() #TODO should i keep this?

		if 'i am tired 'in Input:
			#answer = journal_sleep(Input)
			sentiment = compute_sentiment(Input)

			fields=[datetime.utcnow() , Input, answer, sentiment]
			with open(r'datasette_log', 'a') as f:
				writer = csv.writer(f)
				writer.writerow(fields)
		
		#elif "what\'s up" in Input or 'how are you' in Input:
		#	setReplies = ['Just doing some stuff!', 'I am good!', 'Nice!', 'I am amazing and full of power']
		#	talk(random.choice(setReplies))

		#elif "who are you" in Input or 'where are you' in Input or 'what are you' in Input:
		#	setReplies = [' I am Saati', 'In your system', 'I am an example of AI']
		#	talk(random.choice(setReplies))

		elif 'email' in Input:
			talk('Who is the recipient? ')
			recipient = GivenCommand()

			if 'me' in recipient:
				try:
					talk('What should I say? ')
					content = GivenCommand()

					server = smtplib.SMTP('smtp.gmail.com', 587)
					server.ehlo()
					server.starttls()
					server.login("Your_Username", 'Your_Password')
					server.sendmail('Your_Username', "Recipient_Username", content)
					server.close()
					talk('Email sent!')

				except:
					talk('Sorry ! I am unable to send your message at this moment!')

		elif 'nothing' in Input or 'abort' in Input or 'stop' in Input:

			talk('okay')
			talk('Bye, have a good day.')
			sys.exit()

		elif 'hello' in Input:
			talk('hey')

		elif 'bye' in Input:
			talk('Bye, have a great day.')
			sys.exit()


		elif 'smalltalk' or 'what do you think'  in Input:
			output = smalltalk(Input)
			recipient = GivenCommand()
			sentiment = compute_sentiment(Input)

			fields=[datetime.utcnow() , Input, output, sentiment]
			with open(r'datasette_log', 'a') as f:
				writer = csv.writer(f)
				writer.writerow(fields)
		elif 'explain' in Input:
			logger.debug("longformer is being used")
			explanation = longformer(Input)
			fields=[datetime.utcnow() , Input,explanation, sentiment]
			with open(r'datasette_log', 'a') as f:
				writer = csv.writer(f)
				writer.writerow(fields)
		elif 'can i text you' or 'what is your phone number' in Input:
			talk('1 778 403 5044')

			
		else:
			Input = Input
			


			talk('Searching...')
			try:
				try:
					res = client.Input(Input)
					outputs = next(res.outputs).text
					talk('Alpha says')
					talk('Gotcha')
					talk(outputs)

				except:
					outputs = wikipedia.summary(Input, sentences=3)
					talk('Gotcha')
					talk('Wikipedia says')
					talk(outputs)


			except:
					talk("searching on google for " + Input)
					say = Input.replace(' ', '+')
					webbrowser.open('https://www.google.co.in/search?q=' + Input)
		
			talk("Sorry I can't provide a good response")
		
		#talk('Next Command! Please!')
