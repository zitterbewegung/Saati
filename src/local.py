import os
import sys
import datetime
import pyttsx3
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
    #pipeline,
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    #Conversation,
    TFAutoModelWithLMHead
)
#from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
#from transformers import pipeline
import uuid, json, pickle
from typing import List, Any

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

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

#if 'Windows' == platform.system():
engine = pyttsx3.init()

# client = wolframalpha.Client('Get your own key')

#!pip install streamlit
#!pip install transitions[diagrams]
#!pip install graphviz pygraphviz
#!brew install graphviz
# from transitions.extensions import GraphMachine as Machine
from transitions import Machine

import random
from datetime import datetime

# Set up logging; The basic log level will be DEBUG
import logging
import speech_recognition as sr
import torch
import numpy as np

import streamlit as st

logging.basicConfig(level=logging.INFO)

# engine = pyttsx3.init("nsss")


class Saati(object):
    def __init__(self, name, debugMode=False):
        # No anonymous superheroes on my watch! Every narcoleptic superhero gets
        # a name. Any name at all. SleepyMan. SlumberGirl. You get the idea.
        self.name = name

        # How do we feel about the person.
        self.sentiment = 1

        # Interaction_number
        self.interaction_number = 0

        # Figure out outcome that would put you in the friendzone?
        # self.love_vector = self.impression_points * random.randrange(20) / self.interaction_number

        # Initialize the state machine

        # states represent where you are.
        states = [
            "initializemodels",
            "meetup",
            "hangingout",
            "sleeping",
            "wake_up",
            "leave",
        ]
        self.machine = Machine(model=self, states=states, initial="initializemodels")
        self.machine.add_ordered_transitions()
        self.machine.add_transition(trigger="friendzone", source="*", dest=None)
        # Initialize models


def GivenCommand(test_mode=True):
    Input = ""
    if test_mode:
        Input = input("Resp>>")
        return Input
    else:
        k = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            k.pause_threshold = 1
            audio = k.listen(source)
        try:
            Input = k.recognize_google(audio, language="en-us")
            talk("You: " + Input + "\n")
        except sr.UnknownValueError:
            talk("Gomen! I didn't get that! Try typing it here!")
            Input = str(input("Command: "))
    return Input




def is_a_question(utterance: str) -> bool:
    START_WORDS = [
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "is",
        "can",
        "does",
        "do",
    ]
    for word in START_WORDS:
        if word in START_WORDS:
            return True
    return false


def talk(text: str):
    logging.info("starting waveglow")
    device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    if device_to_use:
        logging.info("Saati: " + text)

        engine.say(text)
        engine.runAndWait()
    else:

        waveglow = torch.hub.load(
            "nvidia/DeepLearningExamples:torchhub", "nvidia_waveglow"
        )
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to(device_to_use)
        waveglow.eval()
        tacotron2 = torch.hub.load(
            "nvidia/DeepLearningExamples:torchhub", "nvidia_tacotron2"
        )
        tacotron2 = tacotron2.to(device_to_use)
        tacotron2.eval()
        # preprocessing
        sequence = np.array(tacotron2.text_to_sequence(text, ["english_cleaners"]))[
            None, :
        ]
        sequence = torch.from_numpy(sequence).to(
            device=device_to_use, dtype=torch.int64
        )

        # run the models
        with torch.no_grad():
            _, mel, _, _ = tacotron2.infer(sequence)
            audio = waveglow.infer(mel)
            audio_numpy = audio[0].data.cpu().numpy()
            rate = 22050

            write("/tmp/audio.wav", rate, audio_numpy)
            with open("/tmp/audio.wav", "rb") as f:
                b = f.read()
                play_obj = sa.play_buffer(b, 2, 2, 22050)

                play_obj.wait_done()

    # return audio


def GivenCommand(test_mode=False):
    Input = ""
    if test_mode:
        Input = input("Resp>>")
    else:
        k = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            k.pause_threshold = 1
            audio = k.listen(source)
        try:
            Input = k.recognize_google(audio, language="en-us")
            print("You: " + Input + "\n")
        except sr.UnknownValueError:
            talk("Gomen! I didn't get that! Try typing it here!")
            Input = str(input("Command: "))
    return Input


class Event(BaseModel):
    uuid: str = uuid.uuid4()
    timestamp: datetime = datetime.now()
    responses: List[str] = []
    sentiment: int = 1
    interactions: int = 1
    sync_ratio: float = 1
    state_machine: Any


# function to add to JSON
def write_json(data, filename="event_log.json"):
    with open(filename, "a+") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


# def compute_sentiment(utterance: str) -> float:
#     nlp = pipeline("sentiment-analysis")
#     result = nlp(utterance)
#     score = result[0]["score"]
#     if result[0]["label"] == "NEGATIVE":
#         score = score * -1

#     logging.info("The score was {}".format(score))
#     return score


def local_ingest():
    """
    If pos or neg pos 5 to 1 relationship doesn't continue
    If exceeds 11 pos 1 neg no challenge
    you wlant not bliss but
    """

    instance = Saati(uuid.uuid4())

    user_input = GivenCommand()

    from pathlib import Path
    import pickle

    my_file = Path("event_log.json")
    state_machine = pickle.dumps(instance)
    current_state = Event()

    if my_file.is_file():
        with open("event_log.json", "r") as f:
            data = f.read()
            save_state = json.loads(data)
    write_json(current_state.json())
    # if my_file.is_file():
    #    current_state = pickle.loads(my_file)
    # else:
    #    current_state = pickle.dumps(current_state)

    while True:
        # instance.get_graph().draw('my_state_diagram.png', prog='dot')

        logging.info("Computing reply")
        responce = smalltalk(user_input)[0]
        talk(responce)
        current_state.responses.append(responce)
        current_state.sentiment = current_state.sentiment + compute_sentiment(
            user_input
        )
        current_state.interactions = current_state.interactions + 1
        current_state.sync_ratio = current_state.sentiment / current_state.interactions
        logging.info(
            "Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {}	| Current State {}".format(
                str(current_state.responses),
                str(current_state.sentiment),
                str(current_state.sync_ratio),
                str(current_state.interactions),
                str(current_state.state_machine),
            )
        )

        if 5 <= current_state.sync_ratio <= 11:
            instance.next_state()
        else:
            print("Hey, lets stay friends")
            instance.friendzone()
            return
    # current_state.state_machine = pickle.dumps(instance)

def answer_question(body):
    instance = Saati(uuid.uuid4())

    sentiment = 1

    interactions = 1
    sync_ratio = sentiment / interactions

    logging.info("Computing reply")
    responce = ["Hello"]  # smallertalk(body)  # [0]
    # resp = MessagingResponse()
    current_state = Event(
        input=body,
        output=responce,
        sentiment=sentiment,
        sync_ratio=sync_ratio,
        interactions=interactions,
        state_machine=instance,
    )

    from pathlib import Path

    my_file = Path("event_log.dat")
    if my_file.is_file():
        save_state = pickle.load(open("event_log.dat", "rb"))
        pickled_state_machine = save_state.get("state_machine")
        state_machine = pickle.loads(pickled_state_machine)
        interactions = current_state.interactions
        print(interactions)

    sentiment = sentiment + compute_sentiment(body)
    interactions = interactions + 1

    logging.info(
        "Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {}	| Current State {}".format(
            str(responce),
            str(sentiment),
            str(sync_ratio),
            str(interactions),
            str(instance.state),
        )
    )
    dump = pickle.dumps(instance)

    save_state = {"state_machine": dump, "current_state": current_state.dict()}

    with open("event_log.dat", "wb") as file:
        data = pickle.dumps(save_state)
        file.write(data)

    # with open("save_state.json", "r+") as file:
    # 	 data = json.load(file)
    # 	 data.update(save_state)
    # 	 file.seek(0)
    # 	 json.dump(data, file)

    # my_dict = {'1': 'aaa', '2': 'bbb', '3': 'ccc'}

    if 5 >= sync_ratio <= 11 or interactions < 10:
        instance.next_state()
    else:
        instance.friendzone()

    return responce

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


""" def voice_ingest(model, scorer, sample_rate=16000, vad_aggressiveness=3):
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
            stream_context = model.createStream() """


def GivenCommand():
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
			#user_identifier = str(request.remote_addr)
            inference = answer_question(Input, user_identifier, 'local')
            #output = smalltalk(Input)
			#recipient = GivenCommand()
                        


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

