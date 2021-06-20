# Near Simplest Language model API, with room to expand!
# runs GPT-J-6B on 3090 and TITAN and servers it using FastAPI
# change "seq" (which is the context size) to adjust footprint
#
# seq   vram usage
# 512   14.7G
# 900   15.3G

# uses FastAPI, so install that
# https://fastapi.tiangolo.com/tutorial/
#   pip install fastapi
#   pip install uvicorn[standard]

# uses https://github.com/kingoflolz/mesh-transformer-jax

# so install jax on your system so recommend you get it working with your GPU first


# !apt install zstd

# the "slim" version contain only bf16 weights and no optimizer parameters, which minimizes bandwidth and memory
# wget https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd

# tar -I zstd -xf step_383500_slim.tar.zstd

# git clone https://github.com/kingoflolz/mesh-transformer-jax.git
# pip install -r mesh-transformer-jax/requirements.txt

# jax 0.2.12 is required due to a regression with xmap in 0.2.13
# pip install mesh-transformer-jax/ jax==0.2.12

# I have cuda 10.1 and python 3.9 so had to update
# pip3 install --upgrade "https://storage.googleapis.com/jax-releases/cuda101/jaxlib-0.1.66+cuda101-cp39-none-manylinux2010_x86_64.whl"

# GO: local execution
# XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform CUDA_VISIBLE_DEVICES=0 python3 jserv.py

# When done try
# http://localhost:8000/docs#/default/read_completions_engines_completions_post

# now you are in FastAPI + EleutherAI land
# note: needs async on the read_completions otherwise jax gets upset
# remember to adjust the location of the checkpoint image

import argparse
import time
from datetime import datetime
from typing import Optional
from typing import Dict
from fastapi import FastAPI
import uvicorn

import os
import requests 
import threading 
import uuid

import jax
from jax.experimental import maps
from jax.config import config
import numpy as np
import optax
import transformers

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

app = FastAPI()
params = {
  "layers": 28,
  "d_model": 4096,
  "n_heads": 16,
  "n_vocab": 50400,
  "norm": "layernorm",
  "pe": "rotary",
  "pe_rotary_dims": 64,
  "early_cast": True,
  "seq": 768,
  "cores_per_replica": 1,
  "per_replica_batch": 1,
}

#>> INFO <<: adjust the location of the checkpoint image
check_point_dir="./step_383500/"

per_replica_batch = params["per_replica_batch"]
cores_per_replica = params["cores_per_replica"]
seq = params["seq"]


params["sampler"] = nucleaus_sample

# here we "remove" the optimizer parameters from the model (as we don't need them for inference)
params["optimizer"] = optax.scale(0)

print("jax.device_count ",jax.device_count())
print("jax.devices ",jax.devices())
print("cores_per_replica ",cores_per_replica)



mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
#devices = np.array(jax.devices()).reshape(mesh_shape)
devices = np.array([jax.devices()[0]]).reshape((1, 1))

maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')


total_batch = per_replica_batch * jax.device_count() // cores_per_replica
print("CausalTransformer")
network = CausalTransformer(params)



#here we load a checkpoint which was written with 8 shards into 1 shard
print("read_ckpt")
network.state = read_ckpt(network.state, check_point_dir,8,shards_out=cores_per_replica)

#network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))
#move the state to CPU/system memory so it's not duplicated by xmap
network.state = jax.device_put(network.state, jax.devices("cpu")[0])

def infer(context,top_k=40, top_p=0.9, temp=1.0, gen_len=512):
	global network

	start = time.time()
	tokens = tokenizer.encode(context)

	provided_ctx = len(tokens)
	pad_amount = seq - provided_ctx

	padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
	batched_tokens = np.array([padded_tokens] * total_batch)
	length = np.ones(total_batch, dtype=np.uint32) * len(tokens)
	
	start = time.time()
	#output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(total_batch) * top_p, "temp": np.ones(total_batch) * temp})
	#output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(per_replica_batch) * top_p, "temp": np.ones(per_replica_batch) * temp})
	output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(per_replica_batch) * top_p, "top_k": top_k is not None and (np.ones(per_replica_batch, dtype=np.int32) * top_k) or None, "temp": np.ones(per_replica_batch) * temp})
	samples = []
	decoded_tokens = output[1][0]

	for o in decoded_tokens[:, :, 0]:
		samples.append(tokenizer.decode(o))

	print(f"completion done in {time.time() - start:06}s")
	return samples

def recursive_infer(initial_context, current_context=None, top_k=40, top_p=0.9, temp=1.0, gen_len=512, depth=0, max_depth=5,recursive_refresh=0):
  lcc=0
  if current_context : 
    lcc = len(current_context)
  print ("recursive_infer:{} {} {} {}".format(len(initial_context),lcc,depth,max_depth))
  
  c=''
  if not current_context :
    c = initial_context
  else:
    if (recursive_refresh == 1):
      c= initial_context + "\r\n ... \r\n"
    c = c + current_context
    
  print ("cc:{}".format(c))
  i = infer(c, top_k, top_p, temp, gen_len)[0]
  #yield i[len(c):]
  yield i
  if depth >= max_depth: return
  yield from recursive_infer(initial_context, i,top_k, top_p, temp, gen_len, depth+1, max_depth)
    
print("PRETEST")
#warms up the processing on startup
pre_prompt = "I am the EleutherAI / GPT-J-6B based AI language model server. I will"
print (pre_prompt)
print(infer(pre_prompt)[0])

print("SERVER SERVING")


@app.post("/engines/completions")
async def read_completions(
#engine_id:str,
		prompt:Optional[str] = None,
		max_tokens: Optional[int]=16,
		temperature: Optional[float]=1.0,
		top_p:Optional[float]=1.0,
		top_k:Optional[int]=40,
		n:Optional[int]=1,
		stream:Optional[bool]=False,
		logprobs:Optional[int]=None,
		echo:Optional[bool]=False,
		stop:Optional[list]=None,
		presence_penalty:Optional[float]=0.0001,
		frequency_penalty:Optional[float]=0.0001,
		best_of:Optional[int]=1,
                recursive_depth:Optional[int]=0,
                recursive_refresh:Optional[int]=0,
		logit_bias:Optional[Dict[str,float]]=None
    ):
	
    text = str(prompt)
    text = text.replace("|","\r\n")
    prompt_len = len(text)	
    #ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    tokens = tokenizer.encode(text)
    max_length = max_tokens + len(tokens)
    do_sample=True
    use_cache=True
    start = time.time()
    num_return_sequences=n
    num_beams = n
    num_beam_groups=n

    mydata = threading.local()
    mydata.env=None
    if (recursive_depth== 0):
        gtext= infer(context=text, top_p=top_p,top_k=top_k, temp=temperature, gen_len=max_length)
    else:
        gtext = recursive_infer(initial_context=text,current_context=None, top_p=top_p,top_k=top_k, temp=temperature, gen_len=max_length,  depth=0, max_depth = recursive_depth,recursive_refresh=recursive_refresh)
        
    last_prompt=text
    choices=[]
    gen_text=''
    for i,out_seq in enumerate(gtext):
        choice={}
        choice['prompt']=last_prompt
        
        choice['text']=out_seq
        choice['index']=i
        choice['logprobs']=None
        choice['finish_reason']='length'
        choices.append(choice)
        print("GenText[{}]:{}".format(i,choice['text']))
        gen_text = gen_text + choice['text']
        if (recursive_depth==0):
          last_prompt = text
        else:
          last_prompt = out_seq
          if (recursive_refresh==1):
            last_prompt = text +"\r\n ... \r\n"+out_seq
          
    
    #gen_text = tokenizer.batch_decode(gen_tokens)[0]
    fin = time.time()
    elapsed = fin - start
    cps = (len(gen_text)-prompt_len) / elapsed

    print("elapsed:{} len:{} cps:{}".format(elapsed,len(gen_text),cps))
    
    response={}
    response['id']=str(uuid.uuid4())
    response['object']='text_completion'
    response['created']=datetime.now()
    response['model']= 'GPT-J-6B' #args.model
    response['choices']=choices
    
    
   
    return(response)

#if __name__ == "__main__":
uvicorn.run(app, host="0.0.0.0", port=8000)
print ("Happy Service!")
