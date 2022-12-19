import sys
TTS_PATH = "TTS/"

# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally

import os
import string
import time
import argparse
import json

import numpy as np


import torch

from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor


from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *


OUT_PATH = 'out/'

# create output path
os.makedirs(OUT_PATH, exist_ok=True)

# model vars 
MODEL_PATH = 'best_model.pth.tar'
CONFIG_PATH = 'config.json'
TTS_LANGUAGES = "language_ids.json"
TTS_SPEAKERS = "speakers.json"
# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False

# load the config
C = load_config(CONFIG_PATH)


# load the audio processor
ap = AudioProcessor(**C.audio)

speaker_embedding = None

C.model_args['d_vector_file'] = TTS_SPEAKERS
C.model_args['use_speaker_encoder_as_loss'] = False

model = setup_model(C)
model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)
# print(model.language_manager.num_languages, model.embedded_language_dim)
# print(model.emb_l)
cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# remove speaker encoder
model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
  if "speaker_encoder" in key:
    del model_weights[key]

model.load_state_dict(model_weights)


model.eval()

if USE_CUDA:
    model = model.cuda()

# synthesize voice
use_griffin_lim = False

CONFIG_SE_PATH = "config_se.json"
CHECKPOINT_SE_PATH = "SE_checkpoint.pth.tar"

from TTS.tts.utils.speakers import SpeakerManager
from pydub import AudioSegment
import librosa

SE_speaker_manager = SpeakerManager(encoder_model_path=CHECKPOINT_SE_PATH, encoder_config_path=CONFIG_SE_PATH, use_cuda=USE_CUDA)

def compute_spec(ref_file):
  y, sr = librosa.load(ref_file, sr=ap.sample_rate)
  spec = ap.spectrogram(y)
  spec = torch.FloatTensor(spec).unsqueeze(0)
  return spec

  # 10/17 generate original conversions

import soundfile as sf
import os

root_dir = "."
out_dir = OUT_PATH
if not os.path.exists(out_dir):
  os.mkdir(out_dir)

conversion_pairs = {
    "scary_croc.wav" : "snape.wav",
    "scared_croc.wav" : "weasley.wav",
    "scary_dentist.wav" :  "voldemort.wav",
    "scared_dentist.wav" : "dobby.wav"
}

for source in conversion_pairs.keys():
    # define and preprocessing inputs
    driving_files = [os.path.join(root_dir, "inputs",  source)]
    target_files = [os.path.join(root_dir, "inputs", conversion_pairs[source])]
    driving_file = driving_files[0]


    # extract speaker embeddings
    target_emb = SE_speaker_manager.compute_d_vector_from_clip(target_files)
    target_emb = torch.FloatTensor(target_emb).unsqueeze(0)

    driving_emb = SE_speaker_manager.compute_d_vector_from_clip(driving_files)
    driving_emb = torch.FloatTensor(driving_emb).unsqueeze(0)

    # conversion
    driving_spec = compute_spec(driving_file)
    y_lengths = torch.tensor([driving_spec.size(-1)])
    if USE_CUDA:
        ref_wav_voc, _, _ = model.voice_conversion(driving_spec.cuda(), y_lengths.cuda(), driving_emb.cuda(), target_emb.cuda())
        ref_wav_voc = ref_wav_voc.squeeze().cpu().detach().numpy()
    else:
        ref_wav_voc, _, _ = model.voice_conversion(driving_spec, y_lengths, driving_emb, target_emb)
        ref_wav_voc = ref_wav_voc.squeeze().detach().numpy()

    # save audio
    name = os.path.join(out_dir, "_".join([source[:-4], "TO" ,os.path.basename(conversion_pairs[source])]))
    sf.write(name, ref_wav_voc, 16000)
    

# 10/26 genrate new conversions with samples recorded by Erqian
import soundfile as sf
import os

root_dir = "/content"
out_dir = os.path.join(root_dir, "converted")
if not os.path.exists(out_dir):
  os.mkdir(out_dir)

conversion_pairs = {
    "Crocodile1.wav" : "weasley.wav",
    "Crocodile2.wav" : "weasley.wav",
    "Dentist1.wav" : "dobby.wav",
    "Dentist2.wav" : "dobby.wav"
}

for source in conversion_pairs.keys():
    # define and preprocessing inputs
    driving_files = [os.path.join(root_dir, source)]
    target_files = [os.path.join(root_dir, conversion_pairs[source])]
    driving_file = driving_files[0]

    for sample in target_files:
        !ffmpeg-normalize $sample -nt rms -t=-27 -o $sample -ar 16000 -f
    for sample in driving_files:
        !ffmpeg-normalize $sample -nt rms -t=-27 -o $sample -ar 16000 -f
    !ffmpeg-normalize $driving_file -nt rms -t=-27 -o $driving_file -ar 16000 -f

    # extract speaker embeddings
    target_emb = SE_speaker_manager.compute_d_vector_from_clip(target_files)
    target_emb = torch.FloatTensor(target_emb).unsqueeze(0)

    driving_emb = SE_speaker_manager.compute_d_vector_from_clip(driving_files)
    driving_emb = torch.FloatTensor(driving_emb).unsqueeze(0)

    # conversion
    driving_spec = compute_spec(driving_file)
    y_lengths = torch.tensor([driving_spec.size(-1)])
    if USE_CUDA:
        ref_wav_voc, _, _ = model.voice_conversion(driving_spec.cuda(), y_lengths.cuda(), driving_emb.cuda(), target_emb.cuda())
        ref_wav_voc = ref_wav_voc.squeeze().cpu().detach().numpy()
    else:
        ref_wav_voc, _, _ = model.voice_conversion(driving_spec, y_lengths, driving_emb, target_emb)
        ref_wav_voc = ref_wav_voc.squeeze().detach().numpy()

    # save audio
    name = os.path.join(out_dir, "_".join([source[:-4], "TO" ,os.path.basename(conversion_pairs[source])]))
    sf.write(name, ref_wav_voc, 16000)


# 12/2 generate Chinese conversion, using google translate and my own recordings

import soundfile as sf
import os

conversion_pairs = {
    "scary_croc_CN.wav" : "snape.wav",
    "scared_croc_CN.wav" : "weasley.wav",
    "scary_doc_CN.wav" :  "voldemort.wav",
    "scared_doc_CN.wav" : "dobby.wav"
}

for source in conversion_pairs.keys():
    # define and preprocessing inputs
    driving_files = [os.path.join(root_dir, "inputs",  source)]
    target_files = [os.path.join(root_dir, "inputs", conversion_pairs[source])]
    driving_file = driving_files[0]


    # extract speaker embeddings
    target_emb = SE_speaker_manager.compute_d_vector_from_clip(target_files)
    target_emb = torch.FloatTensor(target_emb).unsqueeze(0)

    driving_emb = SE_speaker_manager.compute_d_vector_from_clip(driving_files)
    driving_emb = torch.FloatTensor(driving_emb).unsqueeze(0)

    # conversion
    driving_spec = compute_spec(driving_file)
    y_lengths = torch.tensor([driving_spec.size(-1)])
    if USE_CUDA:
        ref_wav_voc, _, _ = model.voice_conversion(driving_spec.cuda(), y_lengths.cuda(), driving_emb.cuda(), target_emb.cuda())
        ref_wav_voc = ref_wav_voc.squeeze().cpu().detach().numpy()
    else:
        ref_wav_voc, _, _ = model.voice_conversion(driving_spec, y_lengths, driving_emb, target_emb)
        ref_wav_voc = ref_wav_voc.squeeze().detach().numpy()

    # save audio
    name = os.path.join(out_dir, "_".join([source[:-4], "TO" ,os.path.basename(conversion_pairs[source])]))
    sf.write(name, ref_wav_voc, 16000)
    