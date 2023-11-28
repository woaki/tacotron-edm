import random
import sys
import torch
import os
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm

from hparams import create_hparams
from model import TacotronSTFT  # STFT
from tacotron2 import Tacotron2
from text import text_to_sequence
from utils import load_wav_to_torch

sys.path.append("waveglow")
from waveglow.denoiser import Denoiser


def get_ref_mel(filepath, hparams):
    stft = TacotronSTFT(hparams.filter_length,
                        hparams.hop_length,
                        hparams.win_length,
                        sampling_rate=hparams.sampling_rate)
    audio, sampling_rate = load_wav_to_torch(filepath)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, stft.sampling_rate))
    melspec = stft.mel_spectrogram(torch.autograd.Variable(audio.unsqueeze(0), requires_grad=False))
    melspec = torch.squeeze(melspec, 0)

    return melspec


def load_models(hparams):
    tacotron2_pth = hparams.etts_ada_pth
    waveglow_pth = hparams.waveglow_checkpoint_path
    # load tacotron2
    tacotron2 = Tacotron2(hparams).cuda()
    tacotron2.load_state_dict(torch.load(tacotron2_pth)['state_dict'], strict=True)
    tacotron2.cuda().eval()
    # load waveglow
    waveglow = torch.load(waveglow_pth)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    denoiser = Denoiser(waveglow).cuda()

    return tacotron2, waveglow, denoiser


# synthesis & inference
def synthesis(tacotron2, waveglow, denoiser, Text, Ref_audio_path, Spk_id, Wav_path):
    hparams = create_hparams()
    hparams.sampling_rate = 16000

    # text input
    sequence = np.array(text_to_sequence(Text, ['mandarin_cleaners']))[None, :]
    text_input = torch.from_numpy(sequence).cuda().long()

    # speaker
    speaker_id = torch.IntTensor([Spk_id]).cuda().long()
    ref_mel = torch.FloatTensor(get_ref_mel(Ref_audio_path, hparams)).cuda().float()

    # outputs
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.inference(text_input, speaker_id, ref_mel, cg=False)
        audio = waveglow.infer(mel_outputs_postnet, sigma=1)
        audio = denoiser(audio, strength=0.3)  # denoise

    audio = audio.squeeze()
    audio = audio.cpu().numpy()

    audio_path = os.path.join(Wav_path)
    write(audio_path, hparams.sampling_rate, audio)


if __name__ == '__main__':
    # griffin limiters
    steps = 60
    # text to infer
    text = "wo3 men5 lia3 he2 bu4 lai2, hai2 jing1 chang2 chao3 jia4."
    hparams = create_hparams()
    taco, glow, de = load_models(hparams)

    for spk_id in tqdm(range(1, 11)):
        # spk_id = 10
        # spk_id = str(spk_id).zfill(4)
        cur = str(random.randint(1, 10)).zfill(4)
        ref_path = [
            r'D:\Datasets\ESD\%s\Angry\train\%s_000445.wav' % (cur, cur),
            r'D:\Datasets\ESD\%s\Happy\train\%s_000805.wav' % (cur, cur),
            r"D:\Datasets\ESD\%s\Surprise\train\%s_001505.wav" % (cur, cur),
            r'D:\Datasets\ESD\%s\Neutral\train\%s_000105.wav' % (cur, cur),
            r'D:\Datasets\ESD\%s\Sad\train\%s_001155.wav' % (cur, cur)
        ]
        # spk_id = str(int(spk_id))
        spk_id = str(spk_id)
        for path in ref_path:
            for i in range(0, 3):
                emotion = path.split('\\')[4].lower()
                wav_name = 'spk{}_{}{}.wav'.format((str(spk_id)).zfill(2), emotion, i)
                wave_path = 'D://backup//etts//etts_ada//demo//nonparallel//%s//%s' % (spk_id.zfill(3), emotion)
                wave_path = os.path.join(wave_path, wav_name)

                synthesis(taco, glow, de, Text=text, Ref_audio_path=path, Spk_id=int(spk_id) - 1, Wav_path=wave_path)
