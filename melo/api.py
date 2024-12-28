import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch

from . import utils
from . import commons
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .mel_processing import spectrogram_torch, spectrogram_torch_conv
from .download_utils import load_or_download_config, load_or_download_model
from .text import symbols, language_tone_start_map

class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_iter(self, text, speaker_id, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, position=None, quiet=False, seed=1234):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        current_frame = 0
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids, word2ph, norm_text = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones_d = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                speakers = torch.LongTensor([speaker_id]).to(device)
                out = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones_d,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                        seed=seed,
                    )
                
                audio = out[0][0, 0].data.cpu().float().numpy()
                attn = out[1][0, 0].data.cpu().float().numpy()
                del x_tst, tones_d, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # 
            # Find position of first non-zero entry for each column (start time for phone)
            phones_start = ((attn != 0).argmax(axis=0) + current_frame).tolist()
            current_frame += attn.shape[0]
            audio = utils.fix_loudness(audio, self.hps.data.sampling_rate)
            time_per_frame = self.hps.data.hop_length / self.hps.data.sampling_rate
            # Round start times to nearest millisecond (keeps as integers and has enough precision)
            phones_start_time = [round(t * time_per_frame * 1000) for t in phones_start]
            renormed_tones = tones.tolist()
            # Measure tones relative to language start
            renormed_tones = [tn - language_tone_start_map[language] if tn > 0 else tn for tn in renormed_tones]
            phones_text = [symbols[i] for i in phones.tolist()]
            # Compute a vector same size as phonemes that is word number, from word2ph which is phoneme lengths for words
            word_index = [[i - 1] * count for i, count in enumerate(word2ph)]
            # Flatten to go from [[0, 0, 0], [1], [2, 2, 2]] to [0, 0, 0, 1, 2, 2, 2]
            word_index = [x for xs in word_index for x in xs]
            # # Add numeric digits for tones that are not 0
            # phones_text = [f'{phones_text[i]}{str(renormed_tones[i]) if renormed_tones[i] > 0 else ''}' for i in range(len(phones_text))]
            filtered_phones = [{ 'phoneme': p, 'tone': tn, 'time': t, 'word': w } for (p, tn, t, w) in zip(phones_text, renormed_tones, phones_start_time, word_index) if p != '_']
            metadata = {
                'text': text,
                'norm_text': norm_text,
                'speaker_id': speaker_id,
                'sdp_ratio': sdp_ratio,
                'noise_scale': noise_scale,
                'noise_scale_w': noise_scale_w,
                'seed': seed,
                'speed': speed,
                'samplerate': self.hps.data.sampling_rate,
                'phonemes': filtered_phones,
            }

            yield audio, metadata

        torch.cuda.empty_cache()

    def tts_to_file(self, text, speaker_id, output_path=None, metadata_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False, seed=1234):
        audio_list = []
        metadata_result = None
        phones_text_list = []
        phones_start_time_list = []
        for audio, metadata in self.tts_iter(text, speaker_id, sdp_ratio, noise_scale, noise_scale_w, speed, pbar, position, quiet, seed=seed):
            audio_list.append(audio)
            if metadata_result is None:
                metadata_result = metadata
            else:
                metadata_result['phonemes'].extend(metadata['phonemes'])

        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if metadata_path is not None:
            metadata_path.write(json.dumps(metadata, indent=4))

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
