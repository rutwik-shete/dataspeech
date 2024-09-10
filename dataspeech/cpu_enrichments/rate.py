from g2p import make_g2p
from g2p.mappings import Mapping,Rule
from g2p.transducer import Transducer

hindi_mapping = Mapping(rules=[
    Rule(rule_input="अ", rule_output="ə"),
    Rule(rule_input="आ", rule_output="aː"),
    Rule(rule_input="इ", rule_output="ɪ"),
    Rule(rule_input="ई", rule_output="iː"),
    Rule(rule_input="उ", rule_output="ʊ"),
    Rule(rule_input="ऊ", rule_output="uː"),
    Rule(rule_input="ए", rule_output="eː"),
    Rule(rule_input="ऐ", rule_output="ɛː"),
    Rule(rule_input="ओ", rule_output="oː"),
    Rule(rule_input="औ", rule_output="ɔː"),
    Rule(rule_input="क", rule_output="k"),
    Rule(rule_input="ख", rule_output="kʰ"),
    Rule(rule_input="ग", rule_output="ɡ"),
    Rule(rule_input="घ", rule_output="ɡʱ"),
    Rule(rule_input="ङ", rule_output="ŋ"),
    Rule(rule_input="च", rule_output="tʃ"),
    Rule(rule_input="छ", rule_output="tʃʰ"),
    Rule(rule_input="ज", rule_output="dʒ"),
    Rule(rule_input="झ", rule_output="dʒʱ"),
    Rule(rule_input="ञ", rule_output="ɲ"),
    Rule(rule_input="ट", rule_output="ʈ"),
    Rule(rule_input="ठ", rule_output="ʈʰ"),
    Rule(rule_input="ड", rule_output="ɖ"),
    Rule(rule_input="ढ", rule_output="ɖʱ"),
    Rule(rule_input="ण", rule_output="ɳ"),
    Rule(rule_input="त", rule_output="t̪"),
    Rule(rule_input="थ", rule_output="t̪ʰ"),
    Rule(rule_input="द", rule_output="d̪"),
    Rule(rule_input="ध", rule_output="d̪ʱ"),
    Rule(rule_input="न", rule_output="n̪"),
    Rule(rule_input="प", rule_output="p"),
    Rule(rule_input="फ", rule_output="pʰ"),
    Rule(rule_input="ब", rule_output="b"),
    Rule(rule_input="भ", rule_output="bʱ"),
    Rule(rule_input="म", rule_output="m"),
    Rule(rule_input="य", rule_output="j"),
    Rule(rule_input="र", rule_output="ɾ"),
    Rule(rule_input="ल", rule_output="l"),
    Rule(rule_input="व", rule_output="ʋ"),
    Rule(rule_input="श", rule_output="ʃ"),
    Rule(rule_input="ष", rule_output="ʂ"),
    Rule(rule_input="स", rule_output="s"),
    Rule(rule_input="ह", rule_output="ɦ"),
    Rule(rule_input="ं", rule_output="̃"),
    Rule(rule_input="ँ", rule_output="̃"),
    Rule(rule_input="ः", rule_output="h"),
    Rule(rule_input="ै", rule_output="ɛː"),
    Rule(rule_input="ो", rule_output="oː"),
    Rule(rule_input="ा", rule_output="aː"),
    Rule(rule_input="े", rule_output="eː"),
    Rule(rule_input="ि", rule_output="ɪ"),
    Rule(rule_input="ू", rule_output="uː"),
    Rule(rule_input="ु", rule_output="ʊ")
])

transducer = Transducer(hindi_mapping)


def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    if isinstance(batch[text_column_name], list):  
        speaking_rates = []
        phonemes_list = []
        if "speech_duration" in batch:
            for text, audio_duration in zip(batch[text_column_name], batch["speech_duration"]):
                phonemes = transducer(text).output_string
                audio_duration = audio_duration if audio_duration != 0 else 0.01
                speaking_rate = len(phonemes) / audio_duration
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
        else:
            for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
                phonemes = transducer(text).output_string
                
                sample_rate = audio["sampling_rate"]
                audio_length = len(audio["array"].squeeze()) / sample_rate
                
                speaking_rate = len(phonemes) / audio_length

                
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
        
        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        phonemes = transducer(batch[text_column_name]).output_string
        if "speech_duration" in batch:
            audio_length = batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
        else:
            sample_rate = batch[audio_column_name]["sampling_rate"]
            audio_length = len(batch[audio_column_name]["array"].squeeze()) / sample_rate

        speaking_rate = len(phonemes) / audio_length
        
        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes

    return batch