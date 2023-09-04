from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to("cuda")

def generate_audio(text, preset, output):
    inputs = processor(text, voice_preset = preset)

    for k,v in inputs.items():
        inputs[k] = v.to("cuda")

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavefile.write(output, rate = sample_rate, data = audio_array)

# def generate_audio(text, preset, output):
#     inputs = processor(text, voice_preset=preset)

#     audio_array = model.generate(**inputs)
#     audio_array = audio_array.squeeze().numpy()
#     sample_rate = model.config.sample_rate
#     scipy.io.wavfile.write(output, rate=sample_rate, data=audio_array)

generate_audio(text="Hi, This is Shahzain. How may I help you?",
    preset = "v2/en_speaker_6",
    output="output.wav")