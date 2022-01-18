import speech_recognition as sr

r = sr.Recognizer()

recording = sr.AudioFile('harvard.wav')
with recording as source:
    audio = r.record(source)

text = r.recognize_google(audio)
print(text)
