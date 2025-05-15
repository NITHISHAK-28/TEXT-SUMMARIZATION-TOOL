import speech_recognition as sr

def recognize_speech(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)  # Read the entire audio file

    try:
        # Uses Google Web Speech API (requires internet)
        text = recognizer.recognize_google(audio)
        print("Transcription:", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

# Example usage
recognize_speech("sample.wav")  # Replace with your .wav file
