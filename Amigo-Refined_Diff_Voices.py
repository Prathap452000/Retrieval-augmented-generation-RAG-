import streamlit as st
import pyttsx3
import speech_recognition as sr
import threading
import time

# Initialize the speech engine
engine = pyttsx3.init()

# Set voice options
voices = engine.getProperty('voices')
female_voice_id = voices[1].id  # Female voice (Beta)
male_voice_id = voices[0].id     # Male voice (Alpha)
selected_voice_id = female_voice_id  # Default voice is female

def speak_response(text):
    """Function to speak the given text."""
    engine.setProperty('voice', selected_voice_id)
    engine.say(text)
    engine.runAndWait()

def listen_for_wake_word():
    """Continuously listen for the wake word."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                print("Listening for the wake word...")
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                if "amigo" in command:  # Wake word
                    st.write("Wake word detected!")
                    speak_response("Hi, Hello, Namaskara, how can I assist you today?")
                    start_conversation()  # Start the conversation after greeting
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                st.write("Could not request results from the recognition service.")
                break

def start_conversation():
    """Handle the conversation flow."""
    global selected_voice_id
    while True:
        user_query = listen_for_user_query()
        if user_query:
            response = generate_response(user_query)
            speak_response(response)
            st.write(f"Amigo: {response}")

def listen_for_user_query():
    """Listen for user queries after the greeting."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for user query...")
        audio = recognizer.listen(source)
        try:
            user_query = recognizer.recognize_google(audio)
            return user_query
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            st.write("Could not request results from the recognition service.")
            return None

def generate_response(query):
    """Generate a response based on the user query."""
    # For demonstration purposes, we're just echoing the query.
    return f"I'm doing well, thank you. You said: {query}"

def switch_voice(voice):
    """Switch between male and female voices."""
    global selected_voice_id
    if voice == "alpha":
        selected_voice_id = male_voice_id
        speak_response("Voice switched to Alpha.")
    elif voice == "beta":
        selected_voice_id = female_voice_id
        speak_response("Voice switched to Beta.")

# Run the wake word listener in a separate thread
threading.Thread(target=listen_for_wake_word, daemon=True).start()

# Streamlit UI setup
st.title("Amigo V1.1.2")
st.write("Listening for wake word...")

