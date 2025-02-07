import warnings
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import os
import pygame
from gtts import gTTS
import tempfile
import torch
import datetime
import requests
import speech_recognition as sr

# Suppress FutureWarning for tokenization spaces
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Load the model and tokenizer globally to avoid reloading every time
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Function to capture speech
def capture_speech(mic_index):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # Lower energy threshold for faster detection
    mic = sr.Microphone(device_index=mic_index)

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
        except sr.WaitTimeoutError:
            return None

    try:
        user_input = recognizer.recognize_google(audio)
        return user_input
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None

# Handle time, date, and location queries
def handle_time_date_queries(user_input):
    if "time" in user_input:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}."
    elif "date" in user_input or "today" in user_input:
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {current_date}."
    elif "location" in user_input or "where am i" in user_input:
        return get_current_location()
    else:
        return None

# Get current location
def get_current_location():
    try:
        response = requests.get("http://ip-api.com/json")
        data = response.json()
        if data['status'] == 'success':
            city = data.get('city', 'unknown city')
            region = data.get('regionName', 'unknown region')
            country = data.get('country', 'unknown country')
            return f"You are currently in {city}, {region}, {country}."
        else:
            return "Sorry, I could not determine your location."
    except Exception as e:
        return f"Error fetching location: {str(e)}"

# Select microphone
def select_microphone():
    mic_list = sr.Microphone.list_microphone_names()
    index = 2
    return index

# Generate response
def generate_response(user_input):
    if "your name" in user_input.lower():
        return "I am Amigo, your all-weather conversational companion, just like the bike that you are riding on."
    
    time_date_response = handle_time_date_queries(user_input)
    if time_date_response:
        return time_date_response

    if any(keyword in user_input.lower() for keyword in ["what is","who is", "when is", "where is", "current", "today", "news", "capital", "places"]):
        return "Thank you for your query, but that's out of my scope of training."

    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=150)
    attention_mask = inputs['attention_mask']
    reply_ids = model.generate(inputs['input_ids'], max_length=150, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(reply_ids[0], skip_special_tokens=True, attention_mask=attention_mask, clean_up_tokenization_spaces=True)
    
    return bot_response

# Listen for wake word
def listen_for_wake_word(mic_index):
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=mic_index)
    
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source)
            
        try:
            user_input = recognizer.recognize_google(audio)
            if "hey amigo" in user_input.lower():
                return 
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            continue

# Speak response
def speak_response(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        tts.save(temp_audio_file.name)
        temp_file_path = temp_audio_file.name
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(temp_file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)

    except Exception as e:
        st.write(f"Error playing audio: {e}")
    finally:
        pygame.mixer.quit()
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Main Streamlit app
def main():
    st.title("Amigo: Your Conversational Companion")

    # Select microphone
    mic_index = select_microphone()
    
    st.write("Waiting for wake word 'Hey Amigo'...")

    if listen_for_wake_word(mic_index):
        st.write("Wake word detected!")
        greeting = "Hi Hello Namaskara, I am Amigo. How can I help you today?"
        st.write(f"Amigo: {greeting}")
        speak_response(greeting)

        while True:
            user_input = capture_speech(mic_index)

            if user_input is None:
                continue

            if "exit" in user_input.lower() or "goodbye" in user_input.lower():
                farewell_message = "Goodbye, have a nice day!"
                st.write(f"Amigo: {farewell_message}")
                speak_response(farewell_message)
                break
            
            bot_response = generate_response(user_input)
            st.write(f"Amigo: {bot_response}")
            speak_response(bot_response)

if __name__ == "__main__":
    main()
