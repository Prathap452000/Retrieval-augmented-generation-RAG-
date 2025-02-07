import warnings
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import os
import pygame
from gtts import gTTS
import speech_recognition as sr
import tempfile
import torch
import datetime                                                                                 
import requests
import random  # Import for random jokes/quotes
import time

from Amigo_Refined_Web_d import handle_time_date_queries  # Import for reminders

# Suppress FutureWarning for tokenization spaces
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

model_name = "facebook/blenderbot-400M-distill" # Larger model with the same functional capabilities facebook/blenderbot-3B # Qwen/Qwen2.5-1.5B-Instruct
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
WEATHER_API_KEY = '6375f15162952d140f9908db1d7ffb99'

def select_microphone():
    mic_index = 2  # Automatically select microphone index 2
    print(f"Using microphone with index: {mic_index}")
    return mic_index
def capture_speech(mic_index):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # Lower energy threshold for faster detection
    mic = sr.Microphone(device_index=mic_index)
    
    with mic as source:
        print("Listening for your query...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

        except sr.WaitTimeoutError:
            print("Listening timed out, please speak again.")
            return None  

    try:
        user_input = recognizer.recognize_google(audio)
        print(f"User said: {user_input}")  
        return user_input
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Error with the Google Speech Recognition service; {e}")
        return None

# Llama-3.2-11B-Vision-Instruct(model output generation and the implementation)

# # Step 1: Import necessary libraries
# from transformers import LlamaTokenizer, LlamaForCausalLM
# import torch

# # Step 2: Use a publicly available lighter LLaMA model to prevent kernel crashes
# model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # Lighter LLaMA model

# # Step 3: Load the tokenizer and model from Hugging Face, without legacy issues
# tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=False)  # Set legacy=False to avoid warning
# model = LlamaForCausalLM.from_pretrained(model_name)

# # Step 4: Define function to generate response using the LLaMA model
# def generate_response(user_input):
#     inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=150, padding_token=eos_token)
#   attention_mask = inputs['attention_mask']
    
#     # Generate response with attention mask
#     output = model.generate(inputs.input_ids, attention_mask=attention_mask, max_length=150)
#     bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
#     return bot_response

# # Dialogpt for Conversational as the key task and its end to end implementation
# def generate_response(input_text):
#     # Load a pre-trained conversational model from Hugging Face
#     conversation_pipeline = pipeline("conversational", model="microsoft/DialoGPT-medium")
    
#     if input_text.strip() == "":
#         return "I couldn't understand you. Could you please repeat?"
    
#     response = conversation_pipeline(input_text)
#     generated_response = response[0]['generated_text']
#     print(f"Amigo said: {generated_response}")
    
#     return generated_response

# # Blenderbot Large model(facebook/blenderbot-3B) end to end implementation for both text generation and conversational

# # Track conversational context
# conversation_history = []

# def generate_response(user_input):
#     model_name = "facebook/blenderbot-3B"  # Switched to a larger, more capable model
#     tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
#     model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    
#     # Add user input to conversation history
#     conversation_history.append(user_input)
    
#     # Join conversation history to provide context
#     conversation_context = " ".join(conversation_history[-5:])  # Keeping the last 5 exchanges
    
#     # Adjust parameters for better logic and factual accuracy
#     inputs = tokenizer(conversation_context, return_tensors="pt", truncation=True, max_length=128)
#     reply_ids = model.generate(
#         inputs['input_ids'],
#         max_length=512, 
#         pad_token_id=tokenizer.eos_token_id,
#         temperature=0.7,  # Lower temperature for more logical responses
#         num_beams=5,  # Beam search for more precise responses
#         no_repeat_ngram_size=2  # Avoid repetition
#     )
    
#     bot_response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    
#     # Add bot response to the conversation history
#     conversation_history.append(bot_response)
    
#     return bot_response

# Lighter or more distilled(optimized version) dialogpt-medium end-to-end implementation for faster response time(reduced latency and inference)

# def generate_response(input_text):
#     Load a pre-trained conversational model from Hugging Face
#     conversation_pipeline = pipeline("conversational", model="microsoft/DialoGPT-medium")
    
#     if input_text.strip() == "":
#         return "I couldn't understand you. Could you please repeat?"
    
#     response = conversation_pipeline(input_text)
#     generated_response = response[0]['generated_text']
#     print(f"Amigo said: {generated_response}")
    
#     return generated_response

# Lighter and distilled conversational model end-to-end implementation
# # Load the DistilGPT model and tokenizer
# model_name = "distilgpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Add padding token
# tokenizer.pad_token = tokenizer.eos_token


# Lighter and distilled conversational model end-to-end implementation
# # Load the DistilGPT model and tokenizer
# model_name = "distilgpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Add padding token
# tokenizer.pad_token = tokenizer.eos_token

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

def get_current_weather():
    location = get_current_location()
    if "in" in location:
        city = location.split("in ")[1].split(",")[0]
        try:
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
            response = requests.get(weather_url)
            data = response.json()
            if data['cod'] == 200:
                weather_description = data['weather'][0]['description']
                temperature = data['main']['temp']
                return f"The current weather in {city} is {weather_description} with a temperature of {temperature}°C."
            else:
                return "Sorry, I couldn't get the weather information."
        except Exception:
            return "Error fetching weather."
    else:
        return "I couldn't determine the location for weather."
    
reminders=[];
def add_reminder(time_str, message_str):
    global reminders
    reminders_list = message_str.split('and')
    
    for reminder_message in reminders_list:
        reminder_message = reminder_message.strip()  
        if reminder_message:
            formatted_reminder = f"'{reminder_message}' at {time_str}"  
            reminders.append(formatted_reminder)
    
    return f"Added {len(reminders_list)} reminder(s): {', '.join([f'{msg.strip()} at {time_str}' for msg in reminders_list])}."

def list_reminders():
    if not reminders:
        return "You have no reminders set."
    
    reminder_str = ', '.join(reminders)
    return f"Your reminders are: {reminder_str}."

def delete_all_reminders():
    global reminders
    reminders.clear()
    return "All reminders have been deleted."

def check_reminders():
    current_time = datetime.datetime.now().strftime("%H:%M")
    reminders_to_return = []
    for reminder in reminders:
        if reminder['time'] == current_time:
            reminders_to_return.append(reminder['message'])
    return reminders_to_return if reminders_to_return else None

def handle_calculator_query(user_input):
    try:
        expression = user_input.lower().replace('calculate', '').strip()
        result = eval(expression)
        return f"The result is {result}."
    except Exception as e:
        return "There was an error with your calculation."
jokes = [
    "Parallel lines have so much in common. Too bad they’ll never meet."
    "Why don't scientists trust atoms? Because they make up everything!",
    "A snowman crossed with a vampire? Frostbite!",
    "Did you hear about the scarecrow who won an award? Apparently, he was outstanding in his field!",
    "Someone’s wife was drawing her eyebrows too high. She looked really surprised!",
    "What’s orange and sounds like a parrot? A carrot!",
    "Skeletons never fight each other. They just don’t have the guts.",
    "Giving Elsa a balloon is a bad idea. She’ll just let it go!",
    "Parallel lines have so much in common. Too bad they’ll never meet.",
    "Some couples avoid the gym. Guess some relationships just don’t work out.",
    "The cheese factory explosion? Nothing left but de-brie.",
    "A musician used to play piano by ear but now uses hands. Much easier.",
    "Ever see a bicycle fall over? It was two-tired!",
    "Someone beat their addiction to the hokey pokey by turning themselves around.",
    "If a nose were 12 inches long, it would be a foot!",
    "Did you hear about the kidnapping at the playground? They woke up!",
    "The bank robber got caught after he went camping. Apparently, he stole some tents.",
    "A steak went to the party, but nobody wanted it there. Too rare.",
    "The elevator asked for a promotion. It thought it was due for a lift.",
    "Why did the computer break up with the internet? Too many connections!",
    "A sandwich tried to make friends at the bar, but the bartender told it, 'We don’t serve food here.'",
    "The calendar was so popular... it had a lot of dates!",
    "Why did the golfer bring an extra pair of pants? In case of a hole-in-one!"
   "A guy’s bank called him to say his balance was outstanding. Turns out, they meant his account was empty!",
    "A scarecrow went on a date. Apparently, he was outstanding in his field.",
    "He tried to make a belt out of watches. Total waste of time.",
    "Did you hear about the coffee shop? It’s now offering espresso yourself classes!",
    "A chicken’s dream job? Working in the egg-sport industry.",
    "When she asked the calendar out, he said he was already booked.",
    "The ocean waved at everyone. Typical, it’s always full of itself.",
    "The bakery opened a new branch. Guess you could say business is on the rise.",
    "Saw a documentary on beavers. Best dam movie ever!",
    "Someone tried to catch some fog yesterday. Missed it.",
    "Two Wi-Fi signals got married. The reception was excellent.",
    "A cucumber walked into a bar and got pickled.",
    "If two wrongs don’t make a right, why do two Wrights make an airplane?",
    "Reading a book on anti-gravity? Hard to put down!",
    "The airport called—it said planes are running late because they’re grounded.",
    "The shovel was voted the most groundbreaking invention.",
    "A termite walks into a bar and asks, 'Is the bartender here?'",
    "A tortilla’s dream job? Well, it’s on a roll.",
    "A guy gave all his dead batteries away—free of charge.",
    "Someone stole his Microsoft Office license. He said he’d Excel in finding them.",
    "That kleptomaniac who stole my coffee? Really mugs me off.",
    "A comedian found his biggest fans in the attic—just a bunch of old ceiling fans!",
    "A guy broke his arm trying to catch a baseball. Guess it wasn’t his lucky catch!",
    "The butcher brought his cow to work... talk about a rare situation!"
    "I was going to tell you a joke about a broken pencil, but it’s pointless."
]

def tell_joke():
    if jokes:
        return jokes.pop(0)
    else:
        return "I’ve run out of jokes!"

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
            pygame.time.wait(50)  # Check every 50ms for faster termination

    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        pygame.mixer.quit()
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

import random  # Import for random fillers

# Define fillers for different types of responses
fillers = {
    'greeting': [
        "Hey there, let's dive into this!",
        "Oh, you're asking about that? Let me tell you!",
        "Alright, here we go! Here's what I think...",
        "Hmm, interesting question! Let me check that for you."
    ],
    'time': [
        "Tick-tock! It's Time to check the clock... it's...",
        "Let's check the time! It's...",

    ],
    'date': [
        "Let me check the calendar... Today is...",
        "It’s... just another fabulous day! The date is...",
        "Hold on, let me dig into the calendar... Today is..."
    ],
    'location': [
        "I am trotting the globe to see where you are...",
        "Aha, I see where you are! You are in...",
        "Let me see where you are...",
    ],
    'weather': [
        "Checking the skies for you right now...",
        "Let me see what the weather gods have to say... It's...",
        
    ],
    
    'joke': [
        "Ready for a laugh? Here’s a joke for you...",
        "I’ve got a funny one for you...",
        "I’ve got something funny to share, check this out..."
    ],
     'general': [
        "Sure, let me check.",
        "Hmm, let me think.",
        "Just a moment...",
        "Sure,",

    ],
    'name': [
        "Oh, you're curious about me? Well, I'm Amigo, your loyal conversational companion!",
        "Nice to meet you! I’m Amigo, always here to assist.",
        "Ah, you're wondering who I am? I’m Amigo, I’m here to chat and help out whenever you need...Cheers to life",
        "You can call me Amigo! I’m here to chat and help out whenever you need."
    ]
}

def get_filler(query_type):
    # Pick a random filler from the appropriate category
    return random.choice(fillers.get(query_type, fillers['general']))

def generate_response(user_input):
    user_input = user_input.lower()

    # Handle name-related queries with engaging fillers
    if "what is your name" in user_input or "who are you" in user_input or "what's your name" in user_input:
        return get_filler('name') 
    # Handle reminders
    if "reminder" in user_input:
        if "set" in user_input:
            try:
                parts = user_input.split("for")
                time = parts[1].split("to")[0].strip() if len(parts) > 1 else ""
                messages = parts[1].split("to")[1].strip() if len(parts) > 1 else ""

                if time and messages:
                    return get_filler('reminder') + add_reminder(time, messages)
                else:
                    return get_filler('general') + " Please specify the reminder time and message correctly."
            except:
                return get_filler('general') + " Please specify the reminder time and message correctly."
        elif "list" in user_input:
            return get_filler('reminder') + list_reminders()
        elif "delete all" in user_input or "remove all" in user_input:
            return get_filler('general') + delete_all_reminders()
    
    # Check reminders before generating response
    reminder_notifications = check_reminders()
    if reminder_notifications:
        return get_filler('reminder') + " You have the following reminders: " + ', '.join(reminder_notifications)

    # Handle calculator queries
    if "calculate" in user_input:
        return get_filler('general') + handle_calculator_query(user_input)

    # Handle jokes
    if "joke" in user_input:
        return get_filler('joke') + tell_joke()

    # Handle time and date queries
    time_date_response = handle_time_date_queries(user_input)
    if time_date_response:
        if "time" in user_input:
            return get_filler('time') + time_date_response
        elif "date" in user_input or "today" in user_input:
            return get_filler('date') + time_date_response

    # Handle weather query
    if "weather" in user_input or "forecast" in user_input:
        return get_filler('weather') + get_current_weather()

    # Handle location queries
    if "location" in user_input or "where am i" in user_input:
        return get_filler('location') + get_current_location()

    # Tokenize user input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=150)
    attention_mask = inputs['attention_mask']

    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    
    reply_ids = model.generate(inputs['input_ids'], max_length=150, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(reply_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return get_filler('general') + bot_response

def listen_for_wake_word(mic_index):
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=mic_index)

    print("Waiting for wake word 'Hey Amigo'...")
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source)
            
        try:
            user_input = recognizer.recognize_google(audio)
            if "hey amigo" in user_input.lower():
                print("Wake word detected!")
                return True
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            print(f"Error with the Google Speech Recognition service: {e}")
            continue
def amigo_conversational_companion():
    mic_index = select_microphone()

    if listen_for_wake_word(mic_index):
        greeting = "Hi Hello Namaskara, How can I assist you today?"
        print(f"Amigo: {greeting}")
        speak_response(greeting)

        while True:
            user_input = capture_speech(mic_index)
            if user_input is None:
                continue

            if "exit" in user_input.lower() or "goodbye" in user_input.lower():
                farewell_message = "Goodbye, have a nice day!"
                print(f"Amigo: {farewell_message}")
                speak_response(farewell_message)
                break

            bot_response = generate_response(user_input)
            print(f"Amigo: {bot_response}")
            speak_response(bot_response)
if __name__ == "__main__":
    amigo_conversational_companion()
