import os
import requests
from fastapi import FastAPI, Request
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters
import logging

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Your Flask API endpoint
API_URL = os.getenv("API_URL", "https://brain-classification.onrender.com/api/classify")

app = FastAPI()

# Initialize bot and dispatcher
bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN", "7463448864:AAHWNwDNh14aYAhdV3vfzE2GddlvL7WFceU"))
dispatcher = Dispatcher(bot, None, workers=0)

def start(update: Update, context) -> None:
    update.message.reply_text(
        "Welcome to the Brain Diagnosis Bot! Send me a brain scan image."
    )

def handle_image(update: Update, context) -> None:
    chat_id = update.message.chat_id
    if update.message.photo:
        photo = update.message.photo[-1]
        file = context.bot.get_file(photo.file_id)
        try:
            response = requests.get(file.file_path)
            response.raise_for_status()
            image_data = response.content
            files = {"brain_scan": ("image.jpg", image_data, "image/jpeg")}
            api_response = requests.post(API_URL, files=files)
            api_response.raise_for_status()
            result = api_response.json()
            if result.get("success"):
                data = result["data"]
                response_text = (
                    f"Diagnosis: {data.get('classification', 'Unknown')}\n"
                    f"Confidence: {data.get('confidence', 0):.2%}\n"
                    f"Probabilities:\n"
                    f"- Alzheimer's: {data.get('probabilities', {}).get('alzheimers', 0):.2%}\n"
                    f"- Normal: {data.get('probabilities', {}).get('normal', 0):.2%}\n"
                    f"- Parkinson's: {data.get('probabilities', {}).get('parkinsons', 0):.2%}"
                )
            else:
                response_text = f"Error: {result.get('error', 'Unknown error')}"
            update.message.reply_text(response_text)
        except requests.RequestException as e:
            update.message.reply_text(f"Error contacting the API: {str(e)}")
        except Exception as e:
            update.message.reply_text(f"Error processing image: {str(e)}")
    else:
        update.message.reply_text("Please send an image file.")

# Register handlers
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.photo, handle_image))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, lambda update, context: update.message.reply_text("Please send an image.")))

@app.post("/webhook")
async def webhook(request: Request):
    update = Update.de_json(await request.json(), bot)
    await dispatcher.process_update(update)
    return {"status": "ok"}

@app.get("/")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
