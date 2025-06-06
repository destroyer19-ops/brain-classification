import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Your Flask API endpoint
API_URL = os.getenv("API_URL", "https://brain-classification.onrender.com//api/classify")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    await update.message.reply_text(
        "Welcome to the Brain Diagnosis Bot! Send me a brain scan image, and I'll analyze it for Alzheimer's or Parkinson's disease."
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image uploads and send to the API."""
    if update.message.photo:
        # Get the highest resolution photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Download the image
        try:
            response = requests.get(file.file_path)
            response.raise_for_status()
            image_data = response.content
            
            # Send image to your Flask API
            files = {"brain_scan": ("image.jpg", image_data, "image/jpeg")}
            api_response = requests.post(API_URL, files=files)
            api_response.raise_for_status()
            result = api_response.json()
            
            # Process API response
            if result.get("success"):
                data = result["data"]
                classification = data.get("classification", "Unknown")
                confidence = data.get("confidence", 0.0)
                probabilities = data.get("probabilities", {})
                
                # Format response
                response_text = (
                    f"Diagnosis: {classification}\n"
                    f"Confidence: {confidence:.2%}\n"
                    f"Probabilities:\n"
                    f"- Alzheimer's: {probabilities.get('alzheimers', 0):.2%}\n"
                    f"- Normal: {probabilities.get('normal', 0):.2%}\n"
                    f"- Parkinson's: {probabilities.get('parkinsons', 0):.2%}"
                )
            else:
                response_text = f"Error: {result.get('error', 'Unknown error')}"
                
            await update.message.reply_text(response_text)
        except requests.RequestException as e:
            await update.message.reply_text(f"Error contacting the API: {str(e)}")
        except Exception as e:
            await update.message.reply_text(f"Error processing image: {str(e)}")
    else:
        await update.message.reply_text("Please send an image file.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by updates."""
    logger.warning(f'Update "{update}" caused error "{context.error}"')

async def main() -> None:
    """Start the bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "7463448864:AAHWNwDNh14aYAhdV3vfzE2GddlvL7WFceU")
    if token == "your-bot-token-here":
        logger.error("TELEGRAM_BOT_TOKEN not set. Please set the environment variable.")
        return
    
    app = Application.builder().token(token).build()
    
    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: update.message.reply_text("Please send an image.")))
    app.add_error_handler(error_handler)
    
    # Start polling
    logger.info("Bot started")
    await app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
