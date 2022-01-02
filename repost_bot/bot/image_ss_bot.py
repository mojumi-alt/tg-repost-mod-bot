import logging
from telegram import Update
from telegram.ext import (
    Updater,
    MessageHandler,
    Filters,
    CallbackContext,
)
import os
from PIL import Image
from ..similarity_analysis import patch_decomposition, image_distance
from glob import glob
import io
import uuid

# Patch sampling parameters.
# TODO: Should be configurable, maybe over a bot command.
stride = 5
patch_size = 10
distance_threshold = 7000


def put_into_image_storage(buffer, base_path, chat_id):

    # Make cache dir.
    os.makedirs(os.path.join(base_path, str(chat_id)), exist_ok=True)

    # Write file.
    with open(os.path.join(base_path, str(chat_id), str(uuid.uuid4())), "wb") as f:
        f.write(buffer.read())


def get_images_from_storage(base_path, chat_id):
    return [
        Image.open(path) for path in glob(os.path.join(base_path, str(chat_id), "*"))
    ]


def get_image_paths_from_storage(base_path, chat_id):
    return glob(os.path.join(base_path, str(chat_id), "*"))


def get_reference_set(base_path, chat_id):

    # Get images from cache.
    # TODO: Highly ineffective. The images / patches should be cached
    # in memory between calls.
    image_paths = get_image_paths_from_storage(base_path, chat_id)
    images = [Image.open(path) for path in image_paths]
    logging.info(f"Found {len(images)} reference images.")

    # Compute patches for reference set.
    patches = [
        patch_decomposition.get_patches(
            patch_decomposition.image_to_array(i), patch_size, stride
        )
        for i in images
    ]

    return patches, image_paths


def main(bot_token: str, archive_base_path: str) -> None:

    # Create the Updater and (also the bot) with the specified token.
    updater = Updater(bot_token)

    # Image classify callback.
    # FIXME: This should definitly queue the task and reply async
    # using the underlying functions of reply_*.
    def classify_image(update: Update, context: CallbackContext) -> None:

        # Get chat id. Every image cache is bound to one
        # chat.
        chat_id = update.effective_chat.id
        logging.info(f"Responding in chat {chat_id}")

        # Get the reference set for the current operation.
        ref_patches, ref_paths = get_reference_set(archive_base_path, chat_id)

        # Download image and prepare patches.
        # Make in memory buffer.
        buffer = io.BytesIO()
        updater.bot.get_file(file_id=update.message.photo[1]).download(out=buffer)

        # Load image from memory buffer.
        source_image = Image.open(buffer)

        # Make patches for classify image.
        source_patches = patch_decomposition.get_patches(
            patch_decomposition.image_to_array(source_image), patch_size, stride
        )

        # Action only required if reference images present.
        distance, index = None, None
        if ref_patches:

            # Compute distance.
            # FIXME: Again: This should be done in batches.
            index, distance = image_distance.compute_image_set_correspondence(
                [source_patches], ref_patches
            )[0]
            logging.info(f"Found reference image {index} with distance {distance}")

        else:
            logging.info("No reference images found...")

        # Write source image to disk if not already present determined by distance.
        if distance is None or distance > distance_threshold:

            logging.info("Writing image to storage")
            buffer.seek(0)
            put_into_image_storage(
                buffer,
                archive_base_path,
                update.effective_chat.id,
            )
        else:

            logging.info("Did not add image. Reply")
            # Read image and send reply.
            # TODO: Should send the image ref instead of reuploading the image...
            with open(ref_paths[index], "rb") as f:
                update.message.reply_photo(
                    f,
                    quote=True,
                    caption=f"This image looks a lot like this one... Distance: {distance}",
                )

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(
        MessageHandler(Filters.photo & ~Filters.command, classify_image)
    )

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()
