import logging
from repost_bot.bot import image_ss_bot
import sys

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():

    assert len(sys.argv) == 3

    access_token = sys.argv[1]
    archive_base_path = sys.argv[2]

    image_ss_bot.main(access_token, archive_base_path)


if __name__ == "__main__":
    main()
