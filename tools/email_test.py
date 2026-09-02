import uuid
import argparse
import time
import sys
import os

# Local import
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
from src.utils.logger import logger
from src.utils.common import check_inbox, send_email

SENDER_EMAIL = "maplestoryautolevelup@gmail.com"
PASSWORD = "lvxfdhthvvrcuojj"
RECEIVER_EMAIL = "luckyyu910645@gmail.com" # TODO: change this

def wait_for_reply(token, timeout_sec=90, search_interval=10):
    '''
    Wait for user reply for a while
    Checks inbox for a reply containing the token.
    Times out after `timeout_sec` seconds.
    '''
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        reply = check_inbox(SENDER_EMAIL, PASSWORD, token)
        logger.info(f"User replied: {reply}")
        if reply and reply[0] in {"1", "2", "3", "4"}:
            return int(reply[0])

        logger.info(f"[{int(time.time() - start_time)}s] No valid reply yet. Waiting...")
        time.sleep(search_interval)

    logger.error(f"[wait_for_reply] Timeout: "
                 f"No valid reply received in {time.time() - start_time} seconds.")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        type=str,
        default='edit_me',
        help='Choose customized config yaml file in config/'
    )

    token = uuid.uuid4().hex[:8]  # Generate 8-character token
    send_email(SENDER_EMAIL, PASSWORD,
            RECEIVER_EMAIL,
            f"[MS Bot] Help me pass the test({token})",
            "Please directly reply this email\nType '1', '2', '3', or '4'",
            "screenshot/rune_detected_2025-06-23_03-47-05.png")

    user_reply = wait_for_reply(token)
    if user_reply:
        logger.info(f"User selected: {user_reply}", )
    else:
        logger.info(f"Proceeding without user response.")
