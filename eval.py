import argparse
import json
import logging
from datetime import datetime
import os
from PIL import Image
from tqdm import tqdm
import yaml

from agent import VowAgent
from utils import constant


parser = argparse.ArgumentParser()
parser.add_argument('--screenspot_imgs', type=str, required=True)
parser.add_argument('--screenspot_test', type=str, required=True)
parser.add_argument('--task', type=str, default="all", required=True)
parser.add_argument('--max_step', type=int, default=None)
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()


log_filename = "logs/vowagent" + args.task + \
    datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode='w')
    ]
)


agent = VowAgent()
if args.config is None:
    agent.load_config(constant.CONFIG_PATH)
else:
    agent.load_config(args.config)
if args.task == "all":
    tasks = ["mobile", "desktop", "web"]
else:
    tasks = [args.task]
tasks_result = []
result = []
for task in tasks:
    logging.info("TASK: " + task)
    dataset = "screenspot_" + task + ".json"
    screenspot_data = json.load(
        open(os.path.join(args.screenspot_test, dataset), 'r'))
    print("Num of sample: " + str(len(screenspot_data)))

    num_action = 0
    corr_action = 0
    text_correct = []
    icon_correct = []
    num_wrong_format = 0

    for j, item in enumerate(screenspot_data):
        if args.max_step is not None and j >= args.max_step:
            break
        logging.info("PROCESSING STEP: " + str(j))
        num_action += 1
        filename = item["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        if not os.path.exists(img_path):
            print("[ERROR] Image not found: ", img_path)
            print("[ERROR] Please input the correct path of the image")
            input()
        image = Image.open(img_path)
        instruction = item["instruction"]
        bbox = item["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        img_size = image.size
        padding_error = 0.25
        bbox = [bbox[0] / img_size[0] - padding_error, bbox[1] / img_size[1] - padding_error,
                bbox[2] / img_size[0] + padding_error, bbox[3] / img_size[1] + padding_error]
        result = agent.run_pipeline(
            image_path=img_path, user_goals=instruction, prev_actions=None)
        try:
            if result["coordinates"] is not None:
                click_point = [result["coordinates"][0] /
                               img_size[0], result["coordinates"][1] / img_size[1]]
                if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                    corr_action += 1
                    if item["data_type"] == 'text':
                        text_correct.append(1)
                    else:
                        icon_correct.append(1)
                    logging.info("[MATCH] " + str(corr_action / num_action))
                    logging.info("Agent result: " + str(click_point))
                    logging.info("Ground truth: " + str(bbox))
                else:
                    if item["data_type"] == 'text':
                        text_correct.append(0)
                    else:
                        icon_correct.append(0)
                    logging.info("[UNMATCH] " + str(corr_action / num_action))
                    logging.info("Agent result: " + str(click_point))
                    logging.info("Ground truth: " + str(bbox))
                    details = str({"img_path": img_path, "text": instruction, "bbox": bbox,
                                   "pred": click_point, "type": item["data_type"], "source": item["data_source"]})
                    logging.info("Details: " + details)
        except:
            num_wrong_format += 1
            if item["data_type"] == 'text':
                text_correct.append(0)
            else:
                icon_correct.append(0)
            logging.info("[UNMATCH] Step: " + str(j) + " wrong format!")
            logging.info("Agent result: " + str(click_point))
            logging.info("Ground truth: " + str(bbox))

    logging.info("Action Acc: " + str(corr_action / num_action))
    logging.info("Total num: " + str(num_action))
    logging.info("Wrong format num: " + str(num_wrong_format))
    logging.info("Text Acc: " + str(sum(text_correct) /
                 len(text_correct) if len(text_correct) != 0 else 0))
    logging.info("Icon Acc: " + str(sum(icon_correct) /
                 len(icon_correct) if len(icon_correct) != 0 else 0))

    text_acc = sum(text_correct) / \
        len(text_correct) if len(text_correct) != 0 else 0
    icon_acc = sum(icon_correct) / \
        len(icon_correct) if len(icon_correct) != 0 else 0
    tasks_result.append([text_acc, icon_acc])

logging.info(tasks_result)
