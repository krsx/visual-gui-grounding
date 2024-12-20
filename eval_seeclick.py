from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import ast
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from utils.process_utils import pred_2_point, extract_bbox
from utils import constant

torch.manual_seed(1234)

folder_name = "logs/seeclick/"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully!")


parser = argparse.ArgumentParser()
# parser.add_argument('--qwen_path', type=str, required=True)
parser.add_argument('--screenspot_imgs', type=str, required=True)
parser.add_argument('--screenspot_test', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--max_step', type=int, default=None)
args = parser.parse_args()


log_filename = "logs/seeclick/" + args.task + \
    datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode='w')
    ]
)

tokenizer = AutoTokenizer.from_pretrained(
    constant.QWEN_MODEL, trust_remote_code=True)

if args.model is not None and args.model == "qwen":
    model = AutoModelForCausalLM.from_pretrained(
        constant.QWEN_MODEL, device_map="cuda", trust_remote_code=True, bf16=True).eval()
    logging.info("TESTING QWEN-VL MODEL")
else:
    model = AutoModelForCausalLM.from_pretrained(
        constant.SEECLICK_MODEL, device_map="cuda", trust_remote_code=True, bf16=True).eval()
    logging.info("TESTING SEECLICK MODEL")

print("Load Success")
model.generation_config = GenerationConfig.from_pretrained(
    constant.QWEN_MODEL, trust_remote_code=True)

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
    prompt_origin = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with point)?"
    prompt_origin_qwen = "Generate the bounding box of {}"
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
        bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1],
                bbox[2] / img_size[0], bbox[3] / img_size[1]]

        prompt = prompt_origin.format(instruction)
        query = tokenizer.from_list_format([{'image': img_path},  # Either a local path or an url
                                            {'text': prompt}, ])
        # print(query)
        response, history = model.chat(tokenizer, query=query, history=None)
        # print(response)

        try:
            if 'box' in response:
                pred_bbox = extract_bbox(response)
                click_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2,
                               (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
                click_point = [item / 1000 for item in click_point]
            else:
                click_point = pred_2_point(response)
            if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                corr_action += 1
                if item["data_type"] == 'text':
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
                logging.info("[MATCH] " + str(corr_action / num_action))
            else:
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                logging.info("[UNMATCH] " + str(corr_action / num_action))
                logging.info("Agent result: " + str(click_point))
                logging.info("Ground truth: " + str(bbox))
            result.append({"img_path": img_path, "text": instruction, "bbox": bbox, "pred": click_point,
                           "type": item["data_type"], "source": item["data_source"]})
            logging.info("DETAILS: " + str(result[-1]))
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
