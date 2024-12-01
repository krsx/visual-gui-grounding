import argparse
from datetime import datetime
import logging
from tqdm import tqdm
import ast
from PIL import Image
import os
import base64
import random
import requests
import json
from dotenv import load_dotenv, find_dotenv

from utils import prompt as utils_prompt

load_dotenv(find_dotenv())


folder_name = "logs/gpt4v/"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully!")


parser = argparse.ArgumentParser()
parser.add_argument('--screenspot_imgs', type=str, required=True)
parser.add_argument('--screenspot_test', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--max_step', type=int, default=None)
args = parser.parse_args()


log_filename = folder_name + args.task + \
    datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode='w')
    ]
)


def execute_llm(api_key, item, img_filename, img_path, instruction):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = "Localize an element on the GUI image according to my instructions and output its bounding box, [left, top, right, down], with each value between 0 and 1 indicating the ratio of width and height. Please don't call the tool but position the element directly according to the image content. Please don't reply to anything other than a [left, top, right, down] list.\nLocalize \"{}\" in the image using bounding box.\nPlease tell me the results directly without the intermediate analyzing process."
    prompt = prompt.format(instruction)

    result_item = {"img_filename": img_filename,
                   "data_type": item["data_type"], "data_souce": item["data_source"], "prompt": prompt, "correct": False}

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": utils_prompt.encode_image(img_path)
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())

    result_item["response"] = response.json()
    return result_item, response


# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
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
        bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1],
                bbox[2] / img_size[0], bbox[3] / img_size[1]]

        instruction = item["instruction"]

        result_item, response = execute_llm(
            api_key, item, filename, img_path, instruction)
        try:
            pred = ast.literal_eval(
                response.json()['choices'][0]['message']['content'])

            click_point = [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2]

            if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                corr_action += 1
                result_item["correct"] = True
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
            result.append(result_item)
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


# screenspot_imgs_dir = './data/screenspot_imgs'
# test_data = json.load(open('./data/screenspot_mobile.json', 'r'))
# result = []
# num_correct = 0


# for item in tqdm(test_data[:]):

#     img_filename = item["img_filename"]
#     img_path = os.path.join(screenspot_imgs_dir, img_filename)
#     image = Image.open(img_path)
#     img_size = image.size
#     bbox = item["bbox"]
#     bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
#     bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1],
#             bbox[2] / img_size[0], bbox[3] / img_size[1]]

#     instruction = item["instruction"]

#     result_item, response = execute_llm(api_key, item, img_filename, img_path, instruction)

#     try:
#         pred = ast.literal_eval(
#             response.json()['choices'][0]['message']['content'])

#         click_point = [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2]

#         if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
#             num_correct += 1
#             result_item["correct"] = True
#             print("correct")
#         else:
#             print("incorrect")
#     except:
#         print("wrong format")

#     result.append(result_item)

#     json.dump(result, open('./gpt4v_result_mobile.json', 'w'))

# print("Success rate: "+str(num_correct/len(test_data)))
