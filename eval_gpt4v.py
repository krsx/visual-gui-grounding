from tqdm import tqdm
import ast
from PIL import Image
import os
import base64
import random
import requests
import json
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


results = json.load(open('./gpt4v_result_mobile.json', 'r'))
icon_result = [item for item in results if item["data_type"] == "icon"]
text_result = [item for item in results if item["data_type"] == "text"]
print("icon acc: " +
      str((len([item for item in icon_result if item["correct"]])/len(icon_result))))
print("text acc: " +
      str((len([item for item in text_result if item["correct"]])/len(text_result))))
input()

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


screenspot_imgs_dir = './data/screenspot_imgs'
test_data = json.load(open('./data/screenspot_mobile.json', 'r'))
result = []
num_correct = 0
random.shuffle(test_data)
for item in tqdm(test_data[:]):

    img_filename = item["img_filename"]
    img_path = os.path.join(screenspot_imgs_dir, img_filename)
    image = Image.open(img_path)
    img_size = image.size
    bbox = item["bbox"]
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1],
            bbox[2] / img_size[0], bbox[3] / img_size[1]]

    instruction = item["instruction"]

    # Getting the base64 string
    base64_image = encode_image(img_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = "Localize an element on the GUI image according to my instructions and output its bounding box, [left, top, right, down], with each value between 0 and 1 indicating the ratio of width and height. Please don't call the tool but position the element directly according to the image content. Please don't reply to anything other than a [left, top, right, down] list.\nLocalize \"{}\" in the image using bounding box.\nPlease tell me the results directly without the intermediate analyzing process."
    prompt = prompt.format(instruction)

    result_item = {"img_filename": img_filename,
                   "data_type": item["data_type"], "data_souce": item["data_source"], "prompt": prompt, "correct": False}

    payload = {
        "model": "gpt-4-vision-preview",
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
                            "url": f"data:image/jpeg;base64,{base64_image}"
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

    try:
        pred = ast.literal_eval(
            response.json()['choices'][0]['message']['content'])

        click_point = [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2]

        if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
            num_correct += 1
            result_item["correct"] = True
            print("correct")
        else:
            print("incorrect")
    except:
        print("wrong format")

    result.append(result_item)

    json.dump(result, open('./gpt4v_result_mobile.json', 'w'))

print("Success rate: "+str(num_correct/len(test_data)))
