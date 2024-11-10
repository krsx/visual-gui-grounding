import base64
import os
import time
import cv2
from openai import OpenAI
from segment_anything import build_sam, SamAutomaticMaskGenerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image, ImageDraw
import torch
import numpy as np
from dotenv import load_dotenv, find_dotenv

from utils import constant, prompt
from utils import response_model


load_dotenv(find_dotenv())


def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def generate_mask(image_path, mask_generator):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    return masks


def save_cropped_images(cropped_images):
    for idx, image in enumerate(cropped_images):
        path = f"{constant.SEGMENTATION_OUTPUT_PATH}/{idx}.png"
        image.save(path)
        print("Successfully saved cropped image - ", idx)


def cropped_image(image_path, masks):
    image = Image.open(image_path)
    cropped_images = []

    for mask in masks:
        try:
            cropped_images.append(segment_image(image, mask["segmentation"]).crop(
                convert_box_xywh_to_xyxy(mask["bbox"])))
        except Exception as e:
            print("Error cropping image")
            print("Error: ", e)

    return cropped_images


def generate_captioner(model_name=constant.BLIP_MODEL):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, device_map=check_device(), load_in_8bit=True, torch_dtype=torch.float16)

    return processor, model


def caption_image(processor, model, output_folder=constant.SEGMENTATION_OUTPUT_PATH):
    captions = []

    for filename in sorted(os.listdir(output_folder), key=lambda x: int(x.split('.')[0])):
        file_path = os.path.join(output_folder, filename)

        if os.path.isfile(file_path):
            image_path = f"{output_folder}/{filename}"
            print(f"Processing file: {image_path}")
            image = Image.open(image_path)

            try:
                inputs = processor(image, return_tensors="pt").to(
                    check_device(), torch.float16)
                generated_ids = model.generate(**inputs, max_new_tokens=20)
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True)[0].strip()

                if generated_text is not None or generated_text != "":
                    captions.append(generated_text)
                    print(f"Caption: {generated_text}")
                else:
                    captions.append("EMPTY")
                    print("No caption generated")

            except Exception as e:
                print("Error generating caption")
                print("Error: ", e)

                return None

    return captions


def save_to_txt(data_list, filename=constant.CAPTIONS_OUTPUT_PATH):
    try:
        with open(filename, 'w') as file:
            for idx, item in enumerate(data_list):
                data = f"[{idx}] {item}\n"
                file.write(data)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


def init_openai():
    try:
        openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("OpenAI initialized")
        return openai
    except Exception as e:
        print("Error initializing OpenAI")
        print("Error: ", e)

        return None


def execute_llm(llm, user_goals, prev_actions, captions_path=constant.CAPTIONS_OUTPUT_PATH):
    user_input = prompt.format_input(user_goals, prev_actions, captions_path)
    response = llm.beta.chat.completions.parse(
        model=constant.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        response_format=response_model.ThoughtResponse
    )

    return response


# FOR TESTING PURPOSES
USER_GOALS = "I want to buy a new shoes as soon as possible"
PREV_ACTIONS = ""


def main():
    # print("Loading SAM models...")
    # try:
    #     sam_mask = SamAutomaticMaskGenerator(
    #         build_sam(checkpoint=constant.SAM_MODEL).to(check_device()))
    #     print("SAM models loaded")
    # except:
    #     print("SAM models not found")

    # print("Generating masks...")
    # try:
    #     masks = generate_mask(constant.IMAGE_PATH, sam_mask)
    #     print("Masks generated")
    # except:
    #     print("Error generating masks")

    # print("Cropping images...")
    # crop_time_start = time.time()
    # cropped_images = cropped_image(constant.IMAGE_PATH, masks)
    # save_cropped_images(cropped_images)
    # crop_time_end = time.time()

    # print("Loading captioner models...")
    # try:
    #     processor, model = generate_captioner()
    #     print("Captioner models loaded")
    # except:
    #     print("Error loading captioner models")

    # print("Generating segmentation caption...")
    # caption_time_start = time.time()
    # captions = caption_image(processor, model)
    # caption_time_end = time.time()

    # if captions is not None:
    #     print("Segmentation caption generated")

    #     print("Saving captions to txt...")
    #     save_to_txt(captions, "output/captions/captions.txt")
    # else:
    #     print("Error generating captions")

    print("Initializing LLM...")
    openai = init_openai()
    print("Thinking...")
    llm_time_start = time.time()
    result = execute_llm(openai, USER_GOALS, PREV_ACTIONS)
    llm_time_start = time.time()
    print("Final result:\n", result.choices[0].message.content)

    print("\nProcess completed!")
    # print("Time taken to crop images: ", crop_time_end - crop_time_start)
    # print("Time taken to generate captions: ",
    #       caption_time_end - caption_time_start)
    print("Time taken to analyze: ", llm_time_start - llm_time_start)


if __name__ == "__main__":
    main()
