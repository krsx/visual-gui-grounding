from PIL import Image
import streamlit as st
import os
import cv2
from openai import OpenAI
from segment_anything import build_sam, SamAutomaticMaskGenerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image, ImageDraw
import torch
import numpy as np
from dotenv import load_dotenv, find_dotenv
from transformers import BitsAndBytesConfig
import shutil

from utils import constant, prompt, response_model


def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


@st.cache_resource
def init_sam():
    try:
        sam_mask = SamAutomaticMaskGenerator(
            build_sam(checkpoint=constant.SAM_MODEL).to(check_device()))
        print("SAM models loaded")
    except Exception as e:
        print("SAM models not found")
        print("Error: ", e)

        st.error("SAM models not found")
        st.error(e)
    return sam_mask


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
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)

        print("Masks generated")
        return masks
    except Exception as e:
        print("Error generating masks")
        print("Error: ", e)

        st.error("Error generating masks")
        st.error(e)
    return None


def save_cropped_images(cropped_images):
    for idx, image in enumerate(cropped_images):
        path = f"{constant.SEGMENTATION_OUTPUT_PATH}/{idx}.jpeg"
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

            st.error("Error cropping image")
            st.error(e)

    return cropped_images


@st.cache_resource
def generate_captioner(model_name=constant.BLIP_MODEL):
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, device_map=check_device(), quantization_config=quantization_config, torch_dtype=torch.float16)

        print("Captioner models loaded")
        return processor, model
    except Exception as e:
        print("Error loading captioner models")
        print("Error: ", e)

        st.error("Error loading captioner models")
        st.error(e)

    return None


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

                st.error("Error generating caption")
                st.error(e)

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
        print("Error saving .txt file")
        print("Error: ", e)

        st.error("Error saving .txt file")
        st.error(e)


@st.cache_resource
def init_openai():
    try:
        openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("OpenAI initialized")
        return openai
    except Exception as e:
        print("Error initializing OpenAI")
        print("Error: ", e)

        st.error("Error initializing OpenAI")
        st.error(e)

        return None


def execute_llm(llm, user_goals, prev_actions, captions_path=constant.CAPTIONS_OUTPUT_PATH):
    user_input = prompt.format_input(user_goals, prev_actions, captions_path)
    try:
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
    except Exception as e:
        print("Error executing LLM")
        print("Error: ", e)

        st.error("Error executing LLM")
        st.error(e)

    return None


def draw_selected_segment(seg_index, masks, coordinates, dot_size=5):
    original_image = Image.open(constant.TEMP_IMAGE)
    overlay_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    overlay_color = (255, 0, 0, 200)

    segmentation_mask_image = Image.fromarray(
        masks[seg_index]["segmentation"].astype('uint8') * 255)
    draw = ImageDraw.Draw(overlay_image)

    draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

    x, y = coordinates
    top_left = (x - dot_size, y - dot_size)
    bottom_right = (x + dot_size, y + dot_size)

    draw.ellipse([top_left, bottom_right], fill="blue")
    result_image = Image.alpha_composite(
        original_image.convert('RGBA'), overlay_image)

    return result_image


def save_temp_image(image):
    try:
        image.save(constant.TEMP_IMAGE)
        print("Temp analyzed image saved successfully")
    except Exception as e:
        print("Error saving temp image")
        print("Error: ", e)

        st.error("Error saving temp image")
        st.error(e)


def reset_segmentation_output():
    try:
        shutil.rmtree(constant.SEGMENTATION_OUTPUT_PATH)
        os.mkdir(constant.SEGMENTATION_OUTPUT_PATH)
        print("Segmentation output folder reset")
    except Exception as e:
        print("Error resetting segmentation output folder")
        print("Error: ", e)

        st.error("Error resetting segmentation output folder")
        st.error(e)
