import os
import shutil
import time
import cv2
from openai import OpenAI
from segment_anything import build_sam, SamAutomaticMaskGenerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image, ImageDraw
import torch
import numpy as np
from dotenv import load_dotenv, find_dotenv
import json
from transformers import BitsAndBytesConfig
from utils import constant, prompt, response_model


class VowAgent:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.device = self.check_device()
        self.sam_mask = None
        self.processor = None
        self.model = None
        self.openai = None

    @staticmethod
    def check_device():
        """Checks for GPU or CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_sam(self):
        """Initializes the SAM model."""
        try:
            self.sam_mask = SamAutomaticMaskGenerator(
                model=build_sam(checkpoint=constant.SAM_MODEL).to(self.device),
                points_per_side=32,
                pred_iou_thresh=0.92,
                stability_score_thresh=0.95,
                box_nms_thresh=0.7,
            )
            print("SAM models loaded")
        except Exception as e:
            print("[ERROR] initializing SAM model:", e)

    @staticmethod
    def convert_gray_image(image_path, output_path):
        """Converts an image to grayscale."""
        try:
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(output_path, gray_image)
            print("Grayscale image saved successfully")
        except Exception as e:
            print("[ERROR] converting image to grayscale:", e)

    @staticmethod
    def generate_mask(image_path, mask_generator):
        """Generates masks for the given image."""
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image)
            print("Masks generated")
            return masks
        except Exception as e:
            print("[ERROR] generating masks:", e)
            return None

    @staticmethod
    def save_cropped_images(cropped_images, output_folder):
        """Saves cropped images."""
        for idx, image in enumerate(cropped_images):
            path = os.path.join(output_folder, f"{idx}.jpeg")
            image.save(path)
            print("Cropped image saved:", idx)

    @staticmethod
    def cropped_images(image_path, masks):
        """Crops images based on segmentation masks."""
        image = Image.open(image_path)
        cropped_images = []

        for mask in masks:
            try:
                segmentation = mask["segmentation"]
                bbox = mask["bbox"]
                cropped_images.append(VowAgent.segment_image(image, segmentation).crop(
                    VowAgent.convert_box_xywh_to_xyxy(bbox)))
            except Exception as e:
                print("[ERROR] cropping image:", e)

        return cropped_images

    @staticmethod
    def segment_image(image, segmentation_mask):
        """Applies a segmentation mask to the image."""
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

    @staticmethod
    def convert_box_xywh_to_xyxy(box):
        """Converts box format from [x, y, w, h] to [x1, y1, x2, y2]."""
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def generate_captioner(self, model_name=constant.BLIP_MODEL):
        """Initializes the BLIP model."""
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, device_map=self.device, quantization_config=quantization_config, torch_dtype=torch.float16)
            print("Captioner models loaded")
        except Exception as e:
            print("[ERROR] loading captioner models:", e)

    def caption_images(self, output_folder):
        """Generates captions for cropped images."""
        captions = []
        for filename in sorted(os.listdir(output_folder), key=lambda x: int(x.split('.')[0])):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                try:
                    image = Image.open(file_path)
                    inputs = self.processor(
                        image, return_tensors="pt").to(self.device, torch.float16)
                    generated_ids = self.model.generate(
                        **inputs, max_new_tokens=20)
                    generated_text = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True)[0].strip()
                    captions.append(generated_text)
                    if generated_text is not None or generated_text != "":
                        captions.append(generated_text)
                        print(f"Caption for {filename}: {generated_text}")
                    else:
                        captions.append("EMPTY")
                        print(f"Caption for {filename}: EMPTY")
                except Exception as e:
                    print(f"[ERROR] generating caption for {filename}:", e)
        return captions

    @staticmethod
    def save_to_txt(data_list, filename):
        """Saves a list of data to a text file."""
        try:
            with open(filename, 'w') as file:
                for idx, item in enumerate(data_list):
                    file.write(f"[{idx}] {item}\n")
            print(f"Data successfully saved to {filename}")
        except Exception as e:
            print(f"[ERROR] saving to {filename}:", e)

    def init_openai(self):
        """Initializes OpenAI."""
        try:
            self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            print("OpenAI initialized")
        except Exception as e:
            print("[ERROR] initializing OpenAI:", e)

    def execute_llm(self, user_goals, prev_actions, captions_path, image_path):
        """Executes the LLM for reasoning."""
        try:
            user_input = prompt.format_input(
                user_goals, prev_actions, captions_path)
            base64_image = prompt.encode_image(image_path)
            response = self.openai.beta.chat.completions.parse(
                model=constant.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": prompt.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_input,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":  base64_image
                                },
                            },
                        ]
                        # "content": user_input
                    }
                ],
                response_format=response_model.ThoughtResponse
            )
            return response
        except Exception as e:
            print("[ERROR] executing LLM:", e)
            return None

    def reset_segmentation_output(self):
        """Resets the segmentation output folder."""
        try:
            shutil.rmtree(constant.SEGMENTATION_OUTPUT_PATH)
            os.mkdir(constant.SEGMENTATION_OUTPUT_PATH)
            print("Segmentation output folder reset")
        except Exception as e:
            print("[ERROR] resetting segmentation output folder:", e)

    def extract_response(self, response):
        """Extracts the response from the LLM output."""
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print("[ERROR] extracting response:", e)
            return None

    def run_pipeline(self, image_path, user_goals, prev_actions):
        """Runs the complete pipeline."""
        self.reset_segmentation_output()

        gray_image_path = constant.GRAY_IMAGE_PATH
        self.convert_gray_image(image_path, gray_image_path)

        self.init_sam()
        masks = self.generate_mask(gray_image_path, self.sam_mask)

        cropped_images = self.cropped_images(gray_image_path, masks)
        self.save_cropped_images(
            cropped_images, constant.SEGMENTATION_OUTPUT_PATH)

        self.generate_captioner()
        captions = self.caption_images(constant.SEGMENTATION_OUTPUT_PATH)
        self.save_to_txt(captions, constant.CAPTIONS_OUTPUT_PATH)

        self.init_openai()
        response = self.execute_llm(
            user_goals,
            prev_actions,
            constant.CAPTIONS_OUTPUT_PATH,
            image_path)
        print("Final Response:", response)

        result = self.extract_response(response)
        return result


if __name__ == "__main__":
    agent = VowAgent()
    result = agent.run_pipeline(
        image_path=constant.TEST_IMAGE_PATH, user_goals="Create a new project", prev_actions=[])
    print(result)
