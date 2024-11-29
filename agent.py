import os
import shutil
import time
import cv2
from openai import OpenAI
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
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
        self.captioner = None
        self.openai = None

    @staticmethod
    def check_device():
        """Checks for GPU or CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_models(self):
        """Initializes the SAM, BLIP, OpenAI models."""
        if self.sam_mask is None:
            self.init_sam()
        if self.processor is None or self.captioner is None:
            self.generate_captioner()
        if self.openai is None:
            self.init_openai()

    def init_sam(self):
        """Initializes the SAM model."""
        try:
            sam = sam_model_registry[constant.SAM_TYPE](
                checkpoint=constant.SAM_MODEL).to(self.device)
            self.sam_mask = SamAutomaticMaskGenerator(
                model=sam,
                points_per_batch=32,
                # points_per_side=32,
                # pred_iou_thresh=0.92,
                # stability_score_thresh=0.95,
                # box_nms_thresh=0.7,
            )
            print("SAM models loaded")
        except Exception as e:
            print("[ERROR] initializing SAM model:", e)

    def generate_captioner(self, model_name=constant.BLIP_MODEL):
        """Initializes the BLIP model."""
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.captioner = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                device_map=self.device,
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )
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
                    generated_ids = self.captioner.generate(
                        **inputs, max_new_tokens=20)
                    generated_text = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True)[0].strip()

                    if generated_text is not None or generated_text != "":
                        captions.append(generated_text)
                        print(f"Caption for {filename}: {generated_text}")
                    else:
                        captions.append("EMPTY")
                        print(f"Caption for {filename}: EMPTY")
                except Exception as e:
                    print(f"[ERROR] generating caption for {filename}:", e)
        return captions

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
    def save_cropped_images(cropped_images, output_folder, format="jpeg"):
        """Saves cropped images."""
        for idx, image in enumerate(cropped_images):
            path = os.path.join(output_folder, f"{idx}.{format}")
            image.save(path)
            print(f"Cropped image saved: {idx}.{format}")

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

    @staticmethod
    def save_to_txt(data_list, filename):
        """Saves a list of data to a text file."""
        try:
            with open(filename, 'w') as file:
                for idx, item in enumerate(data_list):
                    safe_item = item.encode(
                        'charmap', errors='replace').decode('charmap')
                    file.write(f"[{idx}] {safe_item}\n")
            print(f"Data successfully saved to {filename}")
        except Exception as e:
            print(f"[ERROR] saving to {filename}:", e)

    def extract_response(self, response):
        """Extracts the response from the LLM output."""
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print("[ERROR] extracting response:", e)
            return None

    @staticmethod
    def draw_selected_segment(seg_index, masks, coordinates, dot_size=5):
        original_image = Image.open(constant.TEST_IMAGE_PATH)
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
        result_image.show()

    def reset_segmentation_output(self):
        """Resets the segmentation output folder."""
        try:
            shutil.rmtree(constant.SEGMENTATION_OUTPUT_PATH)
            os.mkdir(constant.SEGMENTATION_OUTPUT_PATH)
            print("Segmentation output folder reset")
        except Exception as e:
            print("[ERROR] resetting segmentation output folder:", e)

    def reset_captions_output(self):
        """Resets the captions output file."""
        try:
            with open(constant.CAPTIONS_OUTPUT_PATH, 'w') as file:
                file.write("")
            print("Captions output file reset")
        except Exception as e:
            print("[ERROR] resetting captions output file:", e)

    def run_pipeline(self, image_path, user_goals, prev_actions, is_display=False):
        """Runs the complete pipeline."""
        self.reset_segmentation_output()
        self.reset_captions_output()

        self.init_models()

        gray_image_path = constant.GRAY_IMAGE_PATH
        self.convert_gray_image(image_path, gray_image_path)

        masks = self.generate_mask(gray_image_path, self.sam_mask)
        cropped_images = self.cropped_images(gray_image_path, masks)
        self.save_cropped_images(
            cropped_images, constant.SEGMENTATION_OUTPUT_PATH)

        captions = self.caption_images(constant.SEGMENTATION_OUTPUT_PATH)
        self.save_to_txt(captions, constant.CAPTIONS_OUTPUT_PATH)

        response = self.execute_llm(
            user_goals,
            prev_actions,
            constant.CAPTIONS_OUTPUT_PATH,
            image_path)
        result = self.extract_response(response)
        seg_index = result["action"]["option_number"]
        try:
            result["coordinates"] = masks[seg_index]["point_coords"][0]
        except Exception as e:
            print("[ERROR] extracting coordinates:", e)
            result["coordinates"] = [99, 99]  # Default coordinates for error
        result = json.dumps(result, indent=4)
        json_result = json.loads(result)

        if is_display:
            self.draw_selected_segment(
                seg_index, masks, json_result["coordinates"])
        print("Final Result:")
        print(result)

        return json_result


if __name__ == "__main__":
    agent = VowAgent()
    agent.run_pipeline(constant.TEST_IMAGE_PATH,
                       "Click on the Maps", None, is_display=True)
