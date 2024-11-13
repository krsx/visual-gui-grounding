import json
import time
from PIL import Image
import streamlit as st

from utils import constant, prompt, response_model
from logic.grounding import (
    extract_show_segment_info,
    init_sam,
    generate_mask,
    save_cropped_images,
    cropped_image,
    generate_captioner,
    caption_image,
    save_to_txt,
    init_openai,
    execute_llm,
    draw_selected_segment,
    save_temp_image,
    reset_segmentation_output,
    show_table_stats
)


st.set_page_config(
    page_title="LLM Grounding Demo",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)


if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = prompt.SYSTEM_PROMPT


def start_analyze(image_file, user_goal):
    if image_file is not None and user_goal is not None:
        st.subheader("**Processed Image**")
        st.image(image_file,
                 caption="Webpage Screenshot", use_container_width=True)

        with st.spinner("Analyzing image..."):
            st.session_state.is_running = True

            print("Reset previous segmentation...")
            reset_segmentation_output()

            print("Loading SAM models...")
            sam_mask = init_sam()

            print("Generating masks...")
            masks = generate_mask(constant.TEMP_IMAGE, sam_mask)

            print("Cropping images...")
            crop_time_start = time.time()
            cropped_images = cropped_image(constant.TEMP_IMAGE, masks)
            save_cropped_images(cropped_images)
            crop_time_end = time.time()
            crop_time_inference = crop_time_end - crop_time_start

            print("Loading captioner models...")
            processor, model = generate_captioner(is_quantized=True)

            print("Generating segmentation caption...")
            caption_time_start = time.time()
            captions = caption_image(processor, model)
            caption_time_end = time.time()
            caption_time_inference = caption_time_end - caption_time_start

            if captions is not None:
                print("Segmentation caption generated")

                print("Saving captions to txt...")
                save_to_txt(captions)
            else:
                print("Error generating captions")

            print("Initializing LLM...")
            openai = init_openai()
            print("Thinking...")
            llm_time_start = time.time()
            result = execute_llm(openai, user_goal, [])
            llm_time_end = time.time()
            llm_time_inference = llm_time_end - llm_time_start
            extract_json = json.loads(result.choices[0].message.content)

            print("Showing selected segment...")
            st.subheader("**Grounding Result**")
            seg_index, coordinates = extract_show_segment_info(
                masks, extract_json)
            result_img = draw_selected_segment(seg_index, masks, coordinates)
            st.image(result_img, caption="Selected Segment",
                     use_container_width=True)

            print("\nProcess completed!")
            print(
                f"Time taken to crop images: {crop_time_inference:.4f} seconds")
            print(
                f"Time taken to generate captions: {caption_time_inference:.4f} seconds")
            print(f"Time taken to analyze: {llm_time_inference:.8f} seconds")

            show_table_stats(crop_time_inference,
                             caption_time_inference, llm_time_inference)

            st.session_state.is_running = False


with st.sidebar:
    st.title('Settings')
    user_goal = st.text_input(
        label="**User Goal**", placeholder="Enter specific user goals here")

    image_file = st.file_uploader(
        "**Upload an Image**", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        try:
            image = Image.open(image_file)
        except Exception as e:
            st.error("The uploaded file is not a valid image.")
            st.error(f"Error: {e}")

    with st.expander("**Current Prompt**"):
        st.markdown(prompt.SYSTEM_PROMPT)
    new_prompt = st.text_area(
        "**Update Prompt (Optional)**", placeholder="Enter a new prompt")

    analyze_clicked = st.button("Run Analyze")
    if analyze_clicked and user_goal and image_file and new_prompt:
        st.session_state.system_prompt = new_prompt if new_prompt else prompt.SYSTEM_PROMPT
        st.success("Prompt updated successfully!")


st.title('LLM Grounding Demo')
st.divider()


if analyze_clicked and user_goal and image_file and not st.session_state.is_running:
    save_temp_image(image)
    start_analyze(image_file, user_goal)
else:
    st.info("Please input user goals and upload an image to start analyzing!")
