import base64


SYSTEM_PROMPT = """

# User input
User goals:
{user goals}

Previous actions:
{previous actions}

Option list:
{list of options and its description}

# Instructions
Imagine you are robot browsing the web, just like a humans. Now you need to complete a task. You will receive a screenshot of a webpage. Carefully observe and analyze the screenshot. Based on previous actions, and previous selected components/elements, you need to pick which option number based on the given option list that are relevant to accomplish user goals. Keep in mind that the option list are in form of a descriptive caption generated based on components/section/content of the webpage screenshot. Then, choose one of the following actions:
1. Click a web element. Be descriptive and specific for which web element should be clicked based on the given option list
2. Type content in a text box/text field/text area based on the given option list
3. Scroll up/down
4. Answer. This action should only be chosen when user goals already accomplished
Action should STRICTLY follow the format:
1. CLICK [option number and its description]
2. TYPE [content] in [option number and its description]
3. SCROLL [up/down]
4. ANSWER [content] in [option number and its description]

Your reply should strictly follow the format:
Thought: {Your brief thoughts (briefly summarize the info that will help ANSWER)}
Action: {One Action format you choose}

"""


# BLIP_PROMPT = """
# Question: Based on this image, is this a TEXT, IMAGE, BUTTON, ICON, or OTHER? Only answer based on the given options.
# """

EVAL_PROMPT = """

"""


def format_input(user_goals: str, prev_actions: list = None, captions_path: str = None):
    formatted_prev_actions = "-" if prev_actions is None else "\n".join(
        prev_actions)

    try:
        with open(captions_path, "r") as captions_file:
            text_data = captions_file.readlines()
        formatted_options = "\n".join(line.strip() for line in text_data)
    except FileNotFoundError:
        formatted_options = "No options available - file not found."

    prompt_input = f"""
* User input *
User goals:
{user_goals}
    
Previous actions:
{formatted_prev_actions}
    
Option list:
{formatted_options}
    """

    return prompt_input


def encode_image(image_path):
    mime_type = "jpeg" if image_path.lower().endswith(
        ".jpg") or image_path.lower().endswith(".jpeg") else "png"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:image/{mime_type};base64,{base64_image}"
