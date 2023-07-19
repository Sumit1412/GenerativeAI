# Import necessary libraries
import os
import openai
import gradio as gr

# Set the OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initial conversation history for the chatbot
message_history = [ {'role':'system', 'content':"""
You are OrderBot, an automated service to collect orders for a online drug delivery. \
You first greet the customer, then collects the order, \
and then asks if it's a pickup from nearest pharmacy or delivery at doorstep. \
Don't suggest any durg outside the provide list, if any drug pertaining to any disease\
is not in the catalog, Just reply "Out of Stock".\
Never Display complete list of drug, instead ask for the "Details of Symptoms"\
accordingly suggest medicine from Catalog.\
You wait to collect the entire order, then summarize it and check for a final \
time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Finally you collect the payment.\
Make sure to clarify all options, extras and sizes to uniquely \
identify the item from the menu.\
You respond in a short, very conversational friendly style. \
The product list includes \
1000 Para 1000mg Tablet 12'S  47.56 \
108 FORTE Tablet 15's   24.52 \
Fepanil Banana Flavour 120 Oral Suspension 60ml 29.76 \
Vicks Vaporub 50 ml 152.21 \
Kapiva Wild Tulsi Giloy Juice 1 ltr ₹298.80 \
Nutrition: \
Diataal D Capsule 10'S 120.00 \
Revital H Capsule 60'S 456.50 \
Softovac SF Sugar Free Powder 100gm 184.50 \
Revital H Capsule 30'S 257.30 \
D Protin Chocolate Powder 500 gm  487.90 \
Stamina Booster:\
    Shilajeet Gold Capsule 10's 45.60\
"""},
{"role": "assistant", "content": "Please Provide the Symptoms or specific medicine name which \
                        you would like to Order"},
                    ]


def predict(user_input):
    """
    Predicts the response of a chatbot using OpenAI's GPT-3.5 language model.

    Parameters:
        user_input (str): The user's input message to be processed by the chatbot.

    Returns:
        list of tuple: A list of tuples containing the chatbot's responses in the form of (user_input, chatbot_response).
                       Each tuple represents a single back-and-forth message exchange in the conversation.

    Description:
        The `predict` function interacts with the OpenAI GPT-3.5 language model to generate the chatbot's response based
        on the conversation history stored in the `message_history` variable. It appends the user's input to the message
        history and then sends the entire conversation to the language model using the `openai.ChatCompletion.create`
        method.

        The function extracts the chatbot's reply from the response and adds it to the message history. The final
        response is then constructed as a list of tuples, where each tuple contains the user's input and the chatbot's
        corresponding response.

    Example:
        # Initializing the conversation with a system message and an assistant prompt
        message_history = [{'role': 'system', 'content': 'You are OrderBot, an automated service to collect orders for an online drug delivery. ...'},
                           {'role': 'assistant', 'content': 'Please provide the Symptoms or specific medicine name which you would like to Order'}]

        # User input
        user_input = "I have a headache and need some medicine."

        # Calling the predict function to get the chatbot's response
        response = predict(user_input)

        # The 'response' variable will now contain the chatbot's reply as a list of tuples.
        # Each tuple in the list represents one message exchange in the conversation.
        # Example: [('I have a headache and need some medicine.', 'Please provide the specific medicine name you'd like to order.')]
    """
    global message_history
    message_history.append({"role": "user", "content": user_input})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history
    )

    reply_content = completion.choices[0].message.content
    message_history.append({"role": "user", "content": reply_content})

    response = [(message_history[i]["content"], message_history[i + 1]["content"]) for i in
                range(2, len(message_history) - 1, 2)]
    return response

# Create a Gradio interface for the chatbot
block = gr.Blocks(theme=gr.themes.Default())
with block:
    gr.Markdown("""<h1 style='text-align: center;'>Welcome to the Drug Delivery Bot Chat</h1>""")
    chatbot = gr.Chatbot()
    message = gr.Textbox(show_label=False, placeholder="Type your message here")
    state = gr.State()
    submit = gr.Button("Chat")
    submit.click(predict,
                 inputs=[message],
                 outputs=[chatbot])
    submit.click(lambda: "", None, message)

    block.launch(debug=True)








