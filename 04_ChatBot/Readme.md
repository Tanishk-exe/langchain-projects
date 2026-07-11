# AI Chatbot with Conversation Memory

A simple chatbot built using **LangChain** and **Google Gemini** to understand how conversation memory works in LLM applications.

Instead of replying to each message independently, the chatbot remembers previous messages and uses them to generate context-aware responses.

## What I Learned

- Chat Models
- System, Human, and AI Messages
- Conversation History
- Context-aware Chatbots
- Multi-turn Conversations

## Project Versions

### Version 1
A basic chatbot with **no memory**. Every message is treated as a new conversation.

### Version 2
Stores previous messages in a Python list. It remembers the conversation but doesn't distinguish between user and AI messages.

### Version 3 (Final)
Uses LangChain's `SystemMessage`, `HumanMessage`, and `AIMessage` to properly maintain conversation history, making the chatbot much more reliable for multi-turn conversations.

## Tech Stack

- Python
- LangChain
- Google Gemini
- python-dotenv

## Run Locally

```bash
pip install -r requirements.txt
