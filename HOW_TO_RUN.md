# How to Run the Project

These steps take you from a fresh download to a running app. They are written for
Windows (PowerShell). Mac/Linux commands are shown where they differ.

## Requirements

- Python 3.10 or newer (check with `python --version`)
- A Mistral API key, which you can create at https://console.mistral.ai/api-keys
- An internet connection (the agents call the Mistral API)

## Step 1: Unzip and open a terminal in the project folder

Unzip `simple_quantile_Assignment.zip`, then open a terminal inside the unzipped
`simple_quantile_Assignment` folder (the one that contains `app.py`):

```powershell
cd path\to\simple_quantile_Assignment
```

In VS Code you can use File > Open Folder, then Terminal > New Terminal, which opens
in the project folder.

## Step 2: Create and activate a virtual environment (recommended)

A virtual environment keeps this project's packages separate from other projects.
You only need to create it once.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Once it is active, you will see `(.venv)` at the start of the terminal line.

If PowerShell says "running scripts is disabled on this system", run this once and
then try activating again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Step 3: Install the packages

```powershell
pip install -r requirements.txt
```

This installs LangGraph, LangChain, the Mistral connector, Streamlit and python-dotenv.

## Step 4: Add your API key

Open the `.env` file in the project folder. If it does not exist, copy `.env.example`
to `.env`. Put your key after the `=` sign:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

Do not use quotes or spaces. Never share or commit this file.

## Step 5: Start the app

```powershell
streamlit run app.py
```

Your browser opens at http://localhost:8501. If it does not, open that address
yourself.

## Step 6: Use the app

1. In the left sidebar, keep "Use sample emails" ticked. You can also upload your own
   `.eml` or `.txt` emails.
2. Click **Run AI Agents**. Processing the 8 sample emails takes about 30 seconds.
3. Open the **Responses & schedule** tab. It shows:
   - a summary table sorted by priority, with urgent emails first
   - one expandable section per email, containing the original email, the
     classification, the drafted reply (you can edit it), the confidence score and
     the scheduled send time
4. Click **Save to outbox** to write the scheduled replies to `data/outbox.json`, or
   use **Download outbox JSON**. No email is actually sent.

To stop the app, press `Ctrl + C` in the terminal.

## Running it again later

You do not need to reinstall anything. Open a terminal in the project folder, then:

```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognized | Activate the virtual environment first, or run `python -m streamlit run app.py` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with the virtual environment active |
| Yellow warning "Add your MISTRAL_API_KEY" | Check the `.env` file, then restart the app |
| `401 Unauthorized` | The API key is wrong or revoked. Create a new one and update `.env` |
| `429 Rate limit exceeded` | Wait a minute and try again, or set `MODEL_NAME = "ministral-8b-latest"` in `config.py` |
| An email shows "Error" | Open that email in the results. The error message says what went wrong |

## Using your own emails

Put `.txt` or `.eml` files in `data/emails/`, or upload them from the sidebar. A
`.txt` email must look like this, with a blank line between the headers and the body:

```
From: Jane Doe <jane@example.com>
Subject: Where is my order?
Date: Sat, 26 Sep 2026 10:00:00 +0530

Hi, my order has not arrived yet...
```
