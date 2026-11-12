# Meeting Minutes Generator

Turn an English meeting recording into a transcript and draft meeting minutes using open-weight speech and language models in Google Colab.

The notebook transcribes audio with **Whisper medium.en**, then uses **Phi-4-mini-instruct** to draft a summary, discussion points, takeaways, and action items in Markdown. Phi runs with 4-bit quantization to reduce its memory footprint.

## How it works

```text
Audio in Google Drive
        |
        v
Whisper medium.en --> Transcript
                          |
                          v
                Phi-4-mini-instruct (4-bit)
                          |
                          v
                 Draft meeting minutes
```

This is a notebook-based learning project. It displays the transcript and generated text inside the notebook; it does not include a web app, speaker identification, or automatic file export.

## Requirements

- A Google account and access to Google Colab and Google Drive.
- A Colab runtime with an NVIDIA GPU. The notebook explicitly uses CUDA; a CPU-only runtime will not run it as written.
- An internet connection to install packages and download model weights.
- A Hugging Face account and a read token stored in Colab Secrets as `HF_TOKEN`.
- An English meeting recording you are permitted to process.
- FFmpeg available in the runtime to decode audio files such as MP3.

GPU availability and session limits depend on [Colab's available resources](https://research.google.com/colaboratory/faq.html). The notebook does not establish a minimum VRAM requirement or guarantee that every Colab GPU can handle every recording.

## Quick start: Google Colab

### 1. Open the notebook

Download [Meeting.ipynb](Meeting.ipynb), open [Google Colab](https://colab.research.google.com/), and choose **File > Upload notebook**. Once this project is on GitHub, you can also open it through Colab's **GitHub** tab.

### 2. Enable a GPU

Choose **Runtime > Change runtime type**, select a GPU hardware accelerator, and save.

For a quick check, run this in a temporary cell:

```python
import torch
print(torch.cuda.is_available())  # Must print True
```

### 3. Configure the Hugging Face token

Create a read token in your [Hugging Face token settings](https://huggingface.co/settings/tokens). In Colab's **Secrets** panel (the key icon), add a secret named `HF_TOKEN` and enable notebook access.

The notebook retrieves this with `google.colab.userdata`. It does not read a `.env` file, and it does not require an OpenAI API key. Both models run in the Colab runtime after their weights are downloaded.

### 4. Add your audio

Upload your recording to Google Drive. The default path in the notebook is:

```python
audio_filename = "/content/drive/MyDrive/denver_extract.mp3"
```

Change `audio_filename` if your recording has a different name or location. The local `denver_extract.mp3` recording is excluded from Git; a clone of this repository does not include audio.

### 5. Run the cells in order

1. Run the installation cell. It installs `bitsandbytes`, `accelerate`, and the notebook's pinned `transformers==4.57.6` version. If Colab requests a runtime restart, restart before continuing with the imports.
2. Run the imports and model constant cells.
3. Mount Google Drive and authorize access, then check the audio path.
4. Authenticate with Hugging Face using the secret you added.
5. Run Whisper transcription. The transcript is printed below the cell.
6. Review the prompt. Replace the reference to a **Denver council meeting** if you are using another recording.
7. Run the quantization and generation cells, then the decoding and display cells.

The first run downloads the model weights and may take some time. Generated text streams into the notebook before the final Markdown display.

## Output and customization

The prompt asks for:

- A summary including attendees, location, and date.
- Key discussion points.
- Takeaways.
- Action items with owners.

These are requested fields, not verified facts. Review the transcript and minutes against the recording, especially names, dates, decisions, and action owners. If the recording omits a detail, mark it as unknown rather than accepting a model's guess.

| Setting | Where to change it |
| --- | --- |
| Recording | `audio_filename` in the Drive cell |
| Meeting context and output sections | `system_message` and `user_prompt` |
| Minutes model | `PHI` (requires a compatible causal language model and chat template) |
| Maximum generated length | `max_new_tokens=2000` in `model.generate` |
| Quantization compute type | `bnb_4bit_compute_dtype` in `BitsAndBytesConfig` |

The current decoder includes the input prompt and special tokens in `response`. To display only the generated minutes, replace the decoding cell with:

```python
generated_tokens = outputs[0, inputs.shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
```

To save that result, add a cell after decoding:

```python
from pathlib import Path

Path("/content/meeting_minutes.md").write_text(response, encoding="utf-8")
```

Download the file from Colab's Files panel before the runtime ends. Export is an optional addition, not part of the existing notebook.

## Project files

```text
Meeting_Minutes/
|-- Meeting.ipynb       # Colab workflow: audio -> transcript -> minutes
|-- README.md          # Setup, usage, and limitations
|-- requirements.txt   # Python dependencies for the workflow
|-- CONTRIBUTING.md    # Contribution and notebook review guidance
|-- .gitignore         # Excludes secrets, recordings, and generated artifacts
`-- .gitattributes     # Consistent line endings for source and notebook files
```

`requirements.txt` lists the Python dependencies, but installing it alone does not make the notebook a local application: `google.colab.drive` and `google.colab.userdata` are Colab-specific. Colab supplies PyTorch, IPython, and its own integration modules. If you also upload `requirements.txt`, you can install the listed dependencies with `%pip install -r requirements.txt` instead of the initial installation cell. Only Transformers is pinned; this is not a fully locked environment.

Running locally would require replacing the Drive mount and secret retrieval, configuring CUDA-compatible PyTorch and FFmpeg, and providing a compatible GPU.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `HF_TOKEN` is missing or inaccessible | Check the exact secret name and enable notebook access in Colab. |
| Audio file not found | Confirm Drive is mounted and `audio_filename` matches the uploaded file. |
| CUDA is unavailable | Select a GPU runtime and reconnect before running the notebook. |
| CUDA out of memory | Restart the runtime to release loaded models; try a shorter recording or smaller models. Both Whisper and Phi remain loaded in the current notebook. |
| BF16 / BFloat16 error | The notebook requests `torch.bfloat16` for Phi's quantized computation. On a GPU without BF16 support, try `torch.float16`. |
| Audio decoding fails | Check that FFmpeg is installed and that the recording is valid. |
| Dependency import fails after installation | Restart the runtime and rerun from the imports cell; retain the pinned Transformers version. |
| Minutes stop early or invent details | Check the transcript, shorten or split long recordings, and review every factual claim. |

## Limitations and data handling

- `whisper-medium.en` is intended for English audio. Transcription errors, overlapping speech, and unclear audio can affect the minutes.
- The notebook does not label speakers, verify attendees, or resolve action owners independently of the transcript.
- Long recordings are not split into separate summarization requests. Input size, GPU memory, and the generation limit can affect results.
- Recordings are stored in Google Drive and processed in a Google-hosted Colab runtime. This is not an offline workflow.
- Transcripts and minutes can contain private information. Clear notebook outputs before committing; `.gitignore` cannot remove information embedded in an `.ipynb` file.

## Models and attribution

- [OpenAI Whisper medium.en](https://huggingface.co/openai/whisper-medium.en): English speech recognition; model card lists Apache 2.0 licensing.
- [Microsoft Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct): minutes generation; model card lists MIT licensing.
- [Bitsandbytes quantization](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes): 4-bit NF4 quantization through Transformers.

Model licenses apply to those models separately. No license has been selected for this project's own code. Audio is not redistributed, and no rights to recordings are granted by this repository.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.
