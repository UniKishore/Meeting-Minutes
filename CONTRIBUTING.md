# Contributing

Keep changes focused on the audio-to-minutes notebook workflow. For larger features, open an issue describing the intended behavior before implementing them.

## Making a change

1. Fork the repository and create a branch for your change.
2. Follow the Colab setup in [README.md](README.md).
3. Make the change and update the README if setup, prompts, dependencies, or output behavior changes.
4. Keep the notebook's installation cell and `requirements.txt` consistent when changing dependencies. Packages already supplied by Colab do not need to be reinstalled in that cell.
5. Check that the notebook runs from a fresh GPU runtime, in order, using a short English recording you have permission to process.
6. Check the transcript and minutes against the recording; verify that the requested sections appear and unsupported details are identified during review.
7. Clear all notebook outputs and save before opening a pull request.

## Pull requests

Describe what changed, why it helps, and how you checked it. For notebook or model changes, include the GPU/runtime used and any limitations observed. State explicitly if you could not run GPU inference.

Do not include tokens, `.env` files, recordings, downloaded model weights, or private meeting content. Notebook outputs are stored inside the notebook itself and require manual clearing even when recordings are ignored by Git.

There is currently no automated inference test suite. Documentation-only changes can be checked by reviewing links, commands, paths, and consistency with the notebook.
