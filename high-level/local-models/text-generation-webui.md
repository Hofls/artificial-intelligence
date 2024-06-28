# text-generation-webui

### Getting started
* https://github.com/oobabooga/text-generation-webui
* Download repository, run `start_windows.bat`
* Pick huggingface model, for example https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ
* Open Web UI - http://localhost:7860/?__theme=dark
* Download model:
    * Model -> Download -> TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ -> Download
* Load model to RAM:
    * Model -> Model loader -> AutoAWQ 
    * gpu-memory in MiB = 2000
    * cpu-memory in MiB = 2000
    * Model -> TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ -> Load
* (Optional) Parameters
    * Parameters -> Preset -> Divine Intellect
    * Parameters -> Chat -> Character / User
    * Chat -> Minimum reply length
* Start typing, AI will continue text:
    * Notebook -> Raw -> Generate
    * For example:
    ```
    List of top 20 tv series: 
    1. The Wire
    2. 
    ```
* Chat with Model:
    * Web UI -> Chat -> New Chat

### Important parameters
* 