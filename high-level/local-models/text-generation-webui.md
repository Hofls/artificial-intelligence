# text-generation-webui

### Getting started
* https://github.com/oobabooga/text-generation-webui
    * Wiki - https://github.com/oobabooga/text-generation-webui/wiki
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
    * Parameters -> Preset -> Divine Intellect (for instructions) / Midnight Enigma (for chat)
    * Parameters -> Chat -> Character / User
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

### Extensions
* long_replies - adds "Minimum reply length" parameter on Chat/Notebook tab
* openai - OpenAI rest API, [wiki](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API)
    * Request example:
    ```
    fetch('http://localhost:5000/v1/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          prompt: "This is a cake recipe:",
          max_tokens: 200,
          temperature: 1,
          top_p: 0.9
        })
    });
    ```
  
### Models
* https://huggingface.co/TheBloke
    * https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ
* https://huggingface.co/bartowski
    * https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF (Model loader - Llama.cpp)

### Important parameters
* temperature - randomness level 
    * 0 - only most likely token is used
    * 1 - average randomness
    * 5 - full random, incoherent rambling
* dry_multiplier - penalty for repetition
    * 0 - AI will often repeat itself
    * 5 - AI will never repeat itself
