### Getting started
* https://github.com/AUTOMATIC1111/stable-diffusion-webui#automatic-installation-on-windows
* Edit `webui-user.bat`, add:
    * `set COMMANDLINE_ARGS=--precision full --no-half --medvram`
    * Some cards generate only black images, fix - `--precision full --no-half`
    * Some cards have low ram, fix - `--medvram` (or `--lowvram`, but its very slow)
* Edit `ui-config.json`, replace:
    * `512` with `448`
    * `Batch size/value": 1,` with `Batch size/value": 2,`
* Run `webui-user.bat`, open [localhost](http://127.0.0.1:7860)

### Features
##### img2img
* `Redraw whole image`
    * Parameters:
        * `Sampling Steps` = 20 (halfway there) to 50 (fully done)
        * `CFG Scale` = 7 (wide variety) to 20 (strictly prompt)
        * `Denoising strength` = 0.25 (recognizable) to 0.5 (heavily changed)
* `Inpaint a part of image`
    * Parameters:
        * `CFG Scale` = 7 to 30 (max)
        * `Denoising strength` = 1.0 (max)
* `SD upscale`
    * `Upscaler` - anything but None
    * `Denoising strength` - 0
* `Script - Poor man's outpainting`
    * `CFG Scale` = max
    * `Denoising strength` = max
* 

### Recipes:
* `Inpaint -> Redraw`
    * `Inpaint` - Generate parts of image (e.g. background, eyes, hat)
    * `Redraw` - Apply common style to an image

### Prompts
* brown fluffy cat, dream symmetry, stunning portrait, by victo ngai, kilian eng vibrant colours, dynamic lighting, digital art, winning award masterpiece, fantastically beautiful, illustration, aesthetically inspired by beksinski and dan mumford, trending on artstation, art by greg rutkowski, 8k
* brown fluffy cat, bubbles, in the style of artgerm, charlie bowater, atey ghailan and mike mignola, vibrant colors and hard shadows and strong rim light, plain background, comic cover art, trending on artstation
* brown fluffy cat, digital painting, old english, whimsical background by marc simonetti, artwork by liam wong
