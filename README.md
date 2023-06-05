## FaeSwamp for webui

Creates a tab which allows running a certain onnx model to process an image, video or multiple images efficiently.

Based on onnx through [insightface V0.7](https://github.com/deepinsight/insightface), but does not have the latter as a requirement.

Some of the gradio bits were templated from other extensions.

### Features:

- Runs on single image, video, directory = select images (though gradio's interface is rather clunky).
  A gif can be placed in directory and will be treated as a video (YMMV, works on my end).
- Gpu friendly: Native support for batch processing (via insightful, and base model is modified at runtime to allow it).
  Meaning, it should be very fast with a recent gpu and high batchsize. Only tested with nvidia.
- Control over source / target selection (either single or multiple targets).
- Fine grained control over video split & merge: codecs, quality, input / output framerate, rotation, padding.
  Some settings have automatic values as -1.
- Settings to control model directories, filename to some extent.
- Automatically unloads models optionally once the processing is complete. Not sure whether the vram is freed though.
- Not too painful to install: Other than onnx (which seems to require [visual cpp 2019 runtime on windows](https://onnxruntime.ai/docs/install/#requirements)), no building wheels or getting devkits. Hopefully. Untested.
- Localised what I could find.
- Gui hanging on errors, obviously intentional cus you've earned it.

### Install

Requires onnx, onnxruntime (gpu), opencv, and two onnx models.

### What models?

BATTERIES NOT INCLUDED. Find them yourself. Consider it gatekeeping.

### DISCLAIMER

This code is provided as is, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software. By using this code, you assume all liability and responsibility for any consequences that may arise from your use of the code.

![FaeswamperV1](https://github.com/Symbiomatrix/faeswamp-for-webui/assets/41131377/27e94a6b-f6a9-4a83-a9fa-253a5ec5071c)
