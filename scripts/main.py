
import os
import sys
import glob
import cv2
# import numpy as np
from importlib import reload
import gradio as gr
import modules.ui
import modules # SBM Apparently, basedir only works when accessed directly.
from modules import paths, scripts, shared, extra_networks
from modules.processing import Processed
from modules.script_callbacks import (on_ui_settings, on_ui_tabs)

import scripts.processor as processor
reload(processor) # update without restarting web-ui.bat

PTEXTENSION = modules.scripts.basedir()
ffulldir = lambda x: os.path.join(PTEXTENSION, x)

####### Script. (nonfunctional? marker?) #########

class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()

  def title(self):
    return "Faeswamper"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    return ()

######## Tab. ##########

# modes. Order must be maintained so dict is inadequate.
FSMODES = ["Image", "Video", "Directory"]
fgrprop = lambda x: {"label": x, "id": "t" + x, "elem_id": "FS_" + x}

# Should also be a list but cba. Fm has a weird rotation scheme.
DTRANS = {
"Normal": None, # Should be the same as 0.
"Clockwise": "1",
"Anticlockwise": "2",
"Upside down": "11",
"Flip": "01",
}

INDLOAD = True # Load models on first call.

def tab_router(selected_tab, inpimg, inptrg, inpvid, inpdir, sidx, tidx, *args):
    """Select processing method depending on current tab.
    
    Dir has a workaround for animated gifs and multiple videos (which gradio handles poorly in general).
    """
    global INDLOAD
    try:
        [batchsizev, codec, quality, videorate, framerate, transpose, pad, batchsized] = args
        transpose = DTRANS[transpose]
        indunload = fseti("unload")
        if INDLOAD:
            processor.update_dirs(fseti("dirswamp"),fseti("dirbuffalo"),
                                  fseti("dirswipe"),fseti("dirobjects"))
            try: # First check if there's a fixed version. Most graceful exit.
                processor.load_models(fseti("dirbuffalo"),fseti("dirswipe"))
            except NotImplementedError: # If not, try to fix the base.
                processor.load_models(fseti("dirbuffalo"),fseti("dirswamp"))
            INDLOAD = False
        # Make dictionary for parms to avoid copypasta.
        dvid = {'sidx': sidx,
                'tidx': tidx,
                'loadsize': fseti("loadsize"),
                'batchsize': batchsizev,
                'dirtemp': fseti("dirtemp"),
                'qv': fseti("qv"),
                'frmcnt': fseti("maxframe"),
                'preset': codec,
                'outfile': os.path.join(fseti("dirout"), fseti("outfile")),
                'videorate': videorate,
                'framerate': framerate,
                'transpose': transpose,
                'quality': quality,
                'pad': pad,
            }
        if selected_tab == "Image":
            vout = processor.process_img(inpimg, inptrg, sidx = sidx, tidx = tidx)
            vret = [vout, None]
        elif selected_tab == "Video":
            vout = processor.process_video(inpimg, inpvid, **dvid)
            vret = [None, vout]
        elif selected_tab == "Directory":
            # File component returns tempfile._TemporaryFileWrapper objects.
            # I dunno how to pass their io.bufferedrandom to cv, so just using the names.
            # This seems to be the only way to pass around animated gifs, so that's what I'm using.
            # Sucks that video parms aren't present in this tab, but whatever.
            vret = [None, None]
            dvid["batchsize"] = batchsized # Switch to dir mode's batchsize for convenience.
            inpdir = [f.name for f in inpdir]
            inpdvid = [f for f in inpdir if processor.fisvid(f)]
            inpdimg = [f for f in inpdir if not processor.fisvid(f)]
            if len(inpdir) == 1 and processor.fisgif(inpdir[0]): # Gif mode.
                vout = processor.process_video(inpimg, inpdir[0], **dvid)
                vret = [None, vout]
            elif len(inpdimg) > 0: # Single images processed and added to gallery.
                vout = processor.process_frames(inpimg, inpdimg, sidx = sidx, tidx = tidx,
                                                loadsize = fseti("loadsize"), batchsize = batchsized,
                                                dirsave = fseti("dirout"))
                vret = [vout, None]
            for ivid in inpdvid: # Videos are processed but not displayed.
                vout = processor.process_video(inpimg, ivid, **dvid)
        else:
            print("Wrong mode.")
            vret = [None, None]
    except Exception as e:
        import traceback
        print("Couldn't run swamper", e, traceback.format_exc())
        vret = [None, None]
        
    if indunload:
        processor.load_models(None, None)
        INDLOAD = True
        
    return vret

def ui_tab(mode):
    """Structures components for mode tab.
    
    Semi harcoded but it's clearer this way.
    """
    vret = None
    if mode == "Image":
        with gr.Row():
            inptrg = gr.Image(label="Target", source="upload", interactive=True, type="numpy", tool="sketch")
        vret = [inptrg]
    elif mode == "Video":
        with gr.Row():
            inpvid = gr.Video(label="Target Video", source="upload")
        with gr.Row():
            batchsizev = gr.Slider(label ="Batchsize", minimum = 1, maximum = 40, step = 1, value = 8)
        with gr.Row():
            lcod = list(processor.FMPRESETS.keys())
            codec = gr.Radio(label = "Codec", choices = lcod, value = lcod[0])
            quality = gr.Slider(label = "Quality", minimum = -1, maximum = 60, step = 0.1, value = -1)
        with gr.Row():
            videorate = gr.Slider(label = "Video framerate", minimum = -1, maximum = 240, step = 0.1, value = -1)
            framerate = gr.Slider(label = "Split rate", minimum = -1, maximum = 240, step = 0.1, value = -1)
        with gr.Row():
            ltrans = list(DTRANS.keys())
            transpose = gr.Radio(label = "Rotate", choices = ltrans, value = ltrans[0])
        with gr.Row():
            pad = gr.Checkbox(label = "Pad", interactive = True, value = False)
        vret = [inpvid, batchsizev, codec, quality, videorate, framerate, transpose, pad]
    elif mode == "Directory":
        with gr.Row():
            # Types crummy. Only a suggestion on the open interface, still allows drag&drop. #3767
            inpdir = gr.File(label = "Multi images", file_count = "multiple", file_types = ["image"])
        with gr.Row():
            batchsized = gr.Slider(label ="Batchsize", minimum = 1, maximum = 40, step = 1, value = 8)
        vret = [inpdir, batchsized]
    
    return vret

def ext_on_ui_tabs():
    """Builds gui, as a tab with media components and monitored tab inside.
    
    Spam.
    """
    with gr.Blocks(analytics_enabled=False) as faeswamp:
        with gr.Row():
            with gr.Column():
                inpimg = gr.Image(label="Source", source="upload", interactive=True, 
                                  type="numpy", tool="sketch")
                srcidx = gr.Slider(label = "Source selection", minimum = 0, maximum = 20, step = 1, value = 0)
                trgidx = gr.Slider(label = "Target selection", minimum = -1, maximum = 20, step = 1, value = -1)
                # Tabbed modes.
                with gr.Tabs(elem_id="FS_mode"):
                    selected_tab = gr.State("Image") # State component to document current tab for gen.
                    # ltabs = []
                    ltabp = []
                    for (i, md) in enumerate(FSMODES):
                        with gr.TabItem(**fgrprop(md)) as tab: # Tabs with a formatted id.
                            # ltabs.append(tab)
                            ltabp.append(ui_tab(md))
                        # Tab switch tags state component.
                        tab.select(fn = lambda tabnum = i: FSMODES[tabnum],
                                   inputs=[], outputs=[selected_tab])
                        
                    [inptrg] = ltabp[0]
                    [inpvid, batchsizev, codec, quality, videorate, framerate, transpose, pad] = ltabp[1]
                    [inpdir, batchsized] = ltabp[2]
                
                btn = gr.Button("Swamp")
            with gr.Column():
                # outimg = gr.Image(label="Output", interactive=False)
                outimg = gr.Gallery(label="Output", interactive = False).style(preview=False, container=False,
                                                                               columns=[1,2,3,4,5,6])
                outvid = gr.Video(label="Output vid", interactive=False)
        
        btn.click(
            fn=tab_router,
            inputs=[selected_tab, inpimg, inptrg, inpvid, inpdir, srcidx, trgidx,
                    *ltabp[0][1:], *ltabp[1][1:], *ltabp[2][1:]],
            outputs=[outimg, outvid],
        )
        
    return [(faeswamp, "Faeswamp", "faeswamp")]

######## Settings. #########

EXTKEY = "faeswamp"
EXTNAME = "Faeswamp"
EXTSETS = [
("debug", "Enable debug mode", "check", dict()),
("dirswamp", "Location of swamper model", "textb",
 dict(vdef = ffulldir("Swamper"))),
("dirbuffalo", "Location of buffalo model", "textb",
 dict(vdef = ffulldir("Buffalo"))),
("dirswipe", "Location of corrected model", "textb",
 dict(vdef = ffulldir("Swiper"))),
("dirobjects", "Location of extra objects", "textb",
 dict(vdef = ffulldir("Objects"))),
("dirout", "Location of result saving", "textb",
 dict(vdef = "outputs/tempswamp")),
("dirtemp", "Location of temp frame saving", "textb",
 dict(vdef = "outputs/tempframe")),
("qv", "Quality of frame extraction", "slider",
 dict(vdef = 2, minimum = 1, maximum = 31, step = 0.1)),
("maxframe", "Maximum number of frames (logarithmic, eg 3 -> 999)", "slider",
 dict(vdef = 6, minimum = 1, maximum = 10)),
("loadsize", "Number of frames to load for conversion (-1 for all)", "slider",
 dict(vdef = 40, minimum = -1, maximum = 200)),
("outfile", "Output video file format", "textb",
 dict(vdef = "out-i{idx}.mp4")),
("unload", "Unload onnx models after each run, slower but maybe frees vram", "check",
 dict(vdef = True)),
]

# Dynamically constructed list of default values, because shared doesn't allocate a value automatically.
# (id: def)
DEXTSETV = dict()
fseti = lambda x: shared.opts.data.get(EXTKEY + "_" + x, DEXTSETV[x])

class Setting_Component():
    """Creates gradio components with some standard req values.
    
    All must supply an id (used in code), label, component type. 
    Default value and specific type settings can be overridden. 
    """
    section = (EXTKEY, EXTNAME)
    def __init__(self, cid, clabel, ctyp, vdef = None, **kwargs):
        self.cid = EXTKEY + "_" + cid
        self.clabel = clabel
        self.ctyp = ctyp
        method = getattr(self, self.ctyp)
        method(**kwargs)
        if vdef is not None:
            self.vdef = vdef
        
    def get(self):
        """Get formatted setting.
        
        Input for shared.opts.add_option().
        """
        if self.ctyp == "textb":
            return (self.cid, shared.OptionInfo(self.vdef, self.clabel, section = self.section))
        return (self.cid, shared.OptionInfo(self.vdef, self.clabel,
                                            self.ccomp, self.cparms, section = self.section))
    
    def textb(self, **kwargs):
        """Textbox unusually requires no component.
        
        Spam.
        """
        self.ccomp = gr.Textbox
        self.vdef = ""
        self.cparms = {}
        self.cparms.update(kwargs)
    
    def check(self, **kwargs):
        self.ccomp = gr.Checkbox
        self.vdef = False
        self.cparms = {"interactive": True}
        self.cparms.update(kwargs)
        
    def slider(self, **kwargs):
        self.ccomp = gr.Slider
        self.vdef = 0
        self.cparms = {"minimum": 1, "maximum": 10, "step": 1}
        self.cparms.update(kwargs)

def ext_on_ui_settings():
    """Dump settings to gui from list.
    
    Spam.
    """
    for (cid, clabel, ctyp, kwargs) in EXTSETS:
        comp = Setting_Component(cid, clabel, ctyp, **kwargs)
        opt = comp.get()
        shared.opts.add_option(*opt)
        DEXTSETV[cid] = comp.vdef

# Create tab, create settings.
on_ui_tabs(ext_on_ui_tabs)
on_ui_settings(ext_on_ui_settings)
