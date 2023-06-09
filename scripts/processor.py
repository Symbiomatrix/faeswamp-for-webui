
import os
import glob
import cv2

from scripts.insightful import FaceAnalysis, get_model
import scripts.insightful as insightful
from importlib import reload
reload(insightful)

DIRMODELBASE = r"Swamper" # Useless.
DIRMODELRECG = r"Buffalo" # Useless.
DIRMODELFIX = r"Swiper" # Mostly useless.
DIROBJECTS = r"Objects" # Useless.
DIROUT = r"outputs/tempswamp"

# Ffmpeg commands.
# SBM I've removed minterpolate, flags from fmmerge. Don't see much use for the average user.
# Transpose is currently the only optional parm.
# SBM I don't really understand what's the input framerate do on merge.
# Interestingly, cv2 also has videocapture + videowriter which by def uses ffmpeg as backend,
# and supports its features through constants, but it's a drag to convert,
# plus hwaccel is not supported, all the quality tags must be controlled manually.

# cli_template = "ffmpeg -hide_banner -loglevel {loglevel} -hwaccel auto -y -framerate {framerate}
# -i \"{inpath}/%5d.jpg\" -r {videorate} {preset} {minterpolate} {flags}
# -metadata title=\"{description}\" -metadata description=\"{info}\"
# -metadata author=\"stable-diffusion\" -metadata album_artist=\"{author}\" \"{outfile}\""
FMSPLIT = "ffmpeg -i \"{vidpath}\" -r {videorate} {transpose} -qscale:v {qv} \"{frmpath}/{frmfmt}\" "
FMGIF = "ffmpeg -i \"{vidpath}\" -vsync 0 -qscale:v {qv} \"{frmpath}/{frmfmt}\""
FMMERGE = ("ffmpeg -hwaccel auto -y -framerate {framerate} -i \"{frmpath}/{frmfmt}\" {pad} "
           "-r {videorate} {preset} \"{outfile}\" ")

SPLITPARMS = ["vidpath", "videorate", "transpose", "qv", "frmpath", "frmfmt"]
MERGEPARMS = ["framerate", "frmpath", "frmfmt", "videorate", "preset", "minerpolate", "flags", "outfile", "pad"]
FRAMECNTDEF = 6 # Number of digits in filename.
FMTFRAME = "%0{}d.jpg"
FMTTRANS = "transpose={transpose}"
FMTPAD = "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2" # Fixes issue with encoding odd h/w. 
FMTFILTERS = "-vf \"{multifilter}\""

DEFSPL = {"transpose": None, "qv": 2, "frmpath": "tempf", #"frmfmt": FMTFRAME.format(FRAMECNTDEF),
          "videorate": 30}
DEFMRG = {"framerate": 30, "frmpath": "tempf", "videorate": 30, #"frmfmt": FMTFRAME.format(FRAMECNTDEF)
          "pad": ""}

# Codec presets. x264 is recommended for running on browsers, others have better compression.
# Each preset has a different quality factor, mostly crf.
# Cannot estimate crf from bitrate.
# Hw acceleration set to automatic, no real need to run nvenc (nvidia's).
FMPRESETS = {
    'x264': ('-vcodec libx264 -preset medium -crf {quality}', 23),
    'x265': ('-vcodec libx265 -preset faster -crf {quality}', 28),
    'vpx-vp9': ('-vcodec libvpx-vp9 -crf {quality} -b:v 0 -deadline realtime -cpu-used 4', 34),
    'aom-av1': ('-vcodec libaom-av1 -crf {quality} -b:v 0 -usage realtime -cpu-used 8 -pix_fmt yuv444p', 28),
    'prores_ks': (('-vcodec prores_ks -profile:v 3 -vendor apl0 -bits_per_mb {quality}'
                   ' -pix_fmt yuv422p10le'), 8000), 
    'nvenc': (('-vcodec hevc_nvenc -preset:v p7 -tune:v hq -rc:v vbr -cq:v {quality}'
               ' -b:v 0 -pix_fmt yuv420p'), 30), # SBM for nvidia
}

global analyser
global swamper
analyser = None
swamper = None

def get_landmarks(img_data):
    """Get list of landmarks, sorted by relevance score (probably).
    
    Spam.
    """
    analysed = analyser.get(img_data)
    return sorted(analysed, key=lambda x: x.det_score)

def load_swamper(fldir = None):
    """Load swamper model.
    
    Spam.
    """
    if fldir is None: fldir = DIRMODELFIX
    lswamp = glob.glob(os.path.join(fldir, "*.onnx"))
    lswamp = sorted(lswamp, key = lambda x: os.path.splitext(x)[0])
    if len(lswamp) > 0:
        return get_model(lswamp[-1])
    else:
        raise NotImplementedError("Main model missing. Get it and place in {}.".format(fldir))

def update_dirs(dirbase, dirrecg, dirfix, dirobj):
    """Update global dir parms.
    
    Spam.
    """
    global DIRMODELBASE
    global DIRMODELRECG
    global DIRMODELFIX
    global DIROBJECTS
    DIRMODELBASE = dirbase
    DIRMODELRECG = dirrecg
    DIRMODELFIX = dirfix
    DIROBJECTS = dirobj
    insightful.update_dirs(dirbase, dirfix, dirobj)

def load_models(diranalyser, dirswamper):
    """Loads models to the global scope, accessed by functions.
    
    Spam.
    """
    global analyser
    global swamper
    if diranalyser is not None:
        analyser = FaceAnalysis(name=diranalyser)
        analyser.prepare(ctx_id = 0, det_size = (640, 640))
    else:
        # analyser.session.close() # Each task has a session. Ugh.
        analyser = None
    if dirswamper is not None:
        swamper = load_swamper(dirswamper)
    else:
        # swamper.session.close() # And onnx sessions don't have close. Great.
        del swamper.session
        swamper = None
    return analyser, swamper

def listpick(l, idx):
    """Pick one item or all and wrap in list.
    
    Spam.
    """
    if idx >= 0:
        if idx >= len(l):
            idx = -1 # Default to last item.
        return [l[idx]]
    else:
        return [l]
    
fisgif = lambda x: "gif" in os.path.splitext(x)[-1].lower()
VIDEXT = [".mp4", ".mkv", ".mov"]
fisvid = lambda x: os.path.splitext(x)[-1].lower() in VIDEXT

def delete_frames(vdir, frmcnt = FRAMECNTDEF):
    """Delete all jpgs from folder.
    
    Filename pattern sought: ###..##.jpg.
    The number of digits is constant, glob only supports wildcards or ranges.
    """
    patname = "".join(["[0-9]"] * frmcnt)
    lfiles = glob.glob(os.path.join(vdir, patname + ".jpg"))
    for pt in lfiles:
        os.remove(pt)

def process_img(source_img, target_img, sidx = 0, tidx = -1):
    """Process single image.
    
    If indexes are passed, will pick one specific landmark.
    -1 will run on all landmarks.
    This function only safely suppoorts 1->M.
    """
    source_lm = listpick(get_landmarks(source_img["image"]),sidx)
    frame = listpick(target_img["image"],tidx)
    target_lms = [get_landmarks(frame[0])]
    result = swamper.get(frame, target_lms, source_lm, paste_back=True, batchsize = None)
    return result

fimg = lambda source_img, target_img: process_img(source_img, target_img)

def load_image(flpath):
    """Load image from file.
    
    Swaps colours to norm over cv's.
    """
    try:
        img = cv2.imread(flpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception: # Could not load.
        img = None
    return img

def save_image(img, flpath):
    """Save image to file.
    
    Spam.
    """
    # Cv's colour scheme is annoying.
    try:
        img = img["image"]
    except Exception:
        pass
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(flpath, img)

ffilename = lambda x: os.path.splitext(os.path.basename(x))[0]

def process_frames(source_img, target_dir, sidx = 0, tidx = -1, loadsize = 40, batchsize = 8,
                   dirsave = None, indret = True):
    """Process directory of target images, or list of images.
    
    Loads several images at a time, sends them to swamper to be processed in batches,
    returns the full list of frames.
    Optional: Landmark index, batchsizes, directory for intermediate saving of results
     (don't flood the ram, make it preemptible).
    Note that loadsize can be much bigger than batchsize, generally ram >> vram.
    However, it doesn't contribute much to speed, loading too many images beforehand.
    Still only single source. Landmark matching is a pain to add.
    """
    if dirsave is None: # Nowhere to save, would be foolish not to return.
        indret = True
    source_lm = listpick(get_landmarks(source_img["image"]),sidx)
    if isinstance(target_dir, list): # List of various frames.
        ltarget = target_dir
    else: # Directory.
        ltarget = os.listdir(target_dir)
        ltarget = [os.path.join(target_dir,f) for f in ltarget]
    # Detect & ignore partial results.
    if dirsave is not None:
        os.makedirs(dirsave, exist_ok = True)
        lsaved = set(ffilename(f) for f in os.listdir(dirsave))
        ltargetflt = [t for t in ltarget
                      if ffilename(t) not in lsaved]
    lres = []
    for i in range(0, len(ltargetflt), loadsize):
        ltrgbd = ltargetflt[i:i + loadsize] # Batch split.
        ltrgbf = [load_image(pt) for pt in ltrgbd] # Load images.
        ltrgbf = [t for t in ltrgbf if t is not None] # Remove non images.
        target_lms = [listpick(get_landmarks(frame), tidx)[0] for frame in ltrgbf]
        result = swamper.get(ltrgbf, target_lms, source_lm, paste_back=True, batchsize = batchsize)
        if dirsave is None:
            lres.extend(result)
        else:
            for (r,pt) in zip(result,ltrgbd):
                save_image(r,os.path.join(dirsave,os.path.basename(pt)))
    if dirsave is not None and indret: # Load all images once done.
        setarget = {ffilename(t) for t in ltarget}
        lres = [load_image(os.path.join(dirsave,pt)) for pt in os.listdir(dirsave) # Load images.
                if ffilename(pt) in setarget] # Exclude unrelated images, folder might be unclean.
        lres = [r for r in lres if r is not None] # Exclude non images.
    if indret:
        return lres
    return None

fframes = lambda source_img, target_dir: process_frames(source_img, target_dir, dirsave = DIROUT)

def video_framerate(vidpt):
    """Get framerate with cv2.
    
    Doesn't have a with syntax, oddly.
    """
    cap = cv2.VideoCapture(vidpt)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return framerate

def maxindex(pt, key = "idx"):
    """Finds maxindex in a path.
    
    To not use regex, idx is replaced with asterisk for glob,
    then the files are stripped from directory and all constant text,
    returning the idx alone. 
    Does not work accurately if key appears multiple times.
    """
    fmtst = os.path.basename(pt).split("{" + key + "}")[0]
    fmted = os.path.basename(pt).split("{" + key + "}")[1]
    if fmtst == fmted: # No key.
        return -1
    lfiles = glob.glob(pt.format(idx = "*"))
    if len(lfiles) == 0:
        return -1
    lfiles = [os.path.basename(f) for f in lfiles]
    lfiles = [f.split(fmtst, 1)[-1] for f in lfiles]
    lfiles = [f.rsplit(fmted, 1)[0] for f in lfiles]
    return max(lfiles)

def defnullneg(d, k, vdef):
    """Returns def if key not in dict, val is null or negative. 
    
    Spam.
    """
    if k not in d or d[k] is None:
        return vdef
    try: # Might not be numeric.
        if d[k] < 0:
            return vdef
    except Exception:
        pass
    return d[k]

def process_video(source_img, target_vid, sidx = 0, tidx = -1, loadsize = 40, batchsize = 8,
                   dirtemp = None, inddel = True, **kwargs):
    """Process video (given as filepath), splitting to frames and remerging.
    
    Saves the edited vid back to same directory as out.mp4 (cba to vary, could be setting).
    Allowed optional parms:
    Videorate: Number of frames to split and output framerate.
     If lower than input, will need to convert less frames and video will be more jerky.
    Transpose: two digit code for allowing upside down and any vertical flip combo.
     0 = ccw + flip, 1 = cw, 2 = ccw, 3 = cw + flip.
    Qv: Quality of frames. 2 takes far less space than uncompressed, and nearly unnoticeable.
    Frmcnt: Maximum number of frames in logscale (for format).
    Frmfmt: Frame format. Generally should be left alone.
    Preset: Codec. Some are browser compatible, some better compressed, some faster.
    Quality: Used by preset, controls quality of output and filesize.
    Framerate: Number of frames in a second as input for conversion.
     Can be lower / higher than input, but that doesn't seem useful since frames are just dropped.
    Handful of pathing parms controlled by the function.
    Sidx / tidx for selecting landmark index.
    Loadsize / batchsize: Rate of supplying frames and running the model on (ram / vram limited)
    Inddel: Delete all frames and edited frames on successful completion.
     If not set or erred, prediction will resume from existing files.
    Outfile: Path to output (dir + name + ext). Ffmpeg needs a valid codec.
    Special mode: gifs are extracted using vsync and merged to vid.
    """
    indgif = False
    if fisgif(target_vid):
        indgif = True
    # Dict editing.
    if dirtemp is None:
        dirtemp = os.path.dirname(target_vid) # Cont: Consider a different def.
    dirtout = os.path.join(dirtemp, "tout") # Output folder nested.
    os.makedirs(dirtout, exist_ok = True)
    kwargs["vidpath"] = target_vid
    kwargs["frmpath"] = dirtemp
    if "outfile" not in kwargs:
        kwargs["outfile"] = os.path.join(os.path.dirname(target_vid), "out.mp4")
    kwargs["outfile"] = kwargs["outfile"].format(idx = int(maxindex(kwargs["outfile"])) + 1)
    # x265 is more modern, but I'm getting "video does not have browser-compatible container or codec".
    kwargs["preset"] = defnullneg(kwargs, "preset", "x264")
    (pres, qual) = FMPRESETS[kwargs["preset"]] # Default quality factor.
    kwargs["quality"] = defnullneg(kwargs, "quality", qual) # Override with parm.
    kwargs["preset"] = pres.format(quality = kwargs["quality"])
    kwargs["frmcnt"] = defnullneg(kwargs, "frmcnt", FRAMECNTDEF)
    if "frmfmt" not in kwargs:
        kwargs["frmfmt"] = FMTFRAME.format(kwargs["frmcnt"])
    # Filters.
    if kwargs.get("transpose", None) is None:
        kwargs["transpose"] = ""
    else:
        ltrans = [FMTTRANS.format(transpose = vdir) for vdir in kwargs["transpose"]]
        strans = ", ".join(ltrans)
        kwargs["transpose"] = FMTFILTERS.format(multifilter = strans)
    
    kwargs["pad"] = defnullneg(kwargs, "pad", False)
    if kwargs["pad"]:
        kwargs["pad"] = FMTFILTERS.format(multifilter = FMTPAD)
    
    # Variable according to source.
    # Generally seems framerate = videorate is best.
    if kwargs.get("videorate", 0) < 0:
        framerate = video_framerate(target_vid)
        kwargs["videorate"] = framerate
        kwargs["framerate"] = framerate
    
    # Cont: Should dirtemp be cleared if inddel so we don't get stale frames?
    
    # Split parms.
    dparms = DEFSPL.copy()
    dparms.update({k:v for (k,v) in kwargs.items() if k in SPLITPARMS})
    if indgif:
        cmd = FMGIF.format(**dparms)
    else:
        cmd = FMSPLIT.format(**dparms)
    vret = os.system(cmd)
    if vret != 0:
        raise ValueError("Could not split video to frames.")
    # Process the frames.
    process_frames(source_img, dirtemp, sidx, tidx, loadsize, batchsize, dirsave = dirtout, indret = False)
    kwargs["frmpath"] = dirtout
    # Convert to video and return it. Optionally wipe temps.
    dparms = DEFMRG.copy()
    dparms.update({k:v for (k,v) in kwargs.items() if k in MERGEPARMS})
    cmd = FMMERGE.format(**dparms)
    os.makedirs(os.path.dirname(dparms["outfile"]), exist_ok = True)
    vret = os.system(cmd)
    if vret != 0:
        raise ValueError("Could not merge video from frames.")
    if inddel: # Delete frames + out optionally.
        # Creep: Should delete frmfmt files rather than assume format,
        # but ffmpeg's format is different from glob's.
        delete_frames(dirtemp, kwargs["frmcnt"])
        delete_frames(dirtout, kwargs["frmcnt"])
    return kwargs["outfile"]

fvideo = lambda source_img, target_vid: process_video(source_img, target_vid)

# with gr.Blocks() as demo:

if __name__ == "__main__":
    # demo.launch()
    print("Gradio is in main.")

