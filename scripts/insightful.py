
"""
Insightface V0.7's operational components which are required,
de-pickled and no automatic download.
Modified swapper. Some compression of OO to functions.
"""

import os
import glob
import torch
import onnx
import onnxruntime
import numpy as np
import cv2
# SBM sqrt and atan2 / arctan2 is ~x10 faster in math than np,
# but on a scale of e^-7. I don't think that justifies an import, but whatever. 
import math

# SBM My constants, need to update them in extensions.
DIRMODELBASE = r"Swamper"
DIRMODELFIX = r"Swiper"
DIROBJECTS = r"Objects"

def update_dirs(dirbase, dirfix, dirobj):
    global DIRMODELBASE
    global DIRMODELFIX
    global DIROBJECTS
    DIRMODELBASE = dirbase
    DIRMODELFIX = dirfix
    DIROBJECTS = dirobj

# Copypasta from inswapper and heavily modded to allow batch.

# class Swapall(INSwapper):
class Swapall():
    """Face swapper with get converted to multi swaps.
    
    
    """
    
    # SBM Monkeypatching the model.
    @staticmethod
    def fix_model_batch(inpt = None, outdir = None):
        """Fix model so it accepts batch input, greatly improving gpu inference.
        
        I'm quite lazy so it just takes the last onnx in directory, no checks,
        sorted by filename ignoring extension.
        """
        # Defaults are no good, they need to be mutable.
        # I do not fancy partial functions in classes.
        if inpt is None: inpt = DIRMODELBASE
        if outdir is None: outdir = DIRMODELFIX
        # import onnx
        from onnx.tools import update_model_dims
        lswap = glob.glob(os.path.join(inpt, "*.onnx"))
        lswap = sorted(lswap, key = lambda x: os.path.splitext(x)[0])
        if len(lswap) == 0: # No files, cannot run.
            return
        # Best option: Fix already exists, return it.
        outnm = os.path.basename(lswap[-1])
        outnm = os.path.splitext(outnm)[0] + "-btc.onnx"
        outpt = os.path.join(outdir,outnm)
        if os.path.isfile(outpt):
            return outpt
        # Read base.
        model = onnx.load(lswap[-1])
        session = PickableInferenceSession(lswap[-1])
        inputs = session.get_inputs()
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        # Second option, found fixed file in base folder, disorganised but works.
        if input_shape[0] != 1:
            return lswap[-1]
        # Worst case, update input & output first dim to batchsize.
        # Creep: Read input + output dim names, update generically.
        # Not much point since it turned closed source, no more models.
        model = update_model_dims.update_inputs_outputs_dims(model,
                                {"target": ["batch", 3, 128, 128], "source": ["batch", 512]},
                                {"output": ["batch", 3, 128, 128]})
        onnx.save(model, outpt)
        
        return outpt # Fixed model.
    
    def __init__(self, model_file=None, session=None, checkmulti = True):
        # SBM Check if model is multi input/output capable.
        if checkmulti:
            vdir = os.path.dirname(model_file)
            # self.model_file = self.fix_model_batch(vdir, vdir)
            self.model_file = self.fix_model_batch(vdir, None)
        else:
            self.model_file = model_file
        if self.model_file != model_file:
            # Changed file, need to get another session.
            # Terrible monkeypatch, but don't see a better way. 
            session = PickableInferenceSession(self.model_file,
                                                session.get_providers(),
                                                session.get_provider_options())
        self.session = session
        model = onnx.load(self.model_file)
        graph = model.graph
        self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
        self.input_mean = 0.0
        self.input_std = 255.0
        #print('input mean and std:', model_file, self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        inputs = self.session.get_inputs()
        self.input_names = []
        for inp in inputs:
            self.input_names.append(inp.name)
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.output_names = output_names
        assert len(self.output_names)==1
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        print('inswapper-shape:', self.input_shape)
        self.input_size = tuple(input_shape[2:4][::-1])

    def forward(self, img, latent):
        img = (img - self.input_mean) / self.input_std
        pred = self.session.run(self.output_names, {self.input_names[0]: img, self.input_names[1]: latent})[0]
        return pred
    
    def pastaface(self, img, bgr_fake, aimg, M):
        """Pasta bgr on top of img given positioning.
        
        Single image only, not sure how well it'd work with cv / gpu. 
        """
        target_img = img
        fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2,:] = 0
        fake_diff[-2:,:] = 0
        fake_diff[:,:2] = 0
        fake_diff[:,-2:] = 0
        IM = cv2.invertAffineTransform(M)
        img_white = np.full((aimg.shape[0],aimg.shape[1]), 255, dtype=np.float32)
        bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        img_white[img_white>20] = 255
        fthresh = 10
        fake_diff[fake_diff<fthresh] = 0
        fake_diff[fake_diff>=fthresh] = 255
        img_mask = img_white
        mask_h_inds, mask_w_inds = np.where(img_mask==255)
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h*mask_w))
        k = max(mask_size//10, 10)
        #k = max(mask_size//20, 6)
        #k = 6
        kernel = np.ones((k,k),np.uint8)
        img_mask = cv2.erode(img_mask,kernel,iterations = 1)
        kernel = np.ones((2,2),np.uint8)
        fake_diff = cv2.dilate(fake_diff,kernel,iterations = 1)
        k = max(mask_size//20, 5)
        #k = 3
        #k = 3
        kernel_size = (k, k)
        blur_size = tuple(2*i+1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        k = 5
        kernel_size = (k, k)
        blur_size = tuple(2*i+1 for i in kernel_size)
        fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
        img_mask /= 255
        fake_diff /= 255
        #img_mask = fake_diff
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
        fake_merged = img_mask * bgr_fake + (1-img_mask) * target_img.astype(np.float32)
        fake_merged = fake_merged.astype(np.uint8)
        return fake_merged
        
        
    def get(self, img, target_face, source_face, paste_back=True, batchsize = None):
        """Multi swap.
        
        Parm variances: Img is list of orig target images (cba to update face_align),
        target_face is list of list of bboxes (should maybe just get kps)
         where each list corresponds to one image (multiple faces),
        source_face is list of cropped faces.
        Will expand source or target to the other lists.
        Flattens all source & targets for session, then faces are chain pasted
         according to target_face structure.
        Auto splits to batches on batchsize parm.
        """
        laimg = []
        lM = []
        lblob = []
        for trgimg, ltrgface in zip(img,target_face):
            for trgface in ltrgface:
                # aimg, M = insightface.utils.face_align.norm_crop2(trgimg, trgface.kps, self.input_size[0])
                aimg, M = norm_crop2(trgimg, trgface.kps, self.input_size[0])
                blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                              (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
                laimg.append(aimg)
                lM.append(M)
                lblob.append(blob)
        llatent = []
        for srcface in source_face:
            latent = srcface.normed_embedding.reshape((1,-1))
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)
            llatent.append(latent)
        
        if len(laimg) == 0 or len(llatent) == 0: # No faces.
            return img # Should return the face if not pasteback. Maybe none.
        elif len(laimg) == 1 and len(llatent) > 1:
            laimg = laimg * len(llatent)
            lM = lM * len(llatent)
            lblob = lblob * len(llatent)
        elif len(laimg) > 1 and len(llatent) == 1:
            llatent = llatent * len(laimg)
        elif len(laimg) != len(llatent):
            raise NotImplementedError("Mismatch in face counts.")
        
        if batchsize is None: # Run all at once.
            batchsize = len(lblob)
        
        # SBM Batch split prediction.
        pred = None
        for i in range(0, len(lblob), batchsize):
            mblob = np.concatenate(lblob[i:i + batchsize], axis = 0)
            mlatent = np.concatenate(llatent[i:i + batchsize], axis = 0)
            mpred = self.session.run(self.output_names, {self.input_names[0]: mblob, self.input_names[1]: mlatent})[0]
            if pred is None:
                pred = mpred
            else:
                pred = np.concatenate([pred,mpred], axis = 0)
        #print(latent.shape, latent.dtype, pred.shape)
        lbgr = []
        for i,_ in enumerate(laimg):
            img_fake = pred[i:i+1].transpose((0,2,3,1))[0]
            bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
            lbgr.append(bgr_fake)
        if not paste_back:
            return lbgr, lM
        else: # Pasta to each image.
            # Zip img + faces to get structure, and used a counter to get the results. 
            i = 0
            lpasta = []
            for (vimg, vtrg) in zip(img, target_face):
                pasta = vimg
                for _ in enumerate(vtrg):
                    pasta = self.pastaface(pasta, lbgr[i], laimg[i], lM[i])
                    i = i + 1
                lpasta.append(pasta) # Add finished image.
            return lpasta

# import numpy as np
from numpy.linalg import norm as l2norm
#from easydict import EasyDict

# Copypasta from common.

class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        #for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property 
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender==1 else 'F'
    
# Copypasta from face_analysis.

# SBM Not necessary? Py2 is long dead.
# from __future__ import division

# import glob
# import os.path as osp
#
# import numpy as np
# import onnxruntime
# from numpy.linalg import norm
#
# from ..model_zoo import model_zoo
# from ..utils import DEFAULT_MP_NAME, ensure_available
# from .common import Face

BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
MODEL_NEEDED = "buffalo_l.zip"
# Contains the following:
# 1k3d68: Landmark , task=landmark_3d_68
# 2d106det: Landmark, landmark_2d_106
# det_10g: RetinaFace, detection
# genderage: Attribute, genderage
# w600k_r50: ArcFaceONNX, recognition

# __all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, name="buffalo_l", root='~/.insightface', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        # self.model_dir = ensure_available('models', name, root=root)
        self.model_dir = name # SBM No auto dl.
        onnx_files = glob.glob(os.path.join(self.model_dir, '*.onnx'))
        # SBM BUG: Sorts ext's period before / after certain ascii chars. I sent a PR.
        # onnx_files = sorted(onnx_files)
        onnx_files = sorted(onnx_files, key = lambda x: os.path.splitext(x)[0])
        if len(onnx_files) == 0: # SBM Inform of download.
            print("You need to obtain the model {} from {} for analysis. Place in {}.".format(
                    MODEL_NEEDED, BASE_REPO_URL, name))
        for onnx_file in onnx_files:
            # model = model_zoo.get_model(onnx_file, **kwargs)
            model = get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

# Copypasta from modelzoo. Don't really need the router class.
# SBM Added defaults for providers here since it raises valueerror in my build.
class PickableInferenceSession(onnxruntime.InferenceSession): 
    # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path, 
                 providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                 provider_options = None, **kwargs):
        super().__init__(model_path, providers = providers,
                         provider_options = provider_options, **kwargs)
        self.model_path = model_path

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        model_path = values['model_path']
        self.__init__(model_path)

# def get_default_providers():
#     return ['CUDAExecutionProvider', 'CPUExecutionProvider']
#
# def get_default_provider_options():
#     return None

# SBM Rerout shortcut.
# providers = onnxruntime.get_available_providers()
# providers.remove("TensorrtExecutionProvider") # Troublesome installation.

# def get_model(onnx_file, **kwargs):
def get_model(onnx_file, providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'],
              provider_options = None):
    # session = PickableInferenceSession(onnx_file, **kwargs)
    session = PickableInferenceSession(onnx_file, providers = providers,
                                       provider_options = provider_options)
    inputs = session.get_inputs()
    input_cfg = inputs[0]
    input_shape = input_cfg.shape
    outputs = session.get_outputs()

    mdl = None
    if len(outputs)>=5:
        mdl = RetinaFace
    elif input_shape[2]==192 and input_shape[3]==192:
        mdl = Landmark
    elif input_shape[2]==96 and input_shape[3]==96:
        mdl = Attribute
    elif len(inputs)==2 and input_shape[2]==128 and input_shape[3]==128:
        mdl = Swapall # SBM Modded.
    elif input_shape[2]==input_shape[3] and input_shape[2]>=112 and input_shape[2]%16==0:
        mdl = ArcFaceONNX
    
    if mdl is not None:
        return mdl(model_file = onnx_file, session = session)

# Copypasta from arcface_onnx.

# from __future__ import division
# import numpy as np
# import cv2
# import onnx
# import onnxruntime
# from ..utils import face_align

# __all__ = [
#     'ArcFaceONNX',
# ]


class ArcFaceONNX:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = 'recognition'
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            #print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        #print('input mean and std:', self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names)==1
        self.output_shape = outputs[0].shape

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img, face):
        # aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        
        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

# Copypasta from utils.face_align.

from skimage import transform as trans

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def norm_crop2(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M

# SBM Renamed from transform.
def transform_face(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M

def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts

def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    #print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)

# Copypasta from utils.transform.

def estimate_affine_matrix_3d23d(X, Y):
    ''' Using least-squares solution 
    Args:
        X: [n, 3]. 3d points(fixed)
        Y: [n, 3]. corresponding 3d points(moving). Y = PX
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
    '''
    X_homo = np.hstack((X, np.ones([X.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(X_homo, Y)[0].T # Affine matrix. 3 x 4
    return P

def P2sRt(P):
    ''' decompositing camera matrix P
    Args: 
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation. 
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t

def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
    return rx, ry, rz

# Copypasta from attribute.
# from __future__ import division
# import numpy as np
# import cv2
# import onnx
# import onnxruntime
# from ..utils import face_align
#
# __all__ = [
#     'Attribute',
# ]


class Attribute:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            #print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
            if nid<3 and node.name=='bn_data':
                find_sub = True
                find_mul = True
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 128.0
        self.input_mean = input_mean
        self.input_std = input_std
        #print('input mean and std:', model_file, self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names)==1
        output_shape = outputs[0].shape
        #print('init output_shape:', output_shape)
        if output_shape[1]==3:
            self.taskname = 'genderage'
        else:
            self.taskname = 'attribute_%d'%output_shape[1]

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img, face):
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0]  / (max(w, h)*1.5)
        #print('param:', img.shape, bbox, center, self.input_size, _scale, rotate)
        # aimg, M = face_align.transform(img, center, self.input_size[0], _scale, rotate)
        aimg, M = transform_face(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        #assert input_size==self.input_size
        blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        pred = self.session.run(self.output_names, {self.input_name : blob})[0][0]
        if self.taskname=='genderage':
            assert len(pred)==3
            gender = np.argmax(pred[:2])
            age = int(np.round(pred[2]*100))
            face['gender'] = gender
            face['age'] = age
            return gender, age
        else:
            return pred

# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

# from __future__ import division
# import numpy as np
# import cv2
# import onnx
# import onnxruntime
# from ..utils import face_align
# from ..utils import transform
# from ..data import get_object
#
# __all__ = [
#     'Landmark',
# ]


class Landmark:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            #print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
            if nid<3 and node.name=='bn_data':
                find_sub = True
                find_mul = True
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 128.0
        self.input_mean = input_mean
        self.input_std = input_std
        #print('input mean and std:', model_file, self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names)==1
        output_shape = outputs[0].shape
        self.require_pose = False
        #print('init output_shape:', output_shape)
        if output_shape[1]==3309:
            self.lmk_dim = 3
            self.lmk_num = 68
            # self.mean_lmk = get_object('meanshape_68.pkl')
            self.mean_lmk = get_object("meanshape_68")
            self.require_pose = True
        else:
            self.lmk_dim = 2
            self.lmk_num = output_shape[1]//self.lmk_dim
        self.taskname = 'landmark_%dd_%d'%(self.lmk_dim, self.lmk_num)

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img, face):
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0]  / (max(w, h)*1.5)
        #print('param:', img.shape, bbox, center, self.input_size, _scale, rotate)
        aimg, M = transform_face(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        #assert input_size==self.input_size
        blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        pred = self.session.run(self.output_names, {self.input_name : blob})[0][0]
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num*-1:,:]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = trans_points(pred, IM)
        face[self.taskname] = pred
        if self.require_pose:
            # P = transform.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
            P = estimate_affine_matrix_3d23d(self.mean_lmk, pred)
            # s, R, t = transform.P2sRt(P)
            s, R, t = P2sRt(P)
            rx, ry, rz = matrix2angle(R)
            pose = np.array( [rx, ry, rz], dtype=np.float32 )
            face['pose'] = pose #pitch, yaw, roll
        return pred

# SBM Have to set dynamically since extension changes the global.
# Copypasta from pickle object, converted to txt for security reasons.
# Originally float32, I saved as float64 for better precision.
def get_object(name, objdir = None):
    # import pickle # I don't like this.
    # objects_dir = osp.join(Path(__file__).parent.absolute(), 'objects')
    # if not name.endswith('.pkl'):
    #     name = name+".pkl"
    if objdir is None: objdir = DIROBJECTS
    filepath = os.path.join(objdir, name + ".txt")
    if not os.path.exists(filepath):
        return None
    obj = np.loadtxt(filepath)
    # with open(filepath, 'rb') as f:
    #     obj = pickle.load(f)
    return obj

# Copypasta from retinaface.

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class RetinaFace:
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        if self.session is None:
            assert self.model_file is not None
            assert os.path.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        #print(input_shape)
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        #print('image_size:', self.image_size)
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.input_mean = 127.5
        self.input_std = 128.0
        #print(self.output_names)
        #assert len(outputs)==10 or len(outputs)==15
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        if len(outputs)==6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs)==9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs)==10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs)==15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in detection model, ignore')
            else:
                self.input_size = input_size

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name : blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                #solution-1, c style:
                #anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                #for i in range(height):
                #    anchor_centers[i, :, 1] = i
                #for i in range(width):
                #    anchor_centers[:, i, 0] = i

                #solution-2:
                #ax = np.arange(width, dtype=np.float32)
                #ay = np.arange(height, dtype=np.float32)
                #xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                #anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                #solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                #print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size = None, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
            
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
