import os
import json
import pickle

import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel


class PanoCLIP:

    def __init__(self, model_name, panos_dir=None, device='cpu', logit_scale=2.6592, fov=60, height=460, width=800, cache_dir='features'):
        if model_name == 'clip':
            model_name = "openai/clip-vit-large-patch14"
        elif model_name == 'openclip':
            model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        else:
            ValueError(f'model name not found: {model_name}')

        self.model_name = model_name

        self.device = device
        self.model = None
        self.processor = None
        self.panos_dir = panos_dir

        self.logit_scale = logit_scale

        self.fov = fov
        self.height = height
        self.width = width
        self.angles = [-90, -45, 0, 45, 90]

        self.cache_dir = os.path.join(cache_dir, self.model_name)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_cache()

        flow_of_traffic_images_path = os.path.join(self.cache_dir, 'flow_of_traffic_images.pickle')
        try:
            with open(flow_of_traffic_images_path, 'rb') as f:
                self.flow_of_traffic_images = pickle.load(f)
        except FileNotFoundError:
            print('traffic flow images not found at: ', flow_of_traffic_images_path)
            self.flow_of_traffic_images = None

        self.num_queries = dict(landmarks=0, images=0)
        self.num_cache_hits = dict(landmarks=0, images=0)
        self.cached_modified = dict(landmarks=False, images=False, mean_std=False)

    def get_model(self):
        if self.model is None:
            print('load CLIP model')
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        return self.model

    def get_processor(self):
        if self.processor is None:
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        return self.processor

    def _load_cache(self):
        meta = dict(fov=self.fov,
                    height=self.height,
                    width=self.width,
                    model_name=self.model_name,
                    angles=self.angles)
        self.cache = dict(meta=meta, landmarks=dict(), images=dict(), mean_std=dict())

        fov = meta['fov']
        height = meta['height']
        width = meta['width']

        self.cache_meta_file = f"meta_{fov}_{height}_{width}.json"
        self.cache_meta_file = os.path.join(self.cache_dir, self.cache_meta_file)
        self.cache_landmarks_file = f"landmarks.pickle"
        self.cache_landmarks_file = os.path.join(self.cache_dir, self.cache_landmarks_file)
        self.cache_mean_std_file = f"landmarks_mean_std.pickle"
        self.cache_mean_std_file = os.path.join(self.cache_dir, self.cache_mean_std_file)
        self.cache_images_file = f"images_{fov}_{height}_{width}.pickle"
        self.cache_images_file = os.path.join(self.cache_dir, self.cache_images_file)

        if os.path.isfile(self.cache_meta_file):
            with open(self.cache_meta_file, 'r') as f:
                self.cache['meta'] = json.load(f)
        if os.path.isfile(self.cache_landmarks_file):
            with open(self.cache_landmarks_file, 'rb') as f:
                self.cache['landmarks'] = pickle.load(f)
        if os.path.isfile(self.cache_mean_std_file):
            with open(self.cache_mean_std_file, 'rb') as f:
                self.cache['mean_std'] = pickle.load(f)
        if os.path.isfile(self.cache_images_file):
            with open(self.cache_images_file, 'rb') as f:
                print('load cached images from :' + f.name)
                self.cache['images'] = pickle.load(f)

        for key in meta:
            assert meta[key] == self.cache['meta'][key]

        self.num_landmarks_cache = len(self.cache['landmarks'])
        self.num_panos_cache = len(self.cache['images'])
        self.num_mean_std_cache = len(self.cache['mean_std'])
        print('loaded PanoCLIP cache with {} landmarks and {} pano_headings and {} mean_std'.format(self.num_landmarks_cache,
                                                                                                    self.num_panos_cache,
                                                                                                    self.num_mean_std_cache))

    def save_cache(self):
        cache_saved = False

        # write cache files
        with open(self.cache_meta_file, 'w') as f:
            json.dump(self.cache['meta'], f)

        if self.cached_modified['landmarks']:
            with open(self.cache_landmarks_file, 'wb') as f:
                pickle.dump(self.cache['landmarks'], f)
            cache_saved = True
        self.cached_modified['landmarks'] = False

        if self.cached_modified['mean_std']:
            with open(self.cache_mean_std_file, 'wb') as f:
                pickle.dump(self.cache['mean_std'], f)
            cache_saved = True
        self.cached_modified['mean_std'] = False

        if self.cached_modified['images']:
            with open(self.cache_images_file, 'wb') as f:
                pickle.dump(self.cache['images'], f)
            cache_saved = True
        self.cached_modified['images'] = False

        if cache_saved:
            num_landmarks = len(self.cache['landmarks'])
            num_mean_std = len(self.cache['mean_std'])
            num_panos = len(self.cache['images'])
            print('saved PanoCLIP cache with {} landmarks and {} pano_headings and {} mean_std'.format(num_landmarks,
                                                                                                       num_panos,
                                                                                                       len(self.cache['mean_std'])))

            print('added {} landmarks and {} panos and  {} mean_std'.format(num_landmarks - self.num_landmarks_cache,
                                                                            num_panos - self.num_panos_cache,
                                                                            num_mean_std - self.num_mean_std_cache))

            self.num_landmarks_cache = num_landmarks
            self.num_mean_std_cache = num_mean_std
            self.num_panos_cache = num_panos

    def get_scores(self, panoid, pano_yaw, heading, landmarks, prompt, ignore_cache=False):
        assert type(heading) == int
        landmark_scores = dict()
        if len(landmarks) == 0:
            return landmark_scores

        self.num_queries['landmarks'] += len(landmarks)
        self.num_queries['images'] += 1

        landmarks_embeds = self.get_text_embeds(landmarks, prompt, ignore_cache=ignore_cache)
        image_embeds = self.get_image_embeds(panoid, pano_yaw, heading, ignore_cache=ignore_cache)

        logits_per_landmark = self.get_logits_per_landmark(landmarks_embeds, image_embeds)

        assert len(landmarks) == len(logits_per_landmark)
        for landmark, scores in zip(landmarks, logits_per_landmark):
            landmark_scores[landmark] = scores.cpu().detach()
            #print('query clip for', panoid, heading, landmark)

        return landmark_scores

    def transpose_imgs(self, imgs):
        new_imgs = []
        for img in imgs:
            img = np.transpose(img, (2, 0, 1))
            new_imgs.append(img)

        return new_imgs

    def get_scores_v2(self, imgs, landmarks, prompt, ignore_cache=False):
        landmarks_scores = dict()
        if len(landmarks) == 0:
            return landmarks_scores

        self.num_queries['landmarks'] += len(landmarks)
        self.num_queries['images'] += 1

        imgs = self.transpose_imgs(imgs)

        landmarks_embeds = self.get_text_embeds(landmarks, prompt)
        image_embeds = self.get_image_embeds_v2(imgs)

        logits_per_landmark = self.get_logits_per_landmark(landmarks_embeds, image_embeds)
        assert len(landmarks) == len(logits_per_landmark)
        for landmark, scores in zip(landmarks, logits_per_landmark):
            landmarks_scores[landmark] = scores.cpu().detach()

        return landmarks_scores

    def get_pano_slices(self, pano_image_path, pano_yaw, pano_heading):
        heading = pano_heading - pano_yaw
        heading = heading % 360

        slice_headings = [heading + angle for angle in self.angles]
        equ = Equirectangular(pano_image_path)
        images = list()
        for i, slice_heading in enumerate(slice_headings):
            img = equ.get_perspective(FOV=self.fov,
                                      THETA=slice_heading,
                                      PHI=0,
                                      height=self.height,  # 460, 600
                                      width=self.width)
            images.append(img)
        return images

    def get_logits_per_landmark(self, text_embeds, image_embeds):
        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t())
        if self.logit_scale is not None:
            logits_per_text *= torch.exp(torch.tensor(self.logit_scale))

        return logits_per_text

    def get_landmarks_mean_std_v2(self, landmarks_scores):
        landmarks_mean_std = {}
        for landmark in landmarks_scores:
            mean = float(landmarks_scores[landmark].mean())
            std = float(landmarks_scores[landmark].std())
            landmarks_mean_std[landmark] = (mean, std)

        return landmarks_mean_std

    def get_landmarks_mean_std(self, landmarks, prompt, panoids, ignore_cache=False):
        mean_stds = dict()
        for landmark in landmarks:
            mean_stds[landmark] = self.get_landmark_mean_std(landmark, prompt, panoids, ignore_cache)
        return mean_stds

    def get_landmark_mean_std(self, landmark, prompt, panoids, ignore_cache=False):
        landmark_cache_key = prompt.format(landmark)
        if not ignore_cache and landmark_cache_key in self.cache['mean_std']:
            return self.cache['mean_std'][landmark_cache_key]

        text_embeds = self.get_text_embeds([landmark], prompt)

        scores = list()
        pano_headings = list()
        for pano_heading in self.cache['images'].keys():
            panoid = pano_heading.rsplit('_', 1)[0]

            if panoid in panoids:
                pano_headings.append(pano_heading)

        for i, pano_heading in enumerate(pano_headings):
            image_embeds = self.cache['images'][pano_heading]
            logits = self.get_logits_per_landmark(text_embeds, image_embeds)[0]
            scores.append(logits)
        scores = torch.concat(scores, dim=0).flatten()
        mean = float(scores.mean())
        std = float(scores.std())

        self.cache['mean_std'][landmark_cache_key] = (mean, std)
        self.cached_modified['mean_std'] = True
        return mean, std

    def get_text_embeds(self, landmarks, prompt, ignore_cache=False):
        text_embeds = list()

        for landmark in landmarks:
            landmark_cache_key = prompt.format(landmark)

            text_emd = None
            if not ignore_cache:
                text_emd = self._read_cache(landmark_cache_key, 'landmarks')

            if text_emd is None:
                text = prompt.format(landmark)
                inputs = self.get_processor()(text=[text], return_tensors="pt", padding=True)
                output = self.get_model().get_text_features(input_ids=inputs['input_ids'].to(self.device),
                                                            attention_mask=inputs['attention_mask'].to(self.device))
                text_emd = output[0].detach()
                text_emd = text_emd / text_emd.norm(p=2, dim=-1, keepdim=True)
                text_emd = text_emd.cpu()

                self._write_cache(landmark_cache_key, text_emd, 'landmarks')
            text_embeds.append(text_emd)

        return torch.stack(text_embeds, dim=0)

    def get_image_embeds(self, panoid, pano_yaw, pano_heading, ignore_cache=False):
        cache_key = f'{panoid}_{pano_heading}'

        image_embeds = None
        if not ignore_cache:
            image_embeds = self._read_cache(cache_key, 'images')

        if image_embeds is None:
            assert self.panos_dir is not None
            pano_image_path = os.path.join(self.panos_dir, panoid + '.jpg')
            images = self.get_pano_slices(pano_image_path, pano_yaw, pano_heading)

            inputs = self.get_processor()(images=images,
                                          return_tensors="pt",
                                          padding=True)
            output = self.get_model().get_image_features(pixel_values=inputs['pixel_values'].to(self.device))
            image_embeds = output.detach()
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            image_embeds = image_embeds.cpu()

            self._write_cache(cache_key, image_embeds, 'images')
        return image_embeds

    def get_image_embeds_v2(self, images):
        inputs = self.get_processor()(images=images,
                                      return_tensors="pt",
                                      padding=True)
        image_embeds = self.get_model().get_image_features(pixel_values=inputs['pixel_values'].to(self.device)).detach()
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        image_embeds = image_embeds.cpu()

        return image_embeds

    def _write_cache(self, cache_key, value, name):
        assert name in ['landmarks', 'images']
        self.cached_modified[name] = True
        self.cache[name][cache_key] = value

        num_landmarks_cache = len(self.cache['landmarks'])
        num_panos_cache = len(self.cache['images'])

        if num_landmarks_cache - self.num_landmarks_cache >= 500:
            self.save_cache()
        if num_panos_cache - self.num_panos_cache >= 500:
            self.save_cache()

    def _read_cache(self, cache_key, name):
        assert name in ['landmarks', 'images']
        if cache_key in self.cache[name]:
            #if name == 'images':
                #print(f'read {name[:-1]}: {cache_key} from cache')
            self.num_cache_hits[name] += 1
            return self.cache[name][cache_key]
        print(f'not in cache: {cache_key}')
        return None

class Equirectangular:
    # https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py
    def __init__(self, img_name):
        im_cv = cv2.imread(img_name, cv2.IMREAD_ANYCOLOR)
        self._img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        [self._height, self._width, _] = self._img.shape
        # cp = self._img.copy()
        # w = self._width
        # self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        # self._img[:, w/8:, :] = cp[:, :7*w/8, :]

    def get_perspective(self, FOV, THETA, PHI, height, width, RADIUS=128):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([height, width, 3], np.float32)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float32)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool_)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool_)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy
        # for x in range(width):
        #    for y in range(height):
        #        cv2.circle(self._img, (int(lon[y, x]), int(lat[y, x])), 1, (0, 255, 0))
        # return self._img

        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)
        return persp
