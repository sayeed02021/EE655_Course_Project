import torch
import math

def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.
    The image data is assumed to be in the range of (0, 1).
    Args:
        image (torch.Tensor): RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps (float, optional): scalar to enforce numarical stability. Default: 1e-6.
    Returns:
        torch.Tensor: HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    # The first or last occurance is not guarenteed before 1.6.0
    # https://github.com/pytorch/pytorch/issues/20414
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    # s: torch.Tensor = deltac
    s: torch.Tensor = deltac / (maxc + eps)

    # avoid division by zero
    deltac = torch.where(deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]

    h = torch.stack([bc - gc, 2.0 * deltac + rc - bc, 4.0 * deltac + gc - rc], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.
    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.
    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out


class ToHSV(object):

    def __call__(self, pic):
        """RGB image to HSV image"""
        # return rgb_to_hsv_mine(pic)
        return rgb_to_hsv(pic)

    def __repr__(self):
        return self.__class__.__name__+'()'
    

class ToRGB(object):
    def __call__(self, img):
        """HSV image to RGB image"""
        return hsv_to_rgb(img)

    def __repr__(self) -> str:
        return self.__class__.__name__+'()'
    

# class ToComplex(object):
#     def __call__(self, img):
#         hue = img[..., 0, : , :]
#         sat = img[..., 1, : , :]
#         val = img[..., 2, : , :]


#         # tmp = 2*math.pi - hue

#         # hue_mod = torch.where(hue<=math.pi, hue, tmp)


#         # real_1 = sat * hue
#         # # real_1 = sat * hue_mod
#         # real_2 = sat * torch.cos(hue)
#         # real_3 = val


#         # imag_1 = val
#         # imag_2 = sat * torch.sin(hue)
#         # imag_3 = sat

#         hue_real = val
#         hue_imag = sat

#         sat_real = sat*hue
#         sat_imag = val

#         val_real = sat*torch.cos(hue)
#         val_imag = sat*torch.sin(hue)



#         real = torch.stack([hue_real, sat_real, val_real], dim= -3)
#         imag = torch.stack([hue_imag, sat_imag, val_imag], dim= -3)

#         comp_tensor = torch.complex(real, imag)

#         assert comp_tensor.dtype == torch.complex64
#         return comp_tensor

#     def __repr__(self):
#         return self.__class__.__name__+'()'


class ToComplex(object):
    def __call__(self, img):
        hue = img[..., 0, :, :]
        sat = img[..., 1, :, :]
        val = img[..., 2, :, :]

        # Directly create real and imaginary components
        real_part = torch.stack([val, sat * hue, sat * torch.cos(hue)], dim=-3)
        imag_part = torch.stack([sat, val, sat * torch.sin(hue)], dim=-3)

        # Combine into a complex tensor
        comp_tensor = torch.complex(real_part, imag_part)

        assert comp_tensor.dtype == torch.complex64
        return comp_tensor.contiguous() # Ensure contiguous memory layout

    def __repr__(self):
        return self.__class__.__name__ + '()'
    

class FromComplex(object):
    def __call__(self, img, eps=0.01):
        real = img.real
        imaginary = img.imag
        
        hue_real = real[..., 0, :,:,:]
        sat_real = real[..., 1, :,:,:]
        val_real = real[..., 2, :,:,:]

        hue_imag = imaginary[..., 0, :,:,:]
        sat_imag = imaginary[..., 0, :,:,:]
        val_imag = imaginary[..., 0, :,:,:]

        hue = sat_real/(torch.abs(val_real+val_imag)+eps)
        sat = torch.abs(val_real+val_imag)+eps
        val = hue_real

        hue = hue%(2*torch.pi) # to restrict hue between [0,2*pi]


        hsv_img = torch.stack([hue, sat, val], dim=-3)
        return hsv_img


    def __repr__(self):
        return self.__class__.__name__+'()'
    

class ToComplex2(object):
    def __call__(self, img):
        hue = img[..., 0, :, :]
        sat = img[..., 1, :, :]
        val = img[..., 2, :, :]

        real1 = val
        real2 = torch.cos(hue)

        imag1 = sat
        imag2 = torch.sin(hue)

        real = torch.stack([real1, real2], dim=-3)
        imag = torch.stack([imag1, imag2], dim=-3)

        return torch.complex(real, imag)