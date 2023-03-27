# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import sclip


def get_clip(name, device="cpu", training=False):
    model, _preprocess = sclip.load(name, device=device)

    for p in model.parameters():
        p.requires_grad_(training)
    model.train(training)

    return model


def get_clip_imgenc(name, device="cpu", training=False):
    clipmodel = get_clip(name, device, training)
    return clipmodel.visual


def get_clip_textenc(name, device="cpu", training=False):
    clipmodel = get_clip(name, device, training)
    # delete unused parameters
    del clipmodel.visual
    del clipmodel.logit_scale
    clipmodel.forward = clipmodel.encode_text

    return clipmodel
