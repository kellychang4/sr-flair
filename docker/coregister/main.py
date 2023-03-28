import ants, argparse, glob, re
import numpy as np
import pandas as pd
import os.path as op
import skimage as ski
from scipy import ndimage as ndi


def read_mni_images(sidedness):
    ### base mni image file name
    paths_mni = op.join("/home", "MNI152NLin2009cAsym")
    mni_fname = "MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz"

    ### whole and sidedness mni brain mask file names
    mni_whole = "MNI152NLin2009cAsym_res-01_desc-nomidbrain_mask.nii.gz"
    mni_mask = mni_whole if sidedness == "whole" else \
        re.sub("(_mask)", "_hemi-%s\\1" % sidedness[0].upper(), mni_whole)

    return {
        "image": ants.image_read(op.join(paths_mni, mni_fname)), 
         "mask": ants.image_read(op.join(paths_mni,  mni_mask))
    }


def read_subject_images():
    ### subject data directory
    paths_data = op.join("/home", "data")

    ### subject file names
    t2_fname     = glob.glob(op.join(paths_data, "*C_T2_av_stripped.nii.gz"))[0]
    flairc_fname = glob.glob(op.join(paths_data, "*C_FLAIR_stripped.nii.gz"))[0]
    flairx_fname = glob.glob(op.join(paths_data, "*X_FLAIR_stripped.nii.gz"))[0]

    ### load subject images
    t2     = ants.image_read(t2_fname)
    flairc = ants.image_read(flairc_fname)
    flairx = ants.image_read(flairx_fname)

    return {
        "t2"    : { "fname": op.basename(    t2_fname), "image":     t2 },
        "flairc": { "fname": op.basename(flairc_fname), "image": flairc }, 
        "flairx": { "fname": op.basename(flairx_fname), "image": flairx }
    }


def coregister_mnimask_to_subject(fixed, moving, mask):
    mtx  = ants.registration(fixed, moving, type_of_transform = "SyN")
    mask = ants.apply_transforms(fixed, mask, transformlist = mtx["fwdtransforms"])
    return ants.threshold_image(mask, 0.01) 


def clean_flairx_image(flairx_image):

    ### define dimension indexing function
    def index_brain(brain, index, dimension):
        if dimension == 0:
            return brain[index,:,:]
        elif dimension == 1:
            return brain[:,index,:]
        elif dimension == 2:
            return brain[:,:,index]

    ### define dimension assigning function
    def assign_brain(brain, image, index, dimension):
        if dimension == 0:
            brain[index,:,:] = image
        elif dimension == 1:
            brain[:,index,:] = image
        elif dimension == 2:
            brain[:,:,index] = image
        return brain

    ### identify the coronal dimension (AP/PA)
    coronal_dim = np.max([flairx_image.orientation.find("A"), 
                            flairx_image.orientation.find("P")])

    ### initialize numpy and empty mask variables
    flairx_np   = flairx_image.numpy()
    flairx_mask = np.zeros(flairx_np.shape)

    ### for each slice in the coronal orientation
    for i in range(flairx_np.shape[coronal_dim]):
        curr_image = index_brain(flairx_np, i, coronal_dim)

        if curr_image.sum() > 0.0: # if there is brain
            ### rescale image to [-1, 1] for conversion to uint 
            img = (((curr_image - curr_image.min()) / (curr_image.max() - curr_image.min())) * 2) - 1
            img = ski.util.img_as_ubyte(img)

            ### rank filter image based on median value
            mask = ski.filters.rank.median(img, ski.morphology.disk(1))

            ### denoise image (outside brain)
            mask = ski.restoration.denoise_nl_means(mask)

            ### threshold image based on yen thresholding
            mask = mask > ski.filters.threshold_yen(mask)

            ### fill thresholded image of "holes"
            mask = ndi.binary_fill_holes(mask)

            ### create convex hull around thresholded image
            mask = ski.morphology.convex_hull_image(mask)        

            ### save mask to new volume
            flairx_mask = assign_brain(flairx_mask, mask, i, coronal_dim)

    ### add gaussian blur to mask to dialate mask coverage
    flairx_mask = ski.filters.gaussian(flairx_mask, 5) > 0.2

    ### convert numpy mask to ants object
    flairx_mask = ants.from_numpy(
        data      = flairx_mask * 1.0,
        origin    = flairx_image.origin, 
        spacing   = flairx_image.spacing,
        direction = flairx_image.direction
    )

    return ( ants.mask_image(flairx_image, flairx_mask), flairx_mask )


def write_image(image, image_name, suffix = None):
    suffix = image_name if not suffix else suffix
    save_name = re.sub("(.nii.gz)$", "_%s\\1" % suffix, image["fname"])
    ants.image_write(image[image_name], op.join("/home/data", save_name))


def create_image_masks(images, mni_images, sidedness):
    ### extract MNI images as variables
    mni_image = mni_images["image"].copy()
    mni_mask  = mni_images["mask"].copy()

    ### create t2 mask (always whole brain)
    images["t2"]["mask"] = coregister_mnimask_to_subject(
        images["t2"]["image"], mni_image, mni_mask)

    ### create flairc mask (always whole brain)
    images["flairc"]["mask"] = coregister_mnimask_to_subject(
        images["flairc"]["image"], mni_image, mni_mask)
    
    ### create flairx mask (whole or hemi brains)
    flairx_image = images["flairx"]["image"].copy()
    if sidedness != "whole": # if half brain, clean image first
        flairx_image, clean_mask = clean_flairx_image(flairx_image)
        images["flairx"].update({ "mask": clean_mask })
        write_image(images["flairx"], "mask", "cleanmask")
        mni_image = ants.mask_image(mni_image, mni_mask)
    images["flairx"]["mask"] = coregister_mnimask_to_subject(
        flairx_image, mni_image, mni_mask)

    return images


def coregister_images(images):
    ### extract t2, flairc, and flairx copied images
    t2, flairc, flairx = [images[k].copy() for k in ("t2", "flairc", "flairx")]

    ### mask all subject images
    masked_t2     = ants.mask_image(    t2["image"],     t2["mask"])
    masked_flairc = ants.mask_image(flairc["image"], flairc["mask"])
    masked_flairx = ants.mask_image(flairx["image"], flairx["mask"])

    ### estimate coregistration matrix (flairc --> t2)
    flairc_to_t2_mtx = ants.registration(masked_t2, masked_flairc, type_of_transform = "Rigid")

    ### estimate coregistration matrix (t2 --> flairx)
    t2_to_flairx_mtx = ants.registration(masked_flairx, masked_t2, type_of_transform = "Affine")

    ### apply coregistration matrix to t2 image and mask
    coreg_t2      = ants.apply_transforms(masked_flairx, t2["image"], transformlist = t2_to_flairx_mtx["fwdtransforms"])
    coreg_t2_mask = ants.apply_transforms(masked_flairx,  t2["mask"], transformlist = t2_to_flairx_mtx["fwdtransforms"])

    ### apply coregistration matrices to flairc image and mask
    mtx_list = t2_to_flairx_mtx["fwdtransforms"] + flairc_to_t2_mtx["fwdtransforms"]
    coreg_flairc      = ants.apply_transforms(masked_flairx, flairc["image"], transformlist = mtx_list)
    coreg_flairc_mask = ants.apply_transforms(masked_flairx,  flairc["mask"], transformlist = mtx_list)

    ### save coregistered images to images dictionary
    images["t2"].update({ "image": coreg_t2, "mask": coreg_t2_mask })
    images["flairc"].update({ "image": coreg_flairc, "mask": coreg_flairc_mask })
        
    return images


def write_coregistered_images(images):
    for image_type in ["t2", "flairc", "flairx"]: # for each image type
        curr_image = images[image_type].copy() # current image type
        write_image(curr_image, "image", "coreg")  # save coreg image
        write_image(curr_image, "mask") # save coreg mask


def main(subject_id):
    ### read in subject sidedess information 
    df = pd.read_csv("subject-sidedness_latest.csv")
    sidedness = df.sidedness[df.subject_id == int(subject_id)].iloc[0].lower()

    ### read in MNI template and mask images
    mni_images = read_mni_images(sidedness)

    ### read subject images (FLAIRc, T2, FLAIRx)
    images = read_subject_images()

    ### create image masks for all images
    images = create_image_masks(images, mni_images, sidedness)

    ### coregister images with each other
    images = coregister_images(images)

    ### write coregistered images and masks
    write_coregistered_images(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_id", type = str)
    args = parser.parse_args()
    main(args.subject_id)