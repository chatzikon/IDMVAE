import os
import numpy as np
import glob

import torch

from fid.inception import InceptionV3
from fid.fid_score import get_activations
from fid.fid_score import calculate_frechet_distance
from eval_functions_CUB import calculate_fid, calculate_fid_dict
import argparse

def calculate_inception_features_for_gen_evaluation(inception_state_dict_path, device, orig_path, gen_path, dims=2048, batch_size=256):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], path_state_dict=inception_state_dict_path)
    model = model.to(device)
    if not os.path.exists(gen_path):
        raise RuntimeError('Invalid path: %s' % gen_path)

    filename_act_real_calc = os.path.join(gen_path, 'real_activations.npy')
    if not os.path.exists(filename_act_real_calc):
        files_real_calc = glob.glob(os.path.join(orig_path, '*' + '.png'))
        act_real_calc = get_activations(files_real_calc, model, device, batch_size, dims, verbose=False)
        np.save(filename_act_real_calc, act_real_calc)

        files_gen = glob.glob(os.path.join(gen_path, '*' + '.png'))
        filename_act = os.path.join(gen_path, 'gen_activations.npy')
        act_rand_gen = get_activations(files_gen, model, device, batch_size, dims, verbose=False)
        np.save(filename_act, act_rand_gen)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    calculate_inception_features_for_gen_evaluation(args.inception_path, device, args.orig_path, args.sample_path)

    # FID calculation
    file_activations_real = os.path.join(args.sample_path, 'real_activations.npy')
    feats_real = np.load(file_activations_real)
    file_activations_randgen = os.path.join(args.sample_path, 'gen_activations.npy')
    feats_randgen = np.load(file_activations_randgen)
    fid_score = calculate_fid(feats_real, feats_randgen)
    print(f"fid_score={fid_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-path", type=str, required=True)
    parser.add_argument("--sample-path", type=str, required=True)
    parser.add_argument('--inception_path', type=str,
                        default='/data/backed_up/shared/Data/CUB/CUBcluster8_256/pt_inception-2015-12-05-6726825d.pth',
                        help='Path to inception module for FID calculation')
    args = parser.parse_args()
    main(args)

"""

Command history and FID results of CelebAMask-HQ:

DiT CKPT ID: 0035000
# IDMVAE
# attr2img
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_orig \
  --sample-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_attr2img_qzpw
==> fid_score=23.238413576293112
  
# mask2img
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_orig \
  --sample-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_mask2img_qzpw
==> fid_score=22.461521490228932
  
# img2img_qzpw
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_orig \
  --sample-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_img2img_qzpw
==> fid_score=17.88585828059513
  
# img2img_qwpz
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_orig \
  --sample-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_img2img_qwpz
==> fid_score=33.747500897827194
  
# img_random
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_orig \
  --sample-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_img_random
==> fid_score=36.77612422595041


# MMVAEplus
# attr2img
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_orig \
  --sample-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/MMVAEplus_11_29_0_ep100_test/images_attr2img_qzpw
==> fid_score=22.98219831298556
  
# mask2img
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_orig \
  --sample-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/MMVAEplus_11_29_0_ep100_test/images_mask2img_qzpw
==> fid_score=20.00759417335408

# img2img_qzpw
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_orig \
  --sample-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/MMVAEplus_11_29_0_ep100_test/images_img2img_qzpw
==> fid_score=15.978268614490048

# img2img_qwpz
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_orig \
  --sample-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/MMVAEplus_11_29_0_ep100_test/images_img2img_qwpz
==> fid_score=38.586711175645746

# img_random
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/IDMVAE_11_30_5_ep100_test/images_orig \
  --sample-path /data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32/_denoiser/_img_denoiseds/MMVAEplus_11_29_0_ep100_test/images_img_random
==> fid_score=39.57035065298902




Command history and FID results of CUB-HQ:
python compute_fid_individual.py --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ --sample-path IDMVAE_Aug10_Cross40_11_15_53_ep50/

4x32x32_10x:
_text2img_qzpw
cd mmvaeplus/src
DMVAE:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/DMVAE_11_19_14_ep50_test/images_text2img_qzpw
-> fid_score=104.16706488375678


MMVAE+:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/MMVAEplus_11_15_56_ep50_test/images_text2img_qzpw
-> fid_score=70.15713176795737

AugMI:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_only_Aug10_11_19_62_ep50_test/images_text2img_qzpw
-> fid_score=72.1658167118766

CrossMI:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_only_Cross40_11_15_55_ep50_test/images_text2img_qzpw
-> fid_score=66.29060898865146

IDMVAE:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_Aug10_Cross40_11_15_53_ep50_test/images_text2img_qzpw
-> fid_score=64.43514335038321

Diffusion:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_Diffdot1_Aug10_Cross40_11_17_60_ep50_test/images_text2img_qzpw
-> fid_score=60.548675065027595


_img2img_qzpw
DMVAE:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/DMVAE_11_19_14_ep50_test/images_img2img_qzpw
-> fid_score=70.53440085292357


MMVAE+:
cd mmvaeplus/src
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/MMVAEplus_11_15_56_ep50_test/images_img2img_qzpw
-> fid_score=62.52823786625814

AugMI:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_only_Aug10_11_19_62_ep50_test/images_img2img_qzpw
-> fid_score=62.93761446015748

CrossMI:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_only_Cross40_11_15_55_ep50_test/images_img2img_qzpw
-> fid_score=69.98818016920501

IDMVAE:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_Aug10_Cross40_11_15_53_ep50_test/images_img2img_qzpw
-> fid_score=58.06535448764336

Diffusion:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_Diffdot1_Aug10_Cross40_11_17_60_ep50_test/images_img2img_qzpw
-> fid_score=59.699879798417385



_img2img_qwpz
DMVAE:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/DMVAE_11_19_14_ep50_test/images_img2img_qwpz
-> fid_score=71.50874635367433

MMVAE+:
cd mmvaeplus/src
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/MMVAEplus_11_15_56_ep50_test/images_img2img_qwpz
-> fid_score=67.60657031303819

AugMI:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_only_Aug10_11_19_62_ep50_test/images_img2img_qwpz
-> fid_score=70.37234811581274

CrossMI:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_only_Cross40_11_15_55_ep50_test/images_img2img_qwpz
-> fid_score==61.41690900627316

IDMVAE:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_Aug10_Cross40_11_15_53_ep50_test/images_img2img_qwpz
-> fid_score=76.35211785032769

Diffusion:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images \
  --sample-path /data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32_10x/denoiser/_img_denoiseds/IDMVAE_Diffdot1_Aug10_Cross40_11_17_60_ep50_test/images_img2img_qwpz
-> fid_score=60.44653479966246














######################

4x32x32_1x:
_text2img_qzpw
MMVAE+:
cd mmvaeplus/src
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_text2img_qzpw/MMVAEplus_11_15_56_ep50_test/
-> fid_score=70.12707786863547

CrossMI:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_text2img_qzpw/IDMVAE_Cross40_11_15_55_ep50_test/
-> fid_score=65.75595111184614

IDMVAE:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_text2img_qzpw/IDMVAE_Aug10_Cross40_11_15_53_ep50_test/
-> fid_score=62.76828701230144

Diffusion:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_text2img_qzpw/IDMVAE_Diffdot1_Aug10_Cross40_11_17_60_ep50_test/
-> fid_score=60.208383043240616


_img2img_qzpw
MMVAE+:
cd mmvaeplus/src
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_img2img_qzpw/MMVAEplus_11_15_56_ep50_test/
-> fid_score=59.67077204146412

CrossMI:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_img2img_qzpw/IDMVAE_Cross40_11_15_55_ep50_test/
-> fid_score=69.98427704306584

IDMVAE:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_img2img_qzpw/IDMVAE_Aug10_Cross40_11_15_53_ep50_test/
-> fid_score=60.289034851664695

Diffusion:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_img2img_qzpw/IDMVAE_Diffdot1_Aug10_Cross40_11_17_60_ep50_test/
-> fid_score=61.26701850964679



_img2img_qwpz
MMVAE+:
cd mmvaeplus/src
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_img2img_qwpz/MMVAEplus_11_15_56_ep50_test/
-> fid_score=66.40064273089638

CrossMI:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_img2img_qwpz/IDMVAE_Cross40_11_15_55_ep50_test/
-> fid_score=61.69107605807301

IDMVAE:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_img2img_qwpz/IDMVAE_Aug10_Cross40_11_15_53_ep50_test/
-> fid_score=78.84570025258822

Diffusion:
python compute_fid_individual.py \
  --orig-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/ \
  --sample-path /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/_img2img_qwpz/IDMVAE_Diffdot1_Aug10_Cross40_11_17_60_ep50_test/
-> fid_score=60.77934308615096

"""
