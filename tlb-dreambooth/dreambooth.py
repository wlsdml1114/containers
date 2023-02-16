'''
Trains DreamBooth image encoder then text encoder sequentially.
'''

import subprocess


# ---------------------------------------------------------------------------- #
#                                 Text Encoder                                 #
# ---------------------------------------------------------------------------- #
def dump_only_textenc(trnonltxt, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, training_steps):
    '''
    Train only the text encoder.
    '''
    text_encoder = subprocess.Popen([
        "accelerate", "launch", "/src/diffusers/examples/dreambooth/train_dreambooth.py",

        f"--pretrained_model_name_or_path={MODELT_NAME}",
        f"--instance_data_dir={INSTANCE_DIR}",
        f"--instance_prompt={PT}",
        f"--output_dir={OUTPUT_DIR}",
        f"--seed={Seed}",
        "--resolution=512",
        "--train_text_encoder",
        "--train_batch_size=1",
        f"--max_train_steps={training_steps}",
        "--gradient_accumulation_steps=1",
        "--gradient_checkpointing",  # ENABLED FOR TESTING
        "--learning_rate=1e-6",
        "--lr_scheduler=linear",
        "--lr_warmup_steps=0",
        f"--mixed_precision={precision}",

        "--image_captions_filename",
        "--dump_only_text_encoder",

        # trnonltxt,  # train_only_text_encoder
        # extrnlcptn,  # external_captions
    ])

    text_encoder.wait()


# ---------------------------------------------------------------------------- #
#                                     UNet                                     #
# ---------------------------------------------------------------------------- #
def train_only_unet(stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, Res, precision, num_train_epochs):
    '''
    Train only the image encoder.
    '''
    unet = subprocess.Popen([
        "accelerate", "launch", "/src/diffusers/examples/dreambooth/train_dreambooth.py",
        "--train_only_unet",
        f"--stop_text_encoder_training={stpsv}",
        f"--save_n_steps={stp}",

        f"--pretrained_model_name_or_path={MODELT_NAME}",
        f"--instance_data_dir={INSTANCE_DIR}",
        f"--output_dir={OUTPUT_DIR}",
        # f"--captions_dir=\'{CAPTIONS_DIR}\'",
        f"--instance_prompt={PT}",
        f"--seed={Seed}",
        f"--resolution={Res}",
        f"--mixed_precision={precision}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--learning_rate=2e-6",
        "--lr_scheduler=linear",
        "--lr_warmup_steps=0",

        f"--num_train_epochs={num_train_epochs}",

        f"--Session_dir={SESSION_DIR}",
    ])

    unet.wait()
