from text import symbols
from easydict import EasyDict


def create_hparams():
    hparams = EasyDict(
        ################################
        #      Experiment Parameters   #
        ################################
        epochs=500,
        iters_per_checkpoint=10000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=["embedding.weight"],
        ################################
        #      Data Parameters         #
        ################################
        load_mel_from_disk=False,
        training_files="Data/demo/filelists/train.list",
        validation_files="Data/demo/filelists/val.list",
        text_cleaners=["basic_cleaners"],
        num_emotions=5,
        num_speakers=10,
        ################################
        #      Audio Parameters        #
        ################################
        # max_wav_value=32768.0,
        sampling_rate=16000,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        ################################
        #      Model Parameters        #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        speaker_embedding_dim=32,
        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,
        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        #################################
        # Optimization Hyper-parameters #
        #################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=1,
        mask_padding=True,  # set model's padded outputs to padded values
        ###########################################
        # EDM Module Parameters #
        ###########################################
        reverse_lambda_val=1,
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,
        # Inference parameters
        back_bone_ckpt="",
        vocoder_ckpt="",
    )

    return hparams
