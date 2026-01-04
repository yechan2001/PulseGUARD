import tensorflow as tf
import datetime
import os
import numpy as np
from model import gen_loss, disc_loss, gen_opt, disc_opt
from metrics import compute_metrics

@tf.function
def train_step(ppg, ecg, generator, discriminator):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_ecg = generator(ppg, training=True)
        real_output = discriminator(ecg, training=True)
        fake_output = discriminator(fake_ecg, training=True)
        g_loss = gen_loss(fake_output, ecg, fake_ecg)
        d_loss = disc_loss(real_output, fake_output)
    gradients_g = gen_tape.gradient(g_loss, generator.trainable_variables)
    gradients_d = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients_g, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_d, discriminator.trainable_variables))
    return g_loss, d_loss

def train(dataset,test_dataset, epochs, generator, discriminator, mse_filepath, ppc_filepath):
    # Initialize TensorBoard summary writer
    log_dir = os.path.join("/root/tmp/PulseGUARD/logs", "gan", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.create_file_writer(log_dir)

    best_mse = float('inf')
    best_pcc = float('-inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for ppg_batch, ecg_batch in dataset:
            g_loss, d_loss = train_step(ppg_batch, ecg_batch, generator, discriminator)
        test_ppg, test_ecg = next(iter(test_dataset))
        pred_ecg = generator(test_ppg, training=False).numpy()
        #print(f"test_ecg shape: {test_ecg.shape}, pred_ecg shape: {pred_ecg.shape}")

        # Compute metrics for all samples in the batch and average
        mse_list, pcc_list, dtw_list = [], [], []
        for i in range(test_ecg.shape[0]):  # Loop over batch
            test_ecg_np = test_ecg[i]  # (375, 1)
            pred_ecg_np = pred_ecg[i]  # (375, 1)
            mse, pcc, dtw_dist = compute_metrics(test_ecg_np, pred_ecg_np)
            mse_list.append(mse)
            pcc_list.append(pcc)
            dtw_list.append(dtw_dist)

        mse = np.mean(mse_list)
        pcc = np.mean(pcc_list)
        dtw_dist = np.mean(dtw_list)

        with summary_writer.as_default():
            tf.summary.scalar('Generator Loss', g_loss, step=epoch)
            tf.summary.scalar('Discriminator Loss', d_loss, step=epoch)
            tf.summary.scalar('MSE', mse, step=epoch)
            tf.summary.scalar('PCC', pcc, step=epoch)
            tf.summary.scalar('DTW Distance', dtw_dist, step=epoch)
        print(f"Gen Loss: {g_loss:.4f}, Disc Loss: {d_loss:.4f}, MSE: {mse:.4f}, PCC: {pcc:.4f}, DTW: {dtw_dist:.2f}")
        if mse < best_mse:
            best_mse = mse
            generator.save(mse_filepath)
            print("✓ Saved best generator (MSE ↓)")
        if pcc > best_pcc:
            best_pcc = pcc
            generator.save(ppc_filepath)
            print("✓ Saved best generator (PCC ↑)")

    print("Training complete. Use `tensorboard --logdir logs/gan/` to view metrics.")