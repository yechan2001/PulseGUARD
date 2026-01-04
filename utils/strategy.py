import tensorflow as tf
from tensorflow.keras import layers

def select_final_model(metrics_dict, safety_threshold=0.85):
    """
    metrics_dict:  {'pcc': ..., 'rpa': ..., 'mse': ..., 'epoch': ...}
    """
    print(f"\n--- Model Selection Strategy (Safety Threshold: RPA >= {safety_threshold}) ---")

    def print_model_info(name, m):
        print(f" Selected: {name}")
        print(f"   Details: Epoch {m['epoch']} | PCC={m['pcc']:.4f} | RPA={m['rpa']:.4f} | MSE={m['mse']:.6f}")

    # 1.  Best PCC 
    pcc_model = metrics_dict.get('best_pcc_model')
    if pcc_model and pcc_model['rpa'] >= safety_threshold:
        print_model_info("Best PCC Model", pcc_model)
        return 'best_pcc_model'

    # 2.  Best Overall 
    overall_model = metrics_dict.get('best_overall_model')
    if overall_model:
        if overall_model['rpa'] >= safety_threshold:
            print_model_info("Best Overall Model", overall_model)
            print("   Reason: Best PCC model failed safety check (RPA < threshold).")
            return 'best_overall_model'
        else:
             print(f"   Skipped Overall Model (RPA={overall_model['rpa']:.4f} < {safety_threshold})")

    # 3. Best RPA
    rpa_model = metrics_dict.get('best_rpa_model')
    print(f" Selected: Best RPA Model")
    if rpa_model:
        print(f"   Details: Epoch {rpa_model['epoch']} | PCC={rpa_model['pcc']:.4f} | RPA={rpa_model['rpa']:.4f} | MSE={rpa_model['mse']:.6f}")
    print("   Reason: Both PCC and Overall models failed safety check. Falling back to safest model.")
    return 'best_rpa_model'


def freeze_for_personalization(generator):
    print(">>> Set up personalized fine-tuning strategy: Freeze Encoder & LSTM, activate Decoder")
    
    for layer in generator.layers:
        layer.trainable = False
        
        # 1. Identify the convolutional layers of the Decoder and unfreeze them (Decoder k=3, Output k=1)
        if isinstance(layer, tf.keras.layers.Conv1D):
            if layer.kernel_size == (3,) or layer.kernel_size == (1,):
                layer.trainable = True
                print(f"  [Active] Decoder Layer: {layer.name} (k={layer.kernel_size})")
            else:
                print(f"  [Frozen] Encoder Layer: {layer.name} (k={layer.kernel_size})")
        
        # 2. The LSTM layer must be frozen
        elif isinstance(layer, tf.keras.layers.Bidirectional):
             print(f"  [Frozen] LSTM Bottleneck: {layer.name}")
             
    return generator