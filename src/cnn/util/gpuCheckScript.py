import tensorflow as tf

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")
    if gpus:
        print("TensorFlow GPU check PASSED")
        try:
            # Print details for each GPU
            for gpu in gpus:
                print(f"  Name: {gpu.name}")
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set successfully.")
        except RuntimeError as e:
            # Memory growth must be set at the start of the program
            print(f"RuntimeError setting memory growth: {e}")
        return True
    else:
        print("TensorFlow GPU check FAILED")
        return False

if __name__ == "__main__":
    check_gpu()

