#insert line 675
for S in [sublist[0] for sublist in MainSub]:
        # Construct the model path
    model_path = f"dataset/model_{filename}_S{S}.p"

    # Check if the model file exists before loading
    if os.path.exists(model_path):
        try:
            model = pickle.load(open(model_path, "rb"))
        except Exception as e:
            continue
    else:
        model_path = 'dataset/model_default.p'
        model = pickle.load(open(model_path, "rb"))
        
    globals()[f'model{S}'] = model