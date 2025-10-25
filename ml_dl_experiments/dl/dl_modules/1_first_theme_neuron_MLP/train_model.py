from metrics import mse

def train_model(X_train,
                y_train, 
                X_val, 
                y_val, 
                model, 
                epochs=500) -> tuple:
    train_losses, val_losses = [], []
    for _ in range(epochs):
        # train
        preds = model.forward(X_train)
        train_losses.append(model.mse(preds, y_train))
        model.backward(y_train)
        model.update_params()
        # val
        preds_val = model.forward(X_val)
        val_losses.append(mse(preds_val, y_val))
    return train_losses, val_losses