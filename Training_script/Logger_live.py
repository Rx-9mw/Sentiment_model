from tensorflow.keras.callbacks import Callback

class EpochLogger(Callback):
    def __init__(self, root, label_epoch, label_loss, label_val_loss, label_loss_difference, label_val_acc):
        super().__init__()
        self.root = root
        self.label_epoch = label_epoch
        self.label_loss = label_loss
        self.label_val_loss = label_val_loss
        self.label_loss_difference = label_loss_difference
        self.label_val_acc = label_val_acc
        self.loss = None
        self.loss_before = None
        self.epoch_count = 0
    
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count = epoch + 1
        self.root.after(0, lambda: self.label_epoch.config(text=f"Working on Epoch {epoch + 1}..."))
        
        
    def on_train_end(self, logs=None):
        self.root.after(0, lambda: self.label_epoch.config(text=f"Finished all {self.epoch_count} Epochs."))
        
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        def progress():
            if epoch > 0:
                if self.loss is not None: 
                    if current_loss < self.loss:
                        return "green", str(round(current_loss - self.loss, 4))
                    else:
                        return "red", "+" + str(round(current_loss - self.loss, 4))
                    
            return "blue", "No difference"
        
        color, difference = progress()
        last_loss = self.loss
        self.loss = current_loss
        val_loss = logs.get("val_loss")
        val_accuracy = logs.get("val_accuracy")
        if last_loss is not None:
            self.root.after(0, lambda: self.label_loss.config(text=f"Loss Before: {last_loss:.4f} vs After: {current_loss:.4f}"))
        else:
            self.root.after(0, lambda: self.label_loss.config(text=f"Loss Before: - vs After: {current_loss:.4f}"))
            
        self.root.after(0, lambda: self.label_loss_difference.config(text=f"Difference: {difference}", fg=color))
        self.root.after(0, lambda: self.label_val_loss.config(text=f"Current Validation Loss: {val_loss:.4f}"))
        self.root.after(0, lambda: self.label_val_acc.config(text=f"Current Validation Accuracy: {val_accuracy:.4f}"))
      
