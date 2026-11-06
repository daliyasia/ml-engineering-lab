from lightning.pytorch.callbacks.callback import Callback


class MLFlowResultsCallback(Callback):
    def __init__(self):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        """results_dir = os.path.join(trainer.default_root_dir, "results")
        output_train = pl_module.train_confusion_country.compute()
        output_val = pl_module.val_confusion_country.compute()
        output_test = pl_module.test_confusion_country.compute()

        #matrix_train = output_train.cpu().numpy()
        matrix_val = output_val.cpu().numpy()
        matrix_test = output_test.cpu().numpy()
        matrices = [matrix_train, matrix_val, matrix_test]
        names = ['Train', 'Val', 'Test']
        for name, matrix in zip(names, matrices):
            fig = plt.figure()
            plt.matshow(matrix)
            plt.title(f'{name} Confusion Matrix LPR')
            plt.colorbar()
            plt.ylabel('True Label')
            plt.xlabel('Predicated Label')
            plt.savefig(os.path.join(results_dir, f'confusion_matrix_{name}.png'))
            plt.close()"""
        pass

    def on_validation_end(self, trainer, pl_module):
        """results_dir = os.path.join(trainer.default_root_dir, "results")
        for batch in pl_module.predictions:
            #mlflow.log_artifact("/tmp/clippy.png", "img")
            img = images[num_img].cpu().numpy().transpose(1,2,0)

            img += [104, 117, 123]
            img = img.astype(np.uint8)"""
        pass
