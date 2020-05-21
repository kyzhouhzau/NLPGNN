import tensorflow as tf


class CheckpointSaver:
    def __init__(self, model):
        self.checkpoint = tf.train.Checkpoint(model=model)
        self.old_metric_score = 0

    def save(self, filepath,
             metric_score,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None):
        if self.old_metric_score < metric_score:
            path = self.checkpoint.save(filepath, overwrite, include_optimizer,
                                        save_format, signatures, options)
            self.old_metric_score = metric_score
            print("Saving Model in {}".format(path))
