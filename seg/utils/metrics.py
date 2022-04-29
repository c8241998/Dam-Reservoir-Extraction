"""
The implementation of some metrics based on Tensorflow.

"""
import tensorflow as tf
from skimage import measure

class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)

class BiIOU(tf.keras.metrics.Metric):

    def __init__(self, name='biou' ,**kwargs):
        super(BiIOU, self).__init__(name=name, **kwargs)
        self.value = self.add_weight(name='value', initializer='zeros')
        self.sum = self.add_weight(name='sum', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        batch = y_true.shape[0]
        self.sum.assign_add(batch)
        
        y_true = tf.reshape(y_true,[batch,-1])
        y_pred = tf.reshape(y_pred,[batch,-1])

        a = tf.cast((y_true+y_pred)==2,tf.float32)
        a = tf.reduce_sum(a,axis=-1)

        b = tf.cast((y_true+y_pred)>0,tf.float32)
        b = tf.reduce_sum(b,axis=-1)

        c = tf.math.divide(a,b+0.00000001)
        c = tf.reduce_sum(c)
        self.value.assign_add(c)
    
    def result(self):
        return self.value / self.sum
 

