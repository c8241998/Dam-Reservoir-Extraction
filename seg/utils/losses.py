"""
The implementation of some losses based on Tensorflow.

"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import losses

backend = tf.keras.backend


def categorical_crossentropy_with_logits(y_true, y_pred):
    # compute cross entropy
    cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

    # compute loss
    loss = backend.mean(backend.sum(cross_entropy, axis=[1, 2]))
    return loss


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        
        y_pred = backend.softmax(y_pred)
        # compute ce loss
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=False)
        # compute weights
        weights = backend.sum(alpha * backend.pow(1 - y_pred, gamma) * y_true, axis=-1)
        
        
        return backend.mean(backend.sum(weights * cross_entropy, axis=[1, 2]))
    
    
    return loss


def miou_loss(weights=None, num_classes=2):
    if weights is not None:
        assert len(weights) == num_classes
        weights = tf.convert_to_tensor(weights)
    else:
        weights = tf.convert_to_tensor([1.] * num_classes)

    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)

        inter = y_pred * y_true
        inter = backend.sum(inter, axis=[1, 2])

        union = y_pred + y_true - (y_pred * y_true)
        union = backend.sum(union, axis=[1, 2])

        return -backend.mean((weights * inter) / (weights * union + 1e-8))

    return loss


def self_balanced_focal_loss(alpha=3, gamma=2.0):
    """
    Original by Yang Lu:

    This is an improvement of Focal Loss, which has solved the problem
    that the factor in Focal Loss failed in semantic segmentation.
    It can adaptively adjust the weights of different classes in semantic segmentation
    without introducing extra supervised information.

    :param alpha: The factor to balance different classes in semantic segmentation.
    :param gamma: The factor to balance different samples in semantic segmentation.
    :return:
    """

    def loss(y_true, y_pred):
        # cross entropy loss
        y_pred = backend.softmax(y_pred, -1)
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred)

        # sample weights
        sample_weights = backend.max(backend.pow(1.0 - y_pred, gamma) * y_true, axis=-1)

        # class weights
        pixel_rate = backend.sum(y_true, axis=[1, 2], keepdims=True) / backend.sum(backend.ones_like(y_true),
                                                                                   axis=[1, 2], keepdims=True)
        class_weights = backend.max(backend.pow(backend.ones_like(y_true) * alpha, pixel_rate) * y_true, axis=-1)

        # final loss
        final_loss = class_weights * sample_weights * cross_entropy
        return backend.mean(backend.sum(final_loss, axis=[1, 2]))

    return loss


class metric_loss(losses.Loss):
    
    def __init__(self,stage,N,margin):
        
        super(metric_loss, self).__init__()
        self.shape = {
            'c1': 64,'c2': 32,'c3': 16,'c4': 16,'c5': 16
        }
        self.channel = {
            'c1': 128,'c2': 256,'c3': 728,'c4': 728,'c5': 2048
        }
        self.N = N
        self.resize_channel = self.channel[stage]
        self.Distance = lambda x,y : tf.reduce_sum(abs(x-y),axis=-1)#lambda x,y : abs(x[1]-y[1]) + abs (x[2]-y[2])
        self.L2 = lambda x,y : tf.sqrt(tf.reduce_sum(tf.square(x-y),axis=-1))
        self.margin = margin
        self.resize = lambda x: tf.squeeze( tf.image.resize(tf.expand_dims(x,-1),[16,16],method='nearest'), -1 )
    
    def call(self, y_true, y_pred):
        
        batch = y_pred.shape[0]
        embedding = tf.reshape(y_pred[:,:16*16*self.resize_channel],[-1,16,16,self.resize_channel])  # none, 16, 16, 2048
        y_pred = tf.reshape(y_pred[:,16*16*self.resize_channel:],[-1,256,256,2])     # none, 256, 256, 2
        
        
        true = tf.cast( tf.argmax(y_true, -1), tf.int64)
        true = self.resize(true) # none, 16, 16
        pred = tf.cast( tf.argmax(y_pred, -1), tf.int64)
        pred = self.resize(pred) # none, 16, 16
        
        
        anchor_id = tf.where((pred+true)==2)
        positive_id = tf.where((true-pred)==1)
        negative_id = tf.where((true-pred)==-1)
        
#         print('anchor shape: ',anchor_id.shape,'  positive shape: ',positive_id.shape,'  negative shape: ',negative_id.shape)
#         print(gt_water_id)
        
        anchors = tf.gather_nd(embedding,anchor_id)
        positives = tf.gather_nd(embedding,positive_id)
        negatives = tf.gather_nd(embedding,negative_id)

        triplets = []
        
        for img_id in range(batch):
            
#             randomly choose N anchors for each image
            img_anchor_id = tf.where(anchor_id[:,0]==img_id)
            N = min(self.N,len(img_anchor_id))
            img_anchor_id = tf.gather(img_anchor_id, tf.random.shuffle(range(N)))[:N]
            img_anchors = tf.gather_nd(anchors,img_anchor_id)
            
            
            for a in img_anchors:
#                 print(a.shape,positives.shape)
                if len(positive_id)==0 or len(negative_id)==0:
                    continue
                
                N = len(positive_id)
                random_positive_id = tf.gather(range(N), tf.random.shuffle(range(N)))[0]
                p = positives[random_positive_id]
                
                N = len(negative_id)
                random_negative_id = tf.gather(range(N), tf.random.shuffle(range(N)))[0]
                n =  negatives[random_negative_id]
                
#                 Distance = lambda a,b : tf.reduce_sum((a-b)*(a-b),-1)
#                 dis = Distance(a,positives)
#                 p = positives[tf.argmax(dis)]
#                 dis = Distance(a,negatives)
#                 n = negatives[tf.argmin(dis)]
    
                triplet = tf.maximum( self.L2(a,p) - self.L2(a,n) + self.margin, 0.0)
                
                triplets.append(triplet)
                
        loss = tf.reduce_mean(triplets)
#         print('triplets: ',len(triplets))
        if(len(triplets)==0):
            return tf.reduce_mean([0.])
        
        return loss
        
class no_batch_metric_loss(losses.Loss):
    
    def __init__(self,stage,N,margin):
        
        super(no_batch_metric_loss, self).__init__()
        self.shape = {
            'c1': 64,'c2': 32,'c3': 16,'c4': 16,'c5': 16
        }
        self.channel = {
            'c1': 128,'c2': 256,'c3': 728,'c4': 728,'c5': 2048
        }
        self.stage = stage
        self.N = N
        self.resize_shape = self.shape[self.stage]
        self.Distance = lambda x,y : tf.reduce_sum(abs(x-y),axis=-1)#lambda x,y : abs(x[1]-y[1]) + abs (x[2]-y[2])
        self.L2 = lambda x,y : tf.sqrt(tf.reduce_sum(tf.square(x-y),axis=-1))
        self.margin = margin
        self.resize = lambda x: tf.squeeze( tf.image.resize(tf.expand_dims(x,-1),[self.resize_shape,self.resize_shape],method='nearest'), -1 )
    
    def call(self, y_true, y_pred):
        
        batch = y_pred.shape[0]
        embedding = tf.reshape(y_pred[:,:16*16*2048],[-1,16,16,2048])  # none, 16, 16, 2048
        y_pred = tf.reshape(y_pred[:,16*16*2048:],[-1,256,256,2])     # none, 256, 256, 2
        
        
        true = tf.cast( tf.argmax(y_true, -1), tf.int64)
        true = self.resize(true) # none, 16, 16
        pred = tf.cast( tf.argmax(y_pred, -1), tf.int64)
        pred = self.resize(pred) # none, 16, 16
        
        
        anchor_id = tf.where((pred+true)==2)
        positive_id = tf.where((true-pred)==1)
        negative_id = tf.where((true-pred)==-1)
        
#         print('anchor shape: ',anchor_id.shape,'  positive shape: ',positive_id.shape,'  negative shape: ',negative_id.shape)
#         print(gt_water_id)
        
        anchors = tf.gather_nd(embedding,anchor_id)
        positives = tf.gather_nd(embedding,positive_id)
        negatives = tf.gather_nd(embedding,negative_id)

        triplets = []
        
        for img_id in range(batch):
            
#             randomly choose N anchors for each image
            img_anchor_id = tf.where(anchor_id[:,0]==img_id)
            N = min(self.N,len(img_anchor_id))
            img_anchor_id = tf.gather(img_anchor_id, tf.random.shuffle(range(N)))[:N]
            img_anchors = tf.gather_nd(anchors,img_anchor_id)
            
            
            for a in img_anchors:
                
                p,n = None,None
                candidate_positives_id = tf.where(positive_id[:,0]==img_id)
                if len(candidate_positives_id)!=0:
                    candidate_positives = tf.gather_nd(positives,candidate_positives_id)
                    dis = self.Distance(img_id,candidate_positives_id)
                    p = candidate_positives[tf.argmin(dis)]
                else:
                    continue

                candidate_negatives_id = tf.where(negative_id[:,0]==img_id)
                if len(candidate_negatives_id)!=0:
                    candidate_negatives = tf.gather_nd(negatives,candidate_negatives_id)
                    dis = self.Distance(img_id,candidate_negatives_id)
                    n = candidate_negatives[tf.argmin(dis)]
                else:
                    continue
    
                triplet = tf.maximum( self.L2(a,p) - self.L2(a,n) + self.margin, 0.0)
                
                triplets.append(triplet)
                
        loss = tf.reduce_mean(triplets)
#         print('triplets: ',len(triplets))
        if(len(triplets)==0):
            return tf.reduce_mean([0.])
        
        return loss        
        
# p,n = None,None
# candidate_positives_id = tf.where(positive_id[:,0]==img_id)
# if len(candidate_positives_id)!=0:
#     candidate_positives = tf.gather_nd(positives,candidate_positives_id)
#     dis = self.Distance(img_id,candidate_positives_id)
#     p = candidate_positives[tf.argmin(dis)]
# else:
#     continue

# candidate_negatives_id = tf.where(negative_id[:,0]==img_id)
# if len(candidate_negatives_id)!=0:
#     candidate_negatives = tf.gather_nd(negatives,candidate_negatives_id)
#     dis = self.Distance(img_id,candidate_negatives_id)
#     n = candidate_negatives[tf.argmin(dis)]
# else:
#     continue

