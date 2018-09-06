#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from scipy.optimize import fmin_ncg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from six.moves import xrange
import random
import PIL.Image as Pimage
import os

class InfluenceExplainer(object):
    def __init__(self,model):
        super(InfluenceExplainer, self).__init__()
        self.model = model
        self.n_classes = self.model.n_classes
        self.sess = model.sess
        self.params = [tf.reshape(param, [-1,]) for param in model.GetWeights()]
        self.num_params = len(self.params)
        self.input_, self.labels_ = model.GetPlaceholders()
        self.v_placeholder = [
                tf.placeholder(
                    tf.float32,
                    shape=a.get_shape(),
                    name= "params"+str(i)+"_pl"
                )
                for i, a in zip(xrange(self.num_params),self.params)
        ]
        self.grad_loss = [tf.reshape(grad, [-1,]) for grad in self.model.GetGradLoss()]
        self.hvp = self._get_approx_hvp(self.grad_loss, self.params, self.v_placeholder)
        self.vec_to_list = self._get_vec_to_list()
        self.cached_influence = {}
                
        
    def Explain(self, test_img, additional_args):
        """
        Explain test image by showing the training sampels with most influence
        on the models classification of the image.
        This method uses the approximation of the Hessian as defined in 
        "Understanding Black-box Predictions via Influence Functions" by 
        Pang Wei Koh and Percy Liang.
        The method involves precomputing
        s_test = Hess_w_min^-1 x Del_w(L(z_test,w_min))
        f.a. test samples, s_test is then substituted into the Inf fn.
        By accurately approximating s_test computation is much faster without 
        much cost.
        Input:
            test_img : test image to explain
            n_max : number of train images to select for explanation, i.e.
            first n_max images from train set in order of descending 
        """
        
        if len(test_img.shape) < 4:
            test_img = np.expand_dims(test_img, axis = 0) 

        ### no default additional arguments (REQUIRED)
        print("influence function explain")

        if "train_x" in additional_args:
            self.train_imgs = additional_args["train_x"]
        else:
            raise ValueError('no training images passed in additional arguments')

        if "train_y" in additional_args:
            self.train_lbls = additional_args["train_y"]
        else:
            raise ValueError('no training labels passed in additional arguments')

        if "max_n_influence_images" in additional_args:
            n_max = additional_args["max_n_influence_images"]
        else:
            n_max = 9

        if "label" in additional_args:
            label = additional_args["label"]
            prediction_scores = ""
        else:
            label, prediction_scores = self.model.Predict(test_img, True)



        if(isinstance(label,list)):
            label = np.array(label)
        if label.size == 1:
            one_hot_label = np.array([0] * self.n_classes)
            one_hot_label[label[0]] = 1
            label = one_hot_label
        label = label.reshape(-1, self.n_classes)



        if "damping" in additional_args:
            self.damping = additional_args["damping"]
        else:
            self.damping = 0.0
        if "mini_batch" in additional_args:
            self.mini_batch = additional_args["mini_batch"]
        else:
            self.mini_batch = True

        """
        If cache is a string, and the image has not been cached before, the resulting max_imgs list will be cached under the test_img
        at ./cached_images/cache
        If cache is a string, and the image has been cached previously at ./cached_images/cache, the previously cached results are loaded
        and returned
        """
        cached_examples = {}

        if "cache" in additional_args:
            cache = additional_args["cache"]
            cache_file = os.path.join(os.getcwd(), "cached_images", cache)
            if os.path.isfile(cache_file):
                cache_file = np.load(cache_file)
                cached_img = cache_file.files["test_img"]
            else:
                cached_img = []
            if np.array_equal(test_img, cached_img):
                cached_examples[cache] = cache_file.files["train_imgs"]
            if cache in cached_examples:
                if cached_examples[test_img].shape[0] >= n_max:
                    return self.FormCollage(cached_examples[test_img][:n_max])
        else:
            cache = False

        try:
            test_img = self.model.CheckInputArrayAndResize(test_img,self.model.min_height,self.model.min_width)
        except:
            print("couldn't use model image size check")  
        
        print("test_img.shape",test_img.shape)
        feed_dict = {
            self.input_: test_img,
            self.labels_: label
        }


        test_loss_gradient = self.sess.run(self.grad_loss,feed_dict)
        s_test = self._get_approx_inv_hvp(test_loss_gradient)
        s_test = [prod.reshape(-1,) for prod in s_test]
        influences = []
        for train_pt in zip(self.train_imgs, self.train_lbls):
            influences.append(self._influence_on_loss_at_test_image(s_test,train_pt))
#           Get the n_max first training images in order of descending influence
        idcs = np.argsort(influences)[-n_max:]
        max_imgs = self.train_imgs[idcs]
        if cache:
            cached_examples[cache] = max_imgs
            self.CacheInfluence(cache, test_img, max_imgs)
            
        max_imgs = self.FormCollage(max_imgs)

        if(not isinstance(prediction_scores,list)):
            prediction_scores = prediction_scores.tolist()
            
        return max_imgs, "", np.argmax(label), {"prediction_scores":prediction_scores}
            
            
    
    def FormCollage(self, img_arr, width = 256, height = 256, cols = 0, rows = 0):
        """
        Added functionality for collaging the images represented as arrays,
        that are returned by Explain()
        returns a PIL Image object
        Input:
            img_arr: list of images each in array format
            width: the desired pixel width of the output collage
            height: the desired pixel height of the output collage
            cols: the number of images per collage row
            rows: the number of images per collage column
        """

#        if cols or rows is 0
        if not (cols and rows):
        #     find nearest square
            n = 0
            while (n**2) < img_arr.shape[0]:
                n = n + 1
            cols = n
            rows = n
            for i in range(img_arr.shape[0], n**2):
                np.append(img_arr, np.zeros(img_arr.shape[1:]))

        width = img_arr.shape[1]
        height = img_arr.shape[2]
        x = 0
        y = 0
        collage = np.zeros([rows * width, cols * height, img_arr.shape[3]])
        for i in range(rows):
            for j in range(cols):
                collage[x:x + width, y:y + height] = img_arr[i*n+j]
                y = y + height
            y = 0
            x = x + width

        return collage[:,:,[2,1,0]]
                

    def CacheInfluence(self, filename, test_img, train_imgs):
        # path = os.path.join(os.getcwd(),filename)
        path = os.path.join(".","cached_images", filename)
        np.savez(path,test_img=test_img, train_imgs=train_imgs)
        print("Saved test and train images at: " + path + ".npz")

    def _influence_on_loss_at_test_image(self, s, train_pt):
        """
        Approximates the influence an image from the models training set has on
        the loss of the model at a test point.
        I_up_loss(z, z_test) is defined as:
            -Grad_w x Loss(z_test,w_min)^T x Hess_w_min^-1 x Grad_w x Loss(z,w_min)
        We precompute s_test equiv. to Hess_w_min^-1 x Grad_w x Loss(z_test, w_min)
        for a test point in order to save time on computations
        
        Input:
            s : the vector precomputed by _get_approx_inv_hvp
            train_pt : a 2-tuple containing the image label pair of one training
            point
        """

#        Get loss  Loss(z,w_min)
        feed_dict = {
                self.input_ : np.expand_dims(train_pt[0],axis=0),
                self.labels_ : train_pt[1].reshape(-1,self.n_classes)
        }
#        Get gradient of loss at training point: Grad_w x Loss(z,w_min)
        grad_train_loss_w_min = self.sess.run(self.grad_loss, feed_dict)
        grad_train_loss_w_min = [grad.reshape(-1,) for grad in grad_train_loss_w_min]
#        Calculate Influence
        influence_on_loss_at_test_image = np.dot(np.concatenate(s),np.concatenate(grad_train_loss_w_min) / len(self.train_lbls))
        
        return influence_on_loss_at_test_image
        
    def _get_approx_inv_hvp(self, v):
        """
        Returns the value of the product of the inverse Hessian of a fn and a 
        given vector v
        This value is approximated via Newton Conjugate Gradient Descent:
        """
        v = [a.reshape(-1,) for a in v]
#        function to minimise
        fmin = self._get_fmin_inv_hvp(v)
#        gradient of function
        grad_fmin = self._get_grad_fmin(v)
#        hessian of function
        hess_fmin_p = self._get_fmin_hvp
#        callback function
        fmin_cg_callback = self._get_cg_callback(v)


        approx_inv_hvp = fmin_ncg(
            f = fmin,
            x0 = np.concatenate(v),
            fprime = grad_fmin,
            fhess_p = hess_fmin_p,
            callback = fmin_cg_callback,
            avextol = 1e-8,
            maxiter = 100
        )
        
        return self.vec_to_list(approx_inv_hvp)
        
    def _get_approx_hvp(self,grads,xs,t):
        """
        Returns the value of the product of the Hessian of a fn ys w.r.t xs,
        and a vector t
        Approximates the product using a backprop-like approach to obtain an
        implicit Hessian-vector product
        """
        
#        N.B. Need to take more time to understand this function, it's almost
#        completely parroted from Percy's implementation
        
#        Validate input shapes
        length = len(xs)
        if len(t) != length:
            raise ValueError("xs and v must have the same length")
        
#        First backprop        
        assert len(grads) == length
        
        elemwise_prods = [
#                stop gradient tells the tf graph to treat the parameter as a 
#                constant, i.e. to not factor it in to gradient computation
                tf.multiply(grad_elem, array_ops.stop_gradient(v_elem))
                for grad_elem, v_elem in zip (grads, t) if grad_elem is not None
        ]
        
#        Second backprop
        grads_w_none = tf.gradients(elemwise_prods, xs)
        
        return_grads = [
                grad_elem if grad_elem is not None else tf.zeros_like(x) \
                for x, grad_elem in zip(xs, grads_w_none)
        ]
        
        return return_grads
        
    
    def _minibatch_hvp(self, v):
        num_samples = self.train_imgs.shape[0]
        if self.mini_batch == True:
            batch_size = 30
            # assert num_samples % batch_size == 0
        else:
            batch_size = num_samples
            
        num_iter = int(num_samples / batch_size)
        
        hess_vec_val = None
        for i in xrange(num_iter):
            idx = random.sample(xrange(i,i+batch_size), batch_size)
            feed_dict = {
                    self.input_: self.train_imgs[idx],
                    self.labels_: self.train_lbls[idx]
            }
            
            for pl, vec in zip(self.v_placeholder, v):
                feed_dict[pl] = vec
            
            hess_vec_val_tmp = self.sess.run(self.hvp, feed_dict)
            if hess_vec_val is None:
                hess_vec_val = [b / float(num_iter) for b in hess_vec_val_tmp]
            else:
                hess_vec_val = [a + (b / float(num_iter)) for (a,b) in zip(hess_vec_val, hess_vec_val_tmp)]
                
        hess_vec_val = [a + self.damping * b for (a,b) in zip(hess_vec_val,v)]
        hess_vec_val = [param_val.reshape(-1,) for param_val in hess_vec_val]
        return hess_vec_val
    def _get_fmin_inv_hvp(self, v):
        """
        H_w_min ^ -1 x v is equiv. to 
        argmin_t{0.5 * t^T * H_w_min * t - (v^T *t)}
        By minimising this using NCG we can approximate s_test (see self.Explain)
        """
        def fmin_inv_hvp(x):
            hvp = self._minibatch_hvp(self.vec_to_list(x))
            inv_hvp_val = 0.5 * np.dot(np.concatenate(hvp),x) - np.dot(np.concatenate(v),x)
            return inv_hvp_val
        return fmin_inv_hvp
    
    def _get_grad_fmin(self,v):
        """
        Returns the function for calculating the gradient of the fmin function
        at a vector x
        The vector v is treated as a conbstant and varies with each test point
        """
        def _grad_fmin(x):
            
            hvp = self._minibatch_hvp(self.vec_to_list(x))
            grad_val = np.concatenate(hvp) - np.concatenate(v)
            return grad_val
        return _grad_fmin
    
    def _get_fmin_hvp(self, v, p):
        """
        Returns the Hessian of the fmin function
        """
        hvp = self._minibatch_hvp(self.vec_to_list(p))
        return np.concatenate(hvp)
    
    def _get_cg_callback(self, v):
        """
        This function appears to only serve as a display option for the N-CG process
        """
        fmin = self._get_fmin_inv_hvp(v)

        def _fmin_inv_hvp_split(x):
            hvp = self._minibatch_hvp(self.vec_to_list(x))
            inv_hvp_val = 0.5 * np.dot(np.concatenate(hvp),x) - np.dot(np.concatenate(v),x)
            return inv_hvp_val
        
        def cg_callback(x):
#            x is current params
            v = self.vec_to_list(x)
            v = [a.reshape(-1) for a in v]
            idx_to_rm = 5
            
            single_train_ex = np.expand_dims(self.train_imgs[idx_to_rm],axis=0)
            single_train_lbl = self.train_lbls[idx_to_rm].reshape(-1,self.n_classes)
            feed_dict = {
                    self.input_ : single_train_ex,
                    self.labels_ : single_train_lbl
            }
            grad_train_loss = self.sess.run(self.grad_loss, feed_dict)
            predicted_del_loss = np.dot(np.concatenate(v), np.concatenate(grad_train_loss)) / self.train_lbls.shape[0]
            
        return cg_callback
    
    def _get_vec_to_list(self):
        params = self.sess.run(self.params)
        def vec_to_list(v):
            # if len(v.shape) == 1:
            #     v = np.expand_dims(v, axis = 1)
            return_ls = []
            pos = 0
            for p in params:
                return_ls.append(v[pos : pos+len(p)])
                pos += len(p)
            
            assert pos == len(v)
            return return_ls
        
        return vec_to_list
