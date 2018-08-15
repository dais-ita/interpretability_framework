See /readme.html for details of the APIs and User Interface

Errors

1. ModelsAAS.py/models/archive
Description: Unable to request model archive for non-folder models (e.g. keras_cnn, keras_vgg etc)

Error trace:
[2018-08-05 09:17:54,007] ERROR in app: Exception on /models/archive [GET]
Traceback (most recent call last):
  File "/usr/local/lib/python2.7/site-packages/flask/app.py", line 2292, in wsgi_app
    response = self.full_dispatch_request()
  File "/usr/local/lib/python2.7/site-packages/flask/app.py", line 1815, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "/usr/local/lib/python2.7/site-packages/flask/app.py", line 1718, in handle_user_exception
    reraise(exc_type, exc_value, tb)
  File "/usr/local/lib/python2.7/site-packages/flask/app.py", line 1813, in full_dispatch_request
    rv = self.dispatch_request()
  File "/usr/local/lib/python2.7/site-packages/flask/app.py", line 1799, in dispatch_request
    return self.view_functions[rule.endpoint](**req.view_args)
  File "ModelsAAS.py", line 176, in GetModelZip
    zipdir(trained_model_dir_path, zipped_model_path)
  File "ModelsAAS.py", line 23, in zipdir
    shutil.make_archive(zip_path, 'zip', path)
  File "/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/python2.7/shutil.py", line 561, in make_archive
    os.chdir(root_dir)
OSError: [Errno 20] Not a directory: '/Users/davebraines/Documents/DL_work/p5_afm_2018_demo/models/keras_cnn/saved_models/Gun_Wielding_Image_Classification'
127.0.0.1 - - [05/Aug/2018 09:17:54] "GET /models/archive?dataset_name=Gun%20Wielding%20Image%20Classification&model_name=keras_cnn HTTP/1.1" 500 -

2. ModelsAAS.py/models/archive
Description: dataset_name parameter needs to be lowercased to prevent errors on case sensitive operating systems.
e.g. the model name in JSON is "Gun Wielding Image Classification" but the directory on the file system is "gun_wielding_image_classification".
This should be done server side because the underscore substitution for spaces is done there already.
