import numpy as np
import cv2
import os
import tensorflow as tf

def get_filelist(home_dir, ext=['png','jpg'], begin=[], number_name=False):
    filelist = []
    for dirpath, dirnames, filenames in os.walk(home_dir):
        for filename in filenames:
            if filename[0] != '.':
                ext_fag = False
                if len(ext)>0:
                    for e in ext:
                        if filename.endswith(e):
                            ext_fag=True
                            break
                else:
                    ext_fag = True

                begin_fag = False
                if len(begin)>0:
                    for b in begin:
                        if filename.startswith(b):
                            begin_fag=True
                            break
                else:
                    begin_fag = True
                if begin_fag and ext_fag:
                    fn = os.path.join(dirpath,filename)
                    filelist.append(fn)
    filelist = sorted(filelist)
    if number_name:
        filelist = sorted(filelist,key=lambda x: int(os.path.basename(x)[:-4]))
    return filelist

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpack_im(im_vector):
    im = np.zeros([32,32,3])
    for i in range(3):
        im[:,:,i] = im_vector[i*1024:i*1024+1024].reshape([32,32])
    im = im.astype('uint8')
    return im

def val2bin(labels):
    bin = np.zeros([len(labels),10])
    for i in range(len(labels)):
        bin[i, labels[i]]=1
    return bin.astype('uint')


def load_cifar(filename):
    dict = unpickle(filename)
    # for key in dict:
    #    print(key)
    # print(dict[b'batch_label'])
    y = np.array(dict[b'labels'])
    #labels = val2bin(labels)
    x = np.array(dict[b'data'])
    return x, y

def divide_data(x, y, train_percent = 0.7, shuffle=True):
    choices = np.random.choice(len(x),len(x),replace=False)
    if shuffle:
        x = x[choices]
        y = y[choices]
    # divide to train and test
    train_num = int(len(x)*train_percent)
    x_train = x[:train_num]
    y_train = y[:train_num]
    x_test = x[train_num:]
    y_test = y[train_num:]
    return x_train, y_train, x_test, y_test    

def transfer_x(x):
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input
    import numpy as np

    # get shape of the extracted feature
    im = unpack_im(x[0,:])
    x_vec = image.img_to_array(im)
    print('x_vec shape', x_vec.shape)

    x_new = np.zeros([len(x)]+list(x_vec.shape))


    for i in range(len(x)):
        im = unpack_im(x[i,:])
        x_vec = image.img_to_array(im)
        x_new[i, :] = x_vec
    # print('x_new', x_new[len(x)-1])
    
    x_new = preprocess_input(x_new)
    return x_new    

def prepare(filename):
    x, y = load_cifar(filename)
    print('x, y shape', x.shape, y.shape)
    x = transfer_x(x)
    print('transfer x', x.shape)
    print('x[i]', x[-1])
    from keras.applications.vgg16 import VGG16
    model = VGG16(weights='imagenet', include_top=False)
    features = model.predict(x)
    print('feature shape', features.shape)
    np.savez(filename+'.npz', features=features, x=x, y=y)

def train_model(filename, epochs=10):
    data = np.load(filename)
    x = data['x']
    y = data['y']
    print('x y shape', x.shape, y.shape)
    # divide data
    x_train, y_train, x_test, y_test = divide_data(x, y, train_percent=0.95)

    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu),
        # keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation=tf.nn.relu),
        # keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('loss, acc', test_loss, test_acc)

    test_loss, test_acc = model.evaluate(x_train, y_train)
    print('loss, acc', test_loss, test_acc)

    # print('y_test', y_test)
    count=0
    for i in range(len(x_test)):
        feat = np.expand_dims(x_test[i], axis=0)
        pred = model.predict(feat)
        #print(np.argmax(pred), y_test[i])
        if (np.argmax(pred)==y_test[i]):
            count+=1
        # print(np.argmax(pred), y_test[i])
    print('count', count, len(y_test))
    return model
    
def test_model(model, folder):
    files = get_filelist(folder)
    count=0
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import VGG16

    model_vgg = VGG16(weights='imagenet', include_top=False)

    for i in range(len(files)):
        im = cv2.imread(files[i])
        im = image.img_to_array(im)
        im = preprocess_input(im)
        im = np.expand_dims(im, axis=0)
        feat = model_vgg.predict(im)
        feat = feat.reshape((1,-1))
        pred = model.predict(feat)
        pred = pred.reshape((-1))
        # pred = np.vstack((pred, np.arange(len(pred))))
        pred_order = np.flip(np.argsort(pred),axis=0)
        pred_value = np.flip(np.sort(pred), axis=0)
        # print(pred)
        print(files[i])
        print(pred_order[:3])
        print(pred_value[:3])
        # print(np.argmax(pred))
        # if (np.argmax(pred)==y_test[i]):
        #     count+=1
        # print(np.argmax(pred), y_test[i])


def new_func():
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_test.shape)
    x = np.vstack((x_train, x_test))
    y = np.vstack((y_train, y_test))
    
    # x = transfer_x(x)
    print('transfer x', x.shape)
    print('x[i]', x[-1])
    from keras.applications.vgg16 import VGG16
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(weights='imagenet', include_top=False)
    # model = VGG16(weights='imagenet', include_top=False)
    features = model.predict(x)
    print('feature shape', features.shape)
    np.savez('new.npz', x=features, y=y)


def read_data(filename):
    data = np.load(filename)
    features = data['features']
    x = features.reshape(len(features),-1)
    y = data['y']
    print('x y shape', x.shape, y.shape)
    return x, y


def stack_vector(mat,vec):
    mat = np.vstack((mat,vec)) if mat.size else vec
    return mat

def combine_data():
    fls = ['./data_batch_1.npz', './data_batch_2.npz', './data_batch_3.npz', './data_batch_4.npz', './data_batch_5.npz']
    x = np.array([])
    y = np.array([])
    for fl in fls:
        x0, y0 = read_data(fl)
        x = stack_vector(x, x0)
        y = np.hstack((y, y0)) if y.size else y0
    print('')
    np.savez('data',x=x,y=y)
    print('x y shape', x.shape, y.shape)
    pass

if __name__ == "__main__":
    import sys
    # prepare(filename=sys.argv[1])
    # combine_data()
    # train_model('data.npz')
    # new_func()
    # data = np.load('new.npz')
    # x = data['x']
    # y = data['y']
    # x = x.reshape(len(x),-1)
    # np.savez('resnet.npz', x=x, y=y)
    model = train_model('data.npz', epochs=5)  #epochs=5
    test_model(model=model, folder='round2')