
def MNIST():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/home/uga_qinglin/Documents/github/myDBN/MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    rbm=RBM(trX.shape[1], 500)
    rbm.build()
    h1=rbm.train(trX)
    np.save('W1.npy',sess.run(rbm.W))

    from utils import tile_raster_images
    tile_raster_images(X=sess.run(rbm.W).T, img_shape=(28, 28), tile_shape=(25, 20), tile_spacing=(1, 1))
    import matplotlib.pyplot as plt
    from PIL import Image
    image = Image.fromarray(tile_raster_images(X=(sess.run(rbm.W)).T, img_shape=(28, 28) ,tile_shape=(25, 20), tile_spacing=(1, 1)))
    ### Plot image
    plt.rcParams['figure.figsize'] = (18.0, 18.0)
    imgplot = plt.imshow(image)
    imgplot.set_cmap('gray')
    plt.savefig('Weights1.png')


def MRI():
    import scipy.io as sio

    trX = np.load("/srv1/HCP_4mm/205725/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR_hp200_s4.feat/signal.npy")
    import sklearn.preprocessing
    trX=sklearn.preprocessing.scale(trX)

    rbm=RBM(trX.shape[1], 100)
    rbm.build()

    rbm.train(trX)
    np.save('W1.npy',sess.run(rbm.W))

    h1=rbm.rbmup(trX)
    np.save('h1.npy',h1)
    rbm=RBM(100, 100)
    rbm.build()
    rbm.train(h1)
    np.save('W2.npy',sess.run(rbm.W))

    h2=rbm.rbmup(h1)
    np.save('h2.npy',h2)
    rbm=RBM(100, 100)
    rbm.build()
    rbm.train(h2)
    np.save('W3.npy',sess.run(rbm.W))

    from subprocess import call
    call(["matlab","-r","run('Map.m');quit;"])

def MRI900():
    import scipy.io as sio
    
    
    #trX = np.load("/srv1/HCP_4mm/100307/MNINonLinear/Results/tfMRI_RELATIONAL_LR/tfMRI_RELATIONAL_LR_hp200_s4.feat/signal.npy")
    #rbm1 = RBM(trX.shape[1], 100)
    rbm1 = RBM(28549, 100)
    rbm2 = RBM(100, 100)
    rbm3 = RBM(100, 100)
    rbm1.build()
    rbm2.build()
    rbm3.build()

    from pathlib import Path
    from itertools import islice
    for ite in range(0,100):
        #for path in pathlist[:2]:
	pathlist = Path("/srv1/HCP_4mm/").glob('**/tfMRI_EMOTION_LR/**/signal.npy')
        gen = (path for path in pathlist)
        #for p in islice(gen,1,2):
        print 'Epoch: %d' % ite  # , 'Error: %f' % errors[-1]
        for p in islice(gen, 1,100):
            # because path is object not string
            path_in_str = str(p)
            #print(idx)
            print(path_in_str)
            trX=np.load(path_in_str)

            import sklearn.preprocessing
            trX=sklearn.preprocessing.scale(trX)

            from sklearn.utils import shuffle
            trX = shuffle(trX)

            rbm1.train(trX)
            h1 = rbm1.rbmup(trX)
            rbm2.train(h1)
            h2 = rbm2.rbmup(h1)
            rbm3.train(h2)
            h3 = rbm3.rbmup(h2)

    np.save('W1.npy', sess.run(rbm1.W))
    np.save('W2.npy', sess.run(rbm2.W))
    np.save('W3.npy', sess.run(rbm3.W))
    np.save('h1.npy', h1)
    np.save('h2.npy', h2)
    np.save('h2.npy', h3)

    from subprocess import call
    call(["matlab", "-r", "run('Map.m');quit;"])
if __name__ == '__main__':
    MRI900()
    #MNIST()

