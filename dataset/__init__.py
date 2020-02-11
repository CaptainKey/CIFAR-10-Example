import numpy as np 
import matplotlib.pyplot as plt
import logging as log 
import argparse

# Classe de la base de donnée
class dataset:
    
    # Initialisation
    def __init__(self,name,path):
        self.name = name
        self.path = path
        self.classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        log.info('init - {}'.format(self.name))
        self.display_classes()

    # Méthode pour lire le fichier binaire
    def read_dataset(self):
        log.info('iread_dataset - {}'.format(self.name))

        labels = []
        imgs = []
        with open(self.path,"rb") as file:
            for j in range(10000):
                byte = file.read(1)
                labels.append(int.from_bytes(byte,byteorder='big'))
                byte_array = file.read(3072)
                img = [byte for byte in byte_array]
                imgs.append(np.array(img,'uint8').reshape(3,32,32))
        return imgs,labels

    # Méthode qui permet l'affichage d'une image de la base
    def display_dataset(self,img,label):
        log.info('display_dataset - {}'.format(self.name))

        shape = img.shape
        rgb = np.array([ [ [img[0][i][j],img[1][i][j],img[2][i][j]] for j in range(shape[2])] for i in range(shape[1])]) 

        plt.suptitle(' CIFAR -10 : {}'.format(self.classes[label]), y=0.8)
        ax = plt.subplot(1,4,1)
        ax.set_title('RGB')
        ax.axis('off')
        plt.imshow(rgb)

        ax = plt.subplot(1,4,2)
        ax.set_title('R')
        ax.axis('off')
        plt.imshow(img[0],cmap='Reds')

        ax = plt.subplot(1,4,3)
        ax.set_title('G')
        ax.axis('off')
        plt.imshow(img[1],cmap='Greens')

        ax = plt.subplot(1,4,4)
        ax.set_title('B')
        ax.axis('off')
        plt.imshow(img[2],cmap='Blues')

        plt.show()

    def get_classes(self):
        return self.classes

    # Affiche les différentes classes
    def display_classes(self):
        log.info("Classes")
        for key,value in enumerate(self.classes):
            log.info('{} => {}'.format(key,value))


"""
    Le code ci-dessous est executé si et seulement si c'est le module qui est executé
    ex : python3 __init__.py --path /home/tanguy/Documents/Cours/M1-Python/1102/base/data_batch_1.bin
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read and display CIFAR-10')
    # Définition de/des arguments
    parser.add_argument('--path', metavar='float', type=str, nargs='?',required=True,help='Path to .bin files')
    # Parse des arguments
    args = parser.parse_args()

    # Execution d'un code pour lire la base et afficher une image
    data = dataset('CIFAR-10',args.path)
    imgs,labels = data.read_dataset()
    for label,img in zip(labels,imgs):
        data.display_dataset(img,label)
        exit(0)
