import dataset as dt #Importation du module dataset
import layers # Importation du module layers
import logging as log # Imporation du module logging pour la normalisation
import numpy as np # Importation du module numpy



# Définition de la classe Network qui représente le réseau de neurones
class Network:
    
    def __init__(self):
        # Définition des différentes couches nécessaires à l'execution
        self.conv1 = layers.convolution('conv1',[3,32,32],[6,3,5,5],1,bias=True)
        self.pool  = layers.maxpooling('maxpooling',2)
        self.conv2 = layers.convolution('conv2',[6,14,14],[16,6,5,5],1,bias=True)
        self.fc1   = layers.linear('fc1',16*5*5,120,bias=True)
        self.fc2   = layers.linear('fc2',120,84,bias=True)
        self.fc3   = layers.linear('fc2',84,10,bias=True)
        self.relu  = layers.relu('relu')

    def __call__(self,x):
        # Définition de la propagation de l'image dans le réseau
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Définition du répertoire
path = "/home/tanguy/Documents/Cours/M1-Python/1102/base/test_batch.bin"

# Création de l'instance de la classe dataset
data = dt.dataset('CIFAR-10',path)
# Utilisation de la méthode read_dataset pour lire le fichier binaire
imgs,labels = data.read_dataset()

# Récupération des différentes classes de la base
classes = data.get_classes()

# Instance de réseau
net = Network()

net.conv1.load_weight(np.load('params/conv1.weight.npy'))
net.conv2.load_weight(np.load('params/conv2.weight.npy'))

net.conv1.load_bias(np.load('params/conv1.bias.npy'))
net.conv2.load_bias(np.load('params/conv2.bias.npy'))


net.fc1.load_weight(np.load('params/fc1.weight.npy').T)
net.fc2.load_weight(np.load('params/fc2.weight.npy').T)
net.fc3.load_weight(np.load('params/fc3.weight.npy').T)

net.fc1.load_bias(np.load('params/fc1.bias.npy'))
net.fc2.load_bias(np.load('params/fc2.bias.npy'))
net.fc3.load_bias(np.load('params/fc3.bias.npy'))

# Boucle sur les images et labels de la base
for label,img in zip(labels,imgs):
    # Affichage de l'image courane
    data.display_dataset(img,label)
    # Propagation de l'image dans le réseau
    output = net(img)

    # Visualisation des scores de sortie
    print('Netoutput => ',output)

    # Récupération de l'indice du maximum dans le vecteur de score
    prediction = output.argmax()

    print('Network prediction : {} => {}'.format(prediction, classes[prediction]))
    print('Truth : {} => {}'.format(label,classes[label]))

    # Insertion dans les logs de la prédiction du réseau et du label
    log.info('Network prediction : {} => {}'.format(prediction, classes[prediction]))
    log.info('Truth : {} => {}'.format(label,classes[label]))
    exit(0)
