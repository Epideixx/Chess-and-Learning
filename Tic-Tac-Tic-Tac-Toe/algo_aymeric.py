

#Connaissances : dico de la forme {position : {"succès" : ..., "joués" : ..., } }
import copy
import random
import math
import game as optimized 
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

#On définit la structure du réseau :
model = keras.Sequential()
model.add(layers.Dense(512, input_dim=81*2 + 9*3 + 9, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(82, activation='sigmoid'))
model.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])



def legal_moves(position):
    return (position.all_moves())

def hach(position):
    if position.restriction != None:
        liste=position.board+[100*position.restriction]
    else :
        liste=position.board+[1000]
    liste=tuple(liste)
    return hash(liste)


#La fonction suivante donne les positions accessibles depuis une position fixée
def legal_positions(position):
    pos=[]
    plateau = position.board
    nbreRonds = 0
    nbreCroix = 0
    for k in plateau:
        if k == "O":
            nbreRonds += 1
        if k == "X":
            nbreCroix += 1
    if nbreRonds == nbreCroix:
        symbole="O"
    else :
        symbole="X"
    for c in legal_moves(position) :
        pos2=position.copy()
        pos2.play(c,symbole)
        pos.append(pos2)
    return pos

def estFinal(position):
    return (position.is_terminal() != 1e-3)

def resultat(position):
    assert estFinal(position)
    return (position.is_terminal())

def isLeaf(position,connaissances):
    coups=legal_positions(position)
    for k in coups:
        if hach(k) in connaissances:
            return False
    return True

def retropropage(connaissances, chemin):
    res = resultat(chemin[-1])
    
    for c in chemin:
        if hach(c) not in connaissances:
            connaissances[hach(c)]={"joués":0,"succès":0}
        connaissances[hach(c)]["joués"]+=1
    if res == 0 :
        for c in chemin:
            connaissances[hach(c)]["succès"]+=0 #Je laisse cette ligne au kazou
    else:
        assert res
        for k in range(len(chemin)):
            if (k - len(chemin)) % 2==1:
                connaissances[hach(chemin[k])]["succès"]+=1
            if (k - len(chemin)) % 2==0:
                connaissances[hach(chemin[k])]["succès"]-=1
    return connaissances     

def clean_base(liste):
    """
    Fonction auxiliaire de start_training qui va separer la base de donnees (liste) au format
    (etat du jeu, politique dans l'etat, qui gagne a la fin) en train_x et train_y pour
    l'entrainement du reseau de neurones
    """
    train_x = [s for (s, _, _) in liste]
    train_y = [np.append(pi, end) for (_, pi, end) in liste]

    return np.array(train_x), np.array(train_y)

def jeuAleatoire(connaissances,chemin):

    etat=chemin[-1]
    while not estFinal(etat):
        coups = legal_positions(etat)
        n=len(coups)
        #print(n)
        k=random.randint(0,n-1)
        etat = coups[k] 
        chemin.append(etat)
    return retropropage(connaissances,chemin)


def UCB(position,connaissances,parent,joueur):
    probability_s = model.predict(np.array([position.representation_reseau(joueur)]))
    if connaissances[hach(position)]["joués"]==0:
        return 10000
    return (connaissances[hach(position)]["succès"]/connaissances[hach(position)]["joués"])+2*probability_s*math.sqrt(math.log(max(1,connaissances[hach(parent)]["joués"]))/connaissances[hach(position)]["joués"])

def IA_MCTS(position,connaissances,num_mcts=100):
    base_de_donnees_temporaire=[]
    for k in range(num_mcts): #On mettra 1600 après
        etat=position
        chemin=[etat]
        chemin_hash=[hach(etat)]
        if position.turn == 0 :
            joueur = "O"
        elif position.turn == 1 :
            joueur = "X"
        if not hach(etat) in connaissances:
            connaissances[hach(etat)] = {"succès" : 0, "joués" : 0}
        m=0 #Pour éviter les éventuels cycles dus au hachage
        while m<100:
            if estFinal(etat):
                connaissances = retropropage(connaissances,chemin)
                l = len(base_de_donnees_temporaire)
                end = position.is_terminal()
                base_de_donnees = [(s, pi, end) if (l-i) % 2 == 0 else (s, pi, -end) for i, (s, pi) in enumerate(base_de_donnees_temporaire)]
                #print("EstFinal !")
                break
            
            elif isLeaf(etat,connaissances) and connaissances[hach(etat)]["joués"]==0:
                connaissances = jeuAleatoire(connaissances,chemin)
                for co in range(81):
                    if co not in position.all_moves():
                        inte=0
                    else :

                        posco = position.copy()
                        posco.play(co,joueur)
                        if hach(posco) not in connaissances:
                            connaissances[hach(posco)] = {"succès" : 0, "joués" : 0}
                        if hach(position) not in connaissances:    
                            connaissances[hach(position)] = {"succès" : 0, "joués" : 0}
                        inte= connaissances[hach(posco)]["joués"]/connaissances[hach(position)]["joués"]
                    base_de_donnees_temporaire.append((position.representation_reseau(joueur),inte))
                break


            else:
                #print("On change !")
                if position.turn == 0 :
                    joueur = "O"
                elif position.turn == 1 :
                    joueur = "X"
                coups = legal_positions(etat)
                maxUCB=0
                coup_envisage=coups[0]
                for c in coups :
                    if hach(c) not in connaissances:
                        connaissances[hach(c)] = {"succès" : 0, "joués" : 0}
                    if UCB(c,connaissances,chemin[-1],joueur) >= maxUCB and c not in chemin:
                        maxUCB = UCB(c,connaissances,chemin[-1],joueur)
                        coup_envisage = c
                for co in range(81):
                    if co not in legal_moves:
                        inte=0
                    else :
                        posco = position.copy()
                        posco.play(co)
                        inte= connaissances[hash(posco)]["joués"]/connaissances[hash(position)]["joués"]
                    base_de_donnees_temporaire.append((position.representation_reseau(joueur),inte))

                etat = coup_envisage 
                chemin.append(etat)
                m+=1
        
        l = len(base_de_donnees_temporaire)
        end = position.is_terminal()
        base_de_donnees = [(s, pi, end) if (l-i) % 2 == 0 else (s, pi, -end) for i, (s, pi) in enumerate(base_de_donnees_temporaire)]
        #print(base_de_donnees)
        train_x, train_y = clean_base(base_de_donnees)
        model.fit(train_x, train_y, epochs=10)            

        #print("tour "+ str(k+1))
    #On a amélioré notre arbre : on doit maintenant choisir notre coup 

    #print(-connaissances[hach(position)]["succès"]/connaissances[hach(position)]["joués"])
    #1 : commencer est très bon. -1 : commencer est très mauvais !
    #print(connaissances)


    plateau = position.board
    nbreRonds = 0
    nbreCroix = 0
    for k in plateau:
        if k == "O":
            nbreRonds += 1
        if k == "X":
            nbreCroix += 1
    if nbreRonds == nbreCroix:
        symbole="O"
    else :
        symbole="X"
    
    l=legal_moves(position)
    m=0
    total=0
    for c in l:
        pos2=position.copy()
        pos2.play(c,symbole)
        visites = connaissances[hach(pos2)]["joués"]
        total+=visites
        if visites > m:
            m = visites
            coup = c

    return coup,connaissances

#Test :
pos=optimized.TicTacTicTacToe()
#pos.play(2,"O")
pos2=optimized.TicTacTicTacToe()
pos2.play(2,"O")
#print(hach(pos))
#print(hach(pos2))
con = {hach(pos):{"joués":0,"succès":0}}

#print(IA_MCTS(pos,con))
#print(jeuAleatoire(pos,con))
#print(estFinal(pos))
#print(pos.is_terminal())
#print(len(legal_positions(pos)))

