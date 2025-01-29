
import numpy as np
import random
def lejeudesfous(nb_joueurs=2, score_but=5, N=100):
    scores = [0 for _ in range(nb_joueurs)]
    while max(scores) < score_but :
        r = random.randint(1,N)
        guesses = [-1 for _ in range(nb_joueurs)]
        for joueur in range(nb_joueurs) :
            guesses[joueur] = int(input("Que vaut le nommbre mystère ? "))

        i_mini = []
        ecart_mini = N+1
        for joueur in range(nb_joueurs):
            if abs(guesses[joueur]-r) == ecart_mini :
                i_mini.append(joueur)
                # ecart_mini = abs(guesses[joueur]-r)
            if abs(guesses[joueur]-r) < ecart_mini :
                i_mini = [joueur]
                ecart_mini = abs(guesses[joueur]-r)

        print(f"le nombre secret était {r}")
        if len(i_mini) > 1 :
            print("égalité! Personne ne gagne de point'")
        else :
            scores[i_mini[0]] += 1
            print( f"le joueur {i_mini[0]} gagne 1 point !")

        print("Rappel des scores :")
        for j in range(nb_joueurs) :
            print(f"Joueur {j+1} : {scores[j]}")


    gagnant = np.argmax(scores)
    print(f"fin du jeu ! Le joueur {gagnant+1} a gagné !")



