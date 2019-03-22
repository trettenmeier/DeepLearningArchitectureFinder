# Klasse, welche die Methoden des genetischen Algorithmus abbildet

from random import randint
from random import uniform
from random import choice
from datetime import datetime
import pickle

# meine Dateien
from modelbuilder import Modelbuilder


class gene:
    idCounter = 0
    epochs = 0

    def __init__(self):
        self.generation = 0
        self.genCode = [8]
        self.fitness = 0
        self.id = gene.idCounter
        gene.idCounter += 1

    def set_idCounter(self, idCounter):
        gene.idCounter = idCounter

# erzeugt die initiale Population
def populationErzeugen(anzahl, epochen, modus):
    gene.epochs = epochen
    population = []

    if modus:   # wenn "single-modus == True"....
        population.append(gene())
        population[0].genCode = [17, 2, 20, 9, 6, 18, 3, 4] # ...werden diese Gene benutzt
        
    else:
        for i in range(0,anzahl):
            population.append(gene())
            population[i].genCode = [
                randint(1,22),
                randint(1,22),
                randint(1,22),
                randint(1,22),
                randint(1,22),
                randint(1,22),
                randint(1,22),
                randint(1,22)]
        
    # Logfile anlegen
    f= open("output/zz_log_genetischer_alg.txt", "a+", 1)
    f.write("Log erstellt: " + str(datetime.now().date()) + " " + str(datetime.now().time()) + "\n")
    f.close()

    return population

# wir haben eine Population mit Genen, die aber noch trainiert werden muss
def berechneFitness(population, epochen):
    for element in population:
        if element.fitness == 0:    # training, nur wenn es nicht schon erfolgt ist.
            # Trainieren der Elemente
            model_neu = Modelbuilder(element)
            model_neu.add_GenCode_To_Model()
            myFitness = model_neu.finalize_Model(epochen)

            # Fitness-Wert zugeweisen
            berechneFitnessEinzel(element, myFitness)

            # Logging (einzeln)
            f= open("output/zz_log_genetischer_alg.txt", "a+", 1)
            f.write(str(element.id) + "\t" + str(element.generation) + "\t" + str(element.genCode) 
            + "\t" + str(element.fitness) + "\n")
            f.close()

            # um bei abbrüchen fortsetzen zu können wird die aktuelle population in ein Logfile geschrieben
            fileObject = open("output/zz_aktuelle_population.pickle",'wb') 
            pickle.dump(population,fileObject)   
            fileObject.close()

    
    # Logging der gesamten Population
    f= open("output/zz_log_genetischer_alg.txt", "a+", 1)
    f.write("\nAnfangspopulation:\n")
    for element in population:
        f.write(str(element.id) + "\t" + str(element.generation) + "\t" + str(element.genCode) 
        + "\t" + str(element.fitness) + "\n")
    f.write ("Ende Anfangspopulation\n\n")
    f.close()

# berechnet die Fitness für ein einzelnes Individuum
def berechneFitnessEinzel(element, myFitness):
    element.fitness = myFitness

# wählt die beiden Eltern für die nächste Rekombination aus
def selektiereEltern(population):
    # zuerst die Population sortieren. die geringsten Fitness-werte sind hier am Anfang
    population.sort(key=lambda x: x.fitness)
    
    rankSumme = 0
    n = len(population)
    # achtung bei division durch 0 (siehe unten) und wenn es nur 1 individuum gibt
    if n == 1:
        return population[0], population[0]

    if n == 2:
        return population[0], population[1]
    
    for i in range (1,n):
        rankSumme += 1/(n-2)
    
    # dann auswählen
    while True:
        randomNumber = uniform(0,rankSumme)
        for j in range(0,n-1):
            probability = (j+1)/(n*(n-1))
            if probability <= randomNumber:
                sel1 = population[j]

        randomNumber = uniform(0,rankSumme)
        for k in range(0,n-1):
            probability = (k+1)/(n*(n-1))
            if probability <= randomNumber:
                sel2 = population[k]
                
        try:
            sel1
        except NameError:
            continue

        try:
            sel2
        except NameError:
            continue

        # nur wenn nicht zweimal das gleiche Individuum ausgewählt wurde geht es weiter
        if sel1.id != sel2.id:
            break

    return sel1, sel2

# rekombiniert zwei Eltern und gibt ein Kind zurück
def rekombination(sel1, sel2, durchlaufNr):
    start = randint(0,6)
    ende = randint(start+1,7)

    newGenCode = []
    for i in range(0,start):
        newGenCode.append(sel1.genCode[i])

    for j in range(start,ende+1):
        newGenCode.append(sel2.genCode[j])

    for k in range(ende+1, 8):
        newGenCode.append(sel1.genCode[k])

    # Individuum mit dem entsprechenden genCode erzeugen
    returnValue = gene()
    returnValue.genCode = newGenCode
    returnValue.generation = durchlaufNr

    return returnValue

# Mutiert die Gene des neuen Individuums
def mutation(individuum):
    # Mutationsrate 1% für jedes Gen:
    for gen in individuum.genCode:
        if randint(1,100) == 1:
            gen = randint(1,22)
    return individuum