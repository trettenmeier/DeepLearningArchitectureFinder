from datetime import datetime
import pickle
from modelbuilder import Modelbuilder
import gene


#################### Parameter #####################
num_population      = 50        # Anzahl der Individuum in der Population
anzahlDurchlaeufe   = 50        # Anzahl der Generationen
epochen             = 10        # Wie lange wird jedes Individuum trainiert
single_modus        = False     # Einzelnes Individuum trainieren => siehe populationErzeugen()
fortsetzen          = False     # Abgebrochenen Lauf fortsetzen


###################### Ablauf ######################
if fortsetzen == False:
    # initiale Population erzeugen und Fitnesswerte berechnen
    population = gene.populationErzeugen(num_population, epochen, single_modus)
    gene.berechneFitness(population, epochen)
    durchlaufNr = 1

else:   # wenn forgesetzt wird:
    try:
        # pickle-file laden
        fileObject = open("output/zz_aktuelle_population.pickle",'rb')  
        population = pickle.load(fileObject)  
        fileObject.close()
    except:
        print("Keine Pickle-Datei gefunden")
        exit()

    # wenn die anfangspopulation noch nicht fertig berechnet war:
    gene.berechneFitness(population, epochen)

    # dann die höchste generation holen, das +1 ist die durchlaufnummer.
    durchlaufNr = 1
    nextID = 0

    for element in population:
        if element.generation > durchlaufNr:
            durchlaufNr = element.generation
        if element.id > nextID:
            nextID = element.id
    
    nextID = durchlaufNr + num_population - 1
    for i in range(0,nextID):
        temp = gene.gene()


# Evolution
while (durchlaufNr <= anzahlDurchlaeufe):
    sel1, sel2 = gene.selektiereEltern(population)
    neuerKandidat = gene.rekombination(sel1, sel2, durchlaufNr)
    neuerKandidat = gene.mutation(neuerKandidat)

    # neues Individuum trainieren
    model_neu = Modelbuilder(neuerKandidat)
    model_neu.add_GenCode_To_Model()
    myFitness = model_neu.finalize_Model(epochen)

    # dann Fitness ermitteln
    gene.berechneFitnessEinzel(neuerKandidat, myFitness)
    
    # neues Individuum wird der Population hinzugefügt
    population.append(neuerKandidat)
    
    # das schwächste Individuum fliegt raus
    population.sort(key=lambda x: x.fitness, reverse=True)
    population.pop() 

    # Logging
    f= open("output/zz_log_genetischer_alg.txt", "a+", 1)
    f.write(str(neuerKandidat.id) + "\t" + str(neuerKandidat.generation) + "\t" 
    + str(neuerKandidat.genCode) + "\t" + str(neuerKandidat.fitness) + "\tbestes individuum: " +
    str(population[0].id) + "\t mit fitness: " + str(population[0].fitness) + "\n")
    f.close()

    # um bei abbrüchen fortsetzen zu können wird die aktuelle population in ein Logfile geschrieben
    fileObject = open("output/zz_aktuelle_population.pickle",'wb') 
    pickle.dump(population,fileObject)   
    fileObject.close()

    durchlaufNr += 1


#################### Fertig ####################
winner = population[0]

print("\n\nand the winner is....")
print("id: " +str(winner.id))
print("generation: " + str(winner.generation))
print(winner.genCode)
print(winner.fitness)

f= open("output/zz_log_genetischer_alg.txt", "a+", 1)
f.write("and the winner is....\n")
f.write("id: " +str(winner.id) + "\n")
f.write("generation: " + str(winner.generation) + "\n")
f.write("genCode: " + str(winner.genCode) + "\n")
f.write("fitness: " + str(winner.fitness) + "\n")
f.close()