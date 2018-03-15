import models
import visualize


# Find some interests through figure

visualize.plot0()
visualize.plot1()
visualize.plot2()
visualize.plot3()

# Start mining
models.init()
# Naive Bayes
models.bayes()
# K-nn model
models.knn()
# Decision tree
models.decision_tree()
# Random forest
models.random_forest()

print("\n")

models.compare()
# Random test
models.final_test()

